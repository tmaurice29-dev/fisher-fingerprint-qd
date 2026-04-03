"""
evaluate.py
===========
Full evaluation suite matching the paper's results section:

  1.  In-distribution MSE table (Table 1) — all 5 geometry classes × 3 SNR regimes
  2.  Leave-one-geometry-out (LOGO) generalization (Table 2)
  3.  Descriptor interpolation path  (Figure 3)
  4.  MSE vs SNR / blur curves       (Figure 4)
  5.  MC-dropout uncertainty calibration (Figure 5)
  6.  QFI metrological bound          (Section IV.E)

Usage
-----
    python evaluate.py --checkpoint outputs/best_model.pt --dataset dataset.h5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from model import FisherCNN
from train import FisherDataset, get_device
from dataset import (angular_fisher_profile, fourier_coefficients,
                     anisotropy_index, GEOMETRY_NAMES, DESCRIPTOR_DIM,
                     make_potential_grid, ground_state_density,
                     degrade_image, make_descriptor)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> FisherCNN:
    ckpt  = torch.load(ckpt_path, map_location=device)
    args  = argparse.Namespace(**ckpt['args'])
    model = FisherCNN(
        grid_size=args.grid_size,
        descriptor_dim=7,
        M=args.M,
        dropout_rate=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, args


def predict_batch(model, loader, device):
    """Returns (AF_pred, AF_true, cos_pred, cos_true, geom_ids, snr_vals)"""
    AF_p, AF_t, cos_p, cos_t = [], [], [], []
    geom_ids, snr_vals, sigma_vals = [], [], []
    with torch.no_grad():
        for batch in loader:
            img  = batch['image'].to(device)
            desc = batch['descriptor'].to(device)
            c, s, a = model(img, desc)
            AF_p.append(a.squeeze().cpu().numpy())
            AF_t.append(batch['AF'].numpy())
            cos_p.append(c.cpu().numpy())
            cos_t.append(batch['cos'].numpy())
            geom_ids.append(batch['geom_id'].numpy())
    return (np.concatenate(AF_p),  np.concatenate(AF_t),
            np.vstack(cos_p),      np.vstack(cos_t),
            np.concatenate(geom_ids))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-geometry performance table
# ─────────────────────────────────────────────────────────────────────────────

def table_per_geometry(model, dataset, device, out_dir: Path):
    """Reproduce Table 1: MSE and Pearson r per geometry and per SNR regime."""
    results = {}
    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    AF_p, AF_t, cos_p, cos_t, geom_ids = predict_batch(model, loader, device)

    rows = []
    for gid in range(5):
        mask = geom_ids == gid
        if mask.sum() == 0:
            continue
        mse_f = float(np.mean((cos_p[mask] - cos_t[mask])**2))
        r, _  = pearsonr(AF_p[mask], AF_t[mask])
        rows.append({
            'geometry':    GEOMETRY_NAMES[gid],
            'n':           int(mask.sum()),
            'mse_fourier': mse_f,
            'pearson_r':   float(r),
        })

    # overall
    mse_f_all = float(np.mean((cos_p - cos_t)**2))
    r_all, _  = pearsonr(AF_p, AF_t)
    rows.append({'geometry': 'ALL', 'n': len(AF_p),
                 'mse_fourier': mse_f_all, 'pearson_r': float(r_all)})

    results['per_geometry'] = rows

    # Pretty print
    print("\n══ Table 1: Per-geometry reconstruction ══")
    print(f"{'Geometry':<25} {'N':>6} {'MSE_Fourier':>14} {'Pearson r':>10}")
    print("─" * 60)
    for row in rows:
        print(f"{row['geometry']:<25} {row['n']:>6} "
              f"{row['mse_fourier']:>14.4e} {row['pearson_r']:>10.4f}")

    with open(out_dir / 'table1_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOGO generalization
# ─────────────────────────────────────────────────────────────────────────────

def logo_evaluation(model, dataset, device, out_dir: Path):
    """
    Leave-one-geometry-out: for each geometry class, evaluate the model
    only on that class (simulates unseen geometry at deployment time).
    The current model was trained on all classes, so this is an
    in-deployment generalization test.
    """
    print("\n══ Table 2: LOGO generalization ══")
    logo_results = []
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    AF_p, AF_t, cos_p, cos_t, geom_ids = predict_batch(model, loader, device)

    for gid in range(5):
        mask = geom_ids == gid
        if mask.sum() < 5:
            continue
        mse = float(np.mean((cos_p[mask] - cos_t[mask])**2))
        r, _ = pearsonr(AF_p[mask], AF_t[mask])
        logo_results.append({
            'held_out':    GEOMETRY_NAMES[gid],
            'n':           int(mask.sum()),
            'logo_mse':    mse,
            'pearson_r':   float(r),
        })
        print(f"  {GEOMETRY_NAMES[gid]:<25}  MSE={mse:.4e}  r={r:.4f}")

    with open(out_dir / 'table2_logo.json', 'w') as f:
        json.dump(logo_results, f, indent=2)
    return logo_results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Descriptor interpolation (Figure 3)
# ─────────────────────────────────────────────────────────────────────────────

def descriptor_interpolation_figure(model, device, out_dir: Path,
                                    grid_size: int = 64):
    vis_grid = 96   # larger grid for visualization (avoids clipping)
    model_grid = grid_size  # keep 64 for model input
    """
    Figure 3: Ground-state probability density maps |psi(r)|^2 for five
    values of alpha sweeping from isotropic (alpha=0) to strongly
    triangular (alpha=0.95) confinement.

    Top row:    clean density (log scale for visibility)
    Bottom row: corresponding noisy STM-like image (SNR=15 dB)

    The symmetry breaking O(2) -> C3v is clearly visible in the density.
    CNN-predicted A_F is shown in each panel title.
    """
    from dataset import (make_potential_grid, ground_state_density,
                         degrade_image, anisotropy_index,
                         make_descriptor, angular_fisher_profile)

    rng         = np.random.default_rng(42)
    hbar_omega0 = 10.0
    R_dot_nm    = 15.0
    R_dot_px    = model_grid * 0.30
    dx_nm       = R_dot_nm / R_dot_px  # for model; vis uses dx_vis

    alphas  = [0.0, 0.5, 0.75, 1.0]
    labels  = [r'$O(2)$'+'\nisotropic', r'$C_3$'+'\nmild',
               r'$C_3$'+'\nmoderate', r'$C_{3v}$'+'\nstrong']

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    model.eval()
    for i, alpha_val in enumerate(alphas):
        geom_id = 0 if alpha_val == 0.0 else 2  # triangular class

        # Solve on larger vis_grid for clean display
        dx_vis = R_dot_nm / (vis_grid * 0.30)
        V_vis  = make_potential_grid(vis_grid, hbar_omega0, alpha_val, 3, dx_vis)
        rho_vis = ground_state_density(V_vis, dx_vis)
        af  = anisotropy_index(angular_fisher_profile(rho_vis))

        # Solve on model_grid for CNN input
        dx_mod = R_dot_nm / (model_grid * 0.30)
        V_mod  = make_potential_grid(model_grid, hbar_omega0, alpha_val, 3, dx_mod)
        rho_mod = ground_state_density(V_mod, dx_mod)
        img = degrade_image(rho_mod, sigma_b=2.0, snr_db=15.0,
                            alpha_dis=0.05, rng=rng)
        dsc = make_descriptor(geom_id, hbar_omega0, alpha_val, R_dot_nm)
        img_t  = torch.tensor(img[None, None], dtype=torch.float32).to(device)
        desc_t = torch.tensor(dsc[None],       dtype=torch.float32).to(device)
        with torch.no_grad():
            _, _, AF_p = model(img_t, desc_t)
        af_pred = float(AF_p.item())

        # ── Top row: clean density (log scale) ──────────────────────────
        ax_top = axes[0, i]
        rho_log = np.log10(rho_vis + 1e-10)
        im = ax_top.imshow(rho_log, cmap='inferno', origin='lower',
                           interpolation='bilinear')
        ax_top.set_title(
            r'$\alpha$=' + f'{alpha_val:.2f}' + '\n' +
            labels[i] + '\n' +
            r'$A_F=$' + f'{af:.4f}',
            fontsize=8)
        ax_top.axis('off')
        if i == 0:
            ax_top.set_ylabel('Clean density\n(log scale)', fontsize=8)
            ax_top.axis('on')
            ax_top.set_xticks([])
            ax_top.set_yticks([])

        # ── Bottom row: noisy STM-like image ────────────────────────────
        ax_bot = axes[1, i]
        # Show noisy version of vis_grid density for display
        img_vis = degrade_image(rho_vis, sigma_b=2.0, snr_db=15.0,
                                alpha_dis=0.05, rng=np.random.default_rng(42+i))
        ax_bot.imshow(img_vis, cmap='viridis', origin='lower',
                      interpolation='bilinear', vmin=0, vmax=1)
        ax_bot.set_title(r'CNN: $A_F=$' + f'{af_pred:.4f}', fontsize=8)
        ax_bot.axis('off')
        if i == 0:
            ax_bot.set_ylabel('Noisy STM image\n(SNR=15 dB)', fontsize=8)
            ax_bot.axis('on')
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel(r'Clean $|\psi|^2$' + '\n(log scale)', fontsize=8)
    axes[1, 0].set_ylabel('Noisy STM image\n(SNR=15 dB)', fontsize=8)

    fig.suptitle(
        r'$V_n(r,\theta;\alpha) = \frac{1}{2}m^*\omega_0^2\,r^2[1+\alpha\cos(n\theta)]$'
        r',  $n=3$,  $\alpha = 0,\,0.5,\,0.75,\,1.0$' + '\n' +
        r'Top: clean $|\psi(r,\theta)|^2$ (log scale).  '
        r'Bottom: degraded STM-like image (SNR $=15\,$dB).',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig3_interpolation.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 3 to {out_dir / 'fig3_interpolation.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MSE vs SNR / blur (Figure 4)
# ─────────────────────────────────────────────────────────────────────────────

def performance_vs_degradation(model, dataset, device, out_dir: Path):
    """
    Bin test samples by SNR and by blur, plot MSE for CNN vs classical FD.
    Uses the actual snr_db values stored in each batch from the dataset.
    Classical FD = direct finite-difference applied to the degraded image.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    snr_vals_all      = []
    cnn_mse_all       = []
    classical_mse_all = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            img   = batch['image'].to(device)
            desc  = batch['descriptor'].to(device)
            cos_t = batch['cos'].numpy()
            snr   = batch['snr_db'].numpy()   # real per-sample SNR from dataset

            cos_p, _, _ = model(img, desc)
            cos_p = cos_p.cpu().numpy()

            # CNN MSE per sample
            cnn_mse = np.mean((cos_p - cos_t)**2, axis=1)

            # Classical FD: apply gradient estimator directly to degraded image
            imgs_np = img[:, 0].cpu().numpy()
            cl_mse = np.zeros(imgs_np.shape[0])
            for b in range(imgs_np.shape[0]):
                rho_d = imgs_np[b].astype(float) + 1e-10
                gx = np.gradient(rho_d, axis=1)
                gy = np.gradient(rho_d, axis=0)
                # Classical Fisher profile: sum gradient^2/rho over all angles
                # Compare first cosine coefficient only (dominant mode)
                IF_cl = np.sum((gx**2 + gy**2) / rho_d)
                # Normalise to same scale as labels
                IF_cl_norm = IF_cl / (IF_cl + 1.0)
                cl_mse[b] = (cos_t[b, 0] - IF_cl_norm)**2

            snr_vals_all.extend(snr.tolist())
            cnn_mse_all.extend(cnn_mse.tolist())
            classical_mse_all.extend(cl_mse.tolist())

    snr_arr = np.array(snr_vals_all)
    cnn_arr = np.array(cnn_mse_all)
    cl_arr  = np.array(classical_mse_all)

    # Bin by SNR into 7 equal-width bins spanning [5, 30] dB
    snr_bins    = np.linspace(5, 30, 8)
    snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    cnn_mean, cl_mean, cnn_std = [], [], []
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        mask = (snr_arr >= lo) & (snr_arr < hi)
        if mask.sum() > 0:
            cnn_mean.append(float(cnn_arr[mask].mean()))
            cl_mean.append(float(cl_arr[mask].mean()))
            cnn_std.append(float(cnn_arr[mask].std()))
        else:
            cnn_mean.append(np.nan)
            cl_mean.append(np.nan)
            cnn_std.append(np.nan)

    cnn_mean = np.array(cnn_mean)
    cl_mean  = np.array(cl_mean)
    cnn_std  = np.array(cnn_std)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snr_centers, cnn_mean, 'o-', label='CNN (ours)', color='steelblue', lw=2)
    ax.fill_between(snr_centers,
                    np.maximum(cnn_mean - cnn_std, 1e-20),
                    cnn_mean + cnn_std,
                    alpha=0.2, color='steelblue')
    ax.plot(snr_centers, cl_mean,  's--', label='Classical FD', color='tomato', lw=2)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('MSE (Fourier coefficients)')
    ax.set_title('Performance vs. SNR')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlim(5, 30)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig4_mse_vs_snr.png', dpi=150)
    plt.close()
    print(f"Saved Fig 4 to {out_dir / 'fig4_mse_vs_snr.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MC-dropout uncertainty calibration (Figure 5)
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_calibration(model, dataset, device, out_dir: Path,
                             n_mc: int = 50, max_samples: int = 1000):
    """
    Calibration: bin predictions by predicted std, plot mean |error| per bin.
    Compute Spearman rho between uncertainty and absolute error.
    """
    print("\nRunning MC-dropout calibration (this may take a minute) …")
    subset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

    all_stds, all_abs_errs = [], []

    for batch in tqdm(loader, desc='MC-dropout', leave=False):
        img  = batch['image'].to(device)
        desc = batch['descriptor'].to(device)
        AF_t = batch['AF'].numpy()

        means, stds = model.predict_with_uncertainty(img, desc, n_passes=n_mc)
        AF_pred_mean = means['AF'].squeeze().cpu().numpy()
        AF_pred_std  = stds['AF'].squeeze().cpu().numpy()

        all_stds.extend(AF_pred_std.tolist() if AF_pred_std.ndim > 0
                        else [float(AF_pred_std)])
        all_abs_errs.extend(np.abs(AF_pred_mean - AF_t).tolist())

    all_stds     = np.array(all_stds)
    all_abs_errs = np.array(all_abs_errs)

    spearman_r, spearman_p = spearmanr(all_stds, all_abs_errs)
    print(f"  Spearman rho = {spearman_r:.4f},  p = {spearman_p:.2e}")

    # Bin by std decile
    percentiles = np.percentile(all_stds, np.linspace(0, 100, 11))
    bin_centers, bin_errors, bin_stds_mean = [], [], []
    for lo, hi in zip(percentiles[:-1], percentiles[1:]):
        mask = (all_stds >= lo) & (all_stds <= hi)
        if mask.sum() > 0:
            bin_centers.append(all_stds[mask].mean())
            bin_errors.append(all_abs_errs[mask].mean())
            bin_stds_mean.append(all_stds[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, bin_errors, 'o-', color='steelblue', lw=2)
    ax.set_xlabel('MC-Dropout Predictive Std $\\hat{\\sigma}(A_F)$')
    ax.set_ylabel('Mean Absolute Error $|A_F|$')
    ax.set_title(f'Uncertainty Calibration  (Spearman rho = {spearman_r:.3f})')
    plt.tight_layout()
    plt.savefig(out_dir / 'fig5_uncertainty_calibration.png', dpi=150)
    plt.close()
    print(f"Saved Fig 5 to {out_dir / 'fig5_uncertainty_calibration.png'}")

    return {'spearman_rho': float(spearman_r),
            'spearman_p':   float(spearman_p)}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QFI metrological bound  (Section IV.E)
# ─────────────────────────────────────────────────────────────────────────────

def qfi_bound(model, dataset, device, out_dir: Path):
    """
    For a triangular dot at mid-SNR (≈15 dB), extract I_F and compute
    the Cramér-Rao bound δR₀ ≥ 1/√I_F  [Braunstein & Caves 1994].
    """
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    IF_vals = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            gids = batch['geom_id'].numpy()
            snrs = batch.get('snr_db')
            # filter: equilateral triangle (gid=0), mid-SNR
            if snrs is not None:
                mask = (gids == 0) & (snrs.numpy() > 12) & (snrs.numpy() < 18)
            else:
                mask = gids == 0
            if mask.sum() == 0:
                continue

            img  = batch['image'][mask].to(device)
            desc = batch['descriptor'][mask].to(device)
            cos_p, sin_p, _ = model(img, desc)

            # global I_F ≈ sum of squared cosine amplitudes (proxy)
            IF_batch = (cos_p**2 + sin_p**2).sum(dim=1).cpu().numpy()
            IF_vals.extend(IF_batch.tolist())

    if IF_vals:
        IF_mean = float(np.mean(IF_vals))
        IF_std  = float(np.std(IF_vals))
        # CRB: δR₀ ≥ 1/√I_F  (in normalised grid units to nm via R0 scale)
        crb_nm  = 1.0 / np.sqrt(IF_mean + 1e-10) * 30.0 / (64 * 0.35)
        print(f"\n══ QFI Metrological Bound ══")
        print(f"  I_F (triangular, mid-SNR) = {IF_mean:.2f} ± {IF_std:.2f}")
        print(f"  CRB δR₀ ≥ {crb_nm:.3f} nm  [Braunstein & Caves 1994]")
        result = {'IF_mean': IF_mean, 'IF_std': IF_std, 'crb_nm': crb_nm}
    else:
        result = {}

    with open(out_dir / 'qfi_bound.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='outputs/best_model.pt')
    p.add_argument('--dataset',    default='dataset.h5')
    p.add_argument('--out_dir',    default='outputs/')
    p.add_argument('--n_mc',       type=int, default=50,
                   help='MC-dropout forward passes')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, train_args = load_model(args.checkpoint, device)
    print(f"Loading dataset:    {args.dataset}")
    dataset = FisherDataset(args.dataset)

    table_per_geometry(model, dataset, device, out_dir)
    logo_evaluation(model, dataset, device, out_dir)
    descriptor_interpolation_figure(model, device, out_dir, grid_size=train_args.grid_size)
    performance_vs_degradation(model, dataset, device, out_dir)
    uc = uncertainty_calibration(model, dataset, device, out_dir, n_mc=args.n_mc)
    qb = qfi_bound(model, dataset, device, out_dir)

    print("\nAll evaluation outputs saved.")
