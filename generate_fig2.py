"""
generate_fig2.py
================
Run in Jupyter after evaluate.py to generate Figure 2:
Fisher-information fingerprint reconstruction across all 4 geometry classes.

Usage:
    %run generate_fig2.py --checkpoint outputs/best_model.pt --dataset dataset.h5 --out_dir outputs/
"""
import argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='outputs/best_model.pt')
parser.add_argument('--dataset',    default='dataset.h5')
parser.add_argument('--out_dir',    default='outputs/')
parser.add_argument('--n_mc',       type=int, default=50)
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
sys.path.insert(0, '.')
from model import FisherCNN

device = torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('cpu')
print(f"Device: {device}")

ckpt = torch.load(args.checkpoint, map_location=device)

# Detect checkpoint format from keys
print("Checkpoint keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))

if isinstance(ckpt, dict):
    # Get training args if saved
    train_args = ckpt.get('args', ckpt.get('train_args', None))
    grid_size  = getattr(train_args, 'grid_size', 64) if train_args else 64
    M          = getattr(train_args, 'M', 6) if train_args else 6
    model = FisherCNN(grid_size=grid_size, descriptor_dim=7, M=M, dropout_rate=0.1)
    # Try all known state dict key names
    for key in ['model_state', 'model_state_dict', 'state_dict']:
        if key in ckpt:
            model.load_state_dict(ckpt[key])
            print(f"Loaded state dict from key: '{key}'")
            break
    else:
        # last resort: ckpt itself is the state dict
        model.load_state_dict(ckpt)
        print("Loaded state dict directly from checkpoint")
else:
    model = ckpt
    grid_size, M = 64, 6
    print("Loaded model object directly")
model.to(device)
model.eval()
print("Model loaded")

# ── Load dataset ───────────────────────────────────────────────────────────────
with h5py.File(args.dataset, 'r') as f:
    geom_ids    = f['geometry_id'][:]
    cos_true_all= f['cos_coeffs'][:]
    sin_true_all= f['sin_coeffs'][:]
    AF_true_all = f['AF'][:]
    images_all  = f['images'][:]
    descs_all   = f['descriptor'][:]
    snr_all     = f['snr_db'][:]

angles = np.linspace(0, 360, 72, endpoint=False)

geom_names = {
    0: r'Isotropic $O(2)$',
    1: r'Trigonal $C_3$',
    2: r'Triangular $C_{3v}$',
    3: r'Square $C_4$'
}

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.patch.set_facecolor('white')

for col, gid in enumerate([0, 1, 2, 3]):
    # Pick sample with largest A_F (most visually interesting)
    mask = (geom_ids == gid) & (snr_all > 12) & (snr_all < 18)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        idxs = np.where(geom_ids == gid)[0]
    chosen = idxs[np.argmax(AF_true_all[idxs])]

    img  = torch.tensor(images_all[chosen:chosen+1],
                        dtype=torch.float32).to(device)
    desc = torch.tensor(descs_all[chosen:chosen+1],
                        dtype=torch.float32).to(device)
    ct   = cos_true_all[chosen]
    st   = sin_true_all[chosen]
    af_t = float(AF_true_all[chosen])
    snr  = float(snr_all[chosen])

    # Point prediction
    with torch.no_grad():
        cos_p, sin_p, AF_p = model(img, desc)
    cos_p = cos_p[0].cpu().numpy()
    sin_p = sin_p[0].cpu().numpy()
    af_p  = float(AF_p.item())

    def fourier_to_profile(cc, sc):
        IF = np.zeros(72)
        for m in range(M):
            IF += cc[m] * np.cos(m * np.deg2rad(angles))
            IF += sc[m] * np.sin(m * np.deg2rad(angles))
        return IF

    IF_true_full = fourier_to_profile(ct, st)

    # MC-dropout uncertainty
    model.train()
    mc_profiles = []
    with torch.no_grad():
        for _ in range(args.n_mc):
            cp, sp, _ = model(img, desc)
            mc_profiles.append(fourier_to_profile(
                cp[0].cpu().numpy(), sp[0].cpu().numpy()))
    model.eval()
    mc_arr  = np.array(mc_profiles)
    mc_mean = mc_arr.mean(axis=0)
    mc_std  = mc_arr.std(axis=0)

    # For high-symmetry classes (O(2) and C4) A_F is near zero;
    # Fourier reconstruction is dominated by numerical noise.
    # Show true profile normalized to its own range for clarity.
    y_range = IF_true_full.max() - IF_true_full.min()
    low_af  = (af_t < 0.001)   # isotropic or square — essentially flat

    # ── Top row: I_F(theta) profiles ─────────────────────────────────────────
    ax = axes[0, col]

    if low_af:
        # High-symmetry class: true profile is flat, show normalised to [-1,1]
        # to avoid matplotlib's offset notation
        true_mean = IF_true_full.mean()
        IF_plot   = IF_true_full - true_mean   # centre at zero
        flat_val  = mc_mean.mean() - true_mean
        flat_std  = mc_std.mean()
        ax.plot(angles, IF_plot, lw=2.2, color='steelblue',
                label=r'True $I_F(\theta)$')
        ax.axhline(flat_val, lw=1.6, color='tomato', linestyle='--',
                   label='CNN (mean)')
        ax.axhspan(flat_val - flat_std, flat_val + flat_std,
                   alpha=0.20, color='tomato', label=r'$\pm1\sigma$ MC')
        ax.set_ylim(-0.005, 0.005)
        ax.set_ylabel(r'$I_F(\theta) - \langle I_F\rangle$ (norm.)', fontsize=9)
        ax.text(0.5, 0.92, 'Flat: high symmetry', transform=ax.transAxes,
                ha='center', fontsize=7.5, color='gray', style='italic')
    else:
        ax.plot(angles, IF_true_full, lw=2.2, color='steelblue',
                label=r'True $I_F(\theta)$')
        ax.plot(angles, mc_mean, lw=1.6, color='tomato',
                linestyle='--', label='CNN (mean)')
        ax.fill_between(angles, mc_mean - mc_std, mc_mean + mc_std,
                        alpha=0.25, color='tomato', label=r'$\pm1\sigma$ MC')
    ax.set_title(f'{geom_names[gid]}\n'
                 r'$A_F^{\rm true}=$' + f'{af_t:.4f}  '
                 r'$A_F^{\rm CNN}=$'  + f'{af_p:.4f}',
                 fontsize=8.5)
    ax.set_xlabel(r'$\theta$ (°)', fontsize=9)
    if col == 0 and not low_af:
        ax.set_ylabel(r'$I_F(\theta)$ (norm.)', fontsize=9)
    if col == 0:
        ax.legend(fontsize=7.5, loc='upper right')
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlim(0, 360)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Bottom row: noisy input image ─────────────────────────────────────────
    ax2 = axes[1, col]
    ax2.imshow(images_all[chosen, 0], cmap='viridis', origin='lower',
               vmin=0, vmax=1, interpolation='bilinear')
    ax2.set_title(f'Input STM image  (SNR = {snr:.0f} dB)', fontsize=8.5)
    ax2.axis('off')

fig.suptitle(
    r'Fisher-information fingerprint reconstruction across symmetry classes'
    '\n'
    r'True $I_F(\theta)$ (blue) vs CNN prediction (red dashed) $\pm1\sigma$ MC-dropout',
    fontsize=10, fontweight='bold')
plt.tight_layout()
out_path = out_dir / 'fig2_fingerprints.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved Fig 2 -> {out_path}")
