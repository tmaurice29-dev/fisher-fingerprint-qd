# Universal AI Framework for Fisher-Information Fingerprints in Quantum Dot Imaging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

> **Paper:** *A Universal AI Framework for Extracting Fisher-Information Fingerprints from Noisy Quantum Confinement Images*  
> Maurice Tiotsop, University of Louisville School of Medicine (2026)

---

## Overview

This repository provides the full code, dataset generation pipeline, and trained model for extracting **Fisher-information fingerprints** from noisy scanning-tunneling microscopy (STM)-like images of semiconductor quantum dots (QDs).

The framework handles four confinement symmetry classes — all derived from a single **anisotropic harmonic potential family**:

$$V_n(r,\theta;\alpha) = \tfrac{1}{2}m^*\omega_0^2\,r^2\bigl[1 + \alpha\cos(n\theta)\bigr]$$

| Class | Symmetry | $n$ | $\alpha$ range |
|-------|----------|-----|---------------|
| 0 — Isotropic | $O(2)$ | — | $0$ (fixed) |
| 1 — Trigonal | $C_3$ | 3 | $0.05$–$0.60$ |
| 2 — Triangular | $C_{3v}$ | 3 | $0.70$–$0.99$ |
| 3 — Square | $C_4$ | 4 | $0.05$–$0.80$ |

### Key results

| Metric | Value |
|--------|-------|
| Fourier MSE (test set) | $1.99 \times 10^{-7}$ |
| Pearson $r$ ($A_F$, all) | $0.9985$ |
| $R^2$ ($A_F$) | $0.998$ |
| CNN vs. classical FD | $> 5 \times 10^6 \times$ improvement |
| Uncertainty calibration (Spearman $\rho$) | $0.890$ |
| QFI bound $\delta R_0$ | $\geq 0.67$ nm |
| Parameters | $877{,}869$ |
| Training time (Apple M1) | $4.15$ h |

---

## Repository structure

```
.
├── dataset.py          # Dataset generation: potential solver + image degradation
├── model.py            # Dual-stream physics-conditioned CNN architecture
├── train.py            # Training loop with MC-dropout and early stopping
├── evaluate.py         # Evaluation suite: all tables + figures 3–5
├── generate_fig2.py    # Figure 2: I_F(θ) reconstruction across symmetry classes
├── run_full.sh         # One-command clean rerun script
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/fisher-fingerprint-qd.git
cd fisher-fingerprint-qd

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.1+, CUDA optional (MPS supported for Apple Silicon).

---

## Quickstart

### Option A — Full pipeline from scratch (~4 hours on M1)

```bash
bash run_full.sh
```

This will:
1. Delete any old `dataset.h5`
2. Generate 80,000 samples across 4 geometry classes
3. Train for 120 epochs with early stopping
4. Save `outputs/best_model.pt` and all figures

### Option B — Use the pretrained model

Download `best_model.pt` from [Releases](https://github.com/YOUR-USERNAME/fisher-fingerprint-qd/releases) and place it in `outputs/`, then:

```bash
# Generate the dataset only (needed for evaluation)
python train.py --n_samples 80000 --grid_size 64 --epochs 0 --out_dir outputs/

# Run evaluation and generate all figures
python evaluate.py \
    --checkpoint outputs/best_model.pt \
    --dataset dataset.h5 \
    --out_dir outputs/ \
    --n_mc 50

# Generate Figure 2
python generate_fig2.py \
    --checkpoint outputs/best_model.pt \
    --dataset dataset.h5 \
    --out_dir outputs/
```

---

## Dataset

The dataset is generated synthetically from the forward-imaging model:

1. **Physics solver** — finite-difference Schrödinger equation on a $64 \times 64$ grid (Lanczos/ARPACK)
2. **Fisher fingerprints** — sector-sum angular mass profile, Fourier decomposition ($M=6$ modes)
3. **Image degradation** — Gaussian blur ($\sigma_b \in [1,4]$ px), Gaussian + Poisson noise (SNR $\in [5,30]$ dB), multiplicative disorder ($\alpha_{\rm dis} \in [0, 0.3]$)

Each sample stores:

| Field | Shape | Description |
|-------|-------|-------------|
| `images` | $(N, 1, 64, 64)$ | Degraded STM-like images |
| `cos_coeffs` | $(N, 6)$ | Fourier cosine coefficients of $I_F(\theta)$ |
| `sin_coeffs` | $(N, 6)$ | Fourier sine coefficients |
| `AF` | $(N,)$ | Anisotropy index $A_F \in [0,1]$ |
| `descriptor` | $(N, 7)$ | Potential descriptor $\boldsymbol{\theta}$ |
| `geometry_id` | $(N,)$ | Class label $0$–$3$ |
| `snr_db` | $(N,)$ | SNR used for degradation |
| `sigma_b` | $(N,)$ | Blur $\sigma$ used |

Physical parameters (GaAs, $m^* = 0.067\,m_e$):

| Parameter | Range | Reference |
|-----------|-------|-----------|
| $\hbar\omega_0$ | $[2, 20]$ meV | Climente et al. PRB 2007 |
| $R_{\rm dot}$ | $[5, 30]$ nm | Reimann & Manninen RMP 2002 |
| $\sigma_b$ | $[1, 4]$ px | Feenstra PRB 1994 |
| SNR | $[5, 30]$ dB | Ugeda et al. Nat. Phys. 2014 |

---

## Model architecture

```
Input: noisy image (1×64×64) + descriptor (7,)
         │                        │
   IMAGE STREAM              POTENTIAL STREAM
   4×ConvBlock                 2-layer MLP
   (1→32→64→128→128)           (7→64→128)
   + Linear(256)               + Dropout(0.1)
         │                        │
         └──── Cross-Attention Fusion ────┘
                     │
               Latent (128-d)
               Linear + ReLU
              /      |       \
        Head cos   Head sin  Head AF
        (Softplus) (Linear)  (Sigmoid)
```

**Total parameters:** 877,869  
**MC-dropout rate:** 0.1 (active at inference for uncertainty quantification)

---

## Training

```bash
python train.py \
    --n_samples 80000 \
    --grid_size 64 \
    --epochs 120 \
    --batch_size 64 \
    --patience 20 \
    --out_dir outputs/
```

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr $= 10^{-3}$, cosine decay |
| Loss weights | $\lambda_F=1.0$, $\lambda_A=0.5$, $\lambda_C=0.1$ |
| Batch size | 64 |
| Early stopping patience | 20 epochs |
| Train / Val / Test split | 80 / 10 / 10 |

---

## Reproducing paper figures

| Figure | Command | Output |
|--------|---------|--------|
| Fig. 1 (architecture) | Included in repo | `fig1_architecture.png` |
| Fig. 2 (fingerprints) | `python generate_fig2.py ...` | `outputs/fig2_fingerprints.png` |
| Fig. 3 (density maps) | `python evaluate.py ...` | `outputs/fig3_interpolation.png` |
| Fig. 4 (MSE vs SNR) | `python evaluate.py ...` | `outputs/fig4_mse_vs_snr.png` |
| Fig. 5 (uncertainty) | `python evaluate.py ...` | `outputs/fig5_uncertainty_calibration.png` |

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{tiotsop2026fisher,
  title   = {A Universal {AI} Framework for Extracting {Fisher}-Information
             Fingerprints from Noisy Quantum Confinement Images},
  author  = {Tiotsop, Maurice},
  journal = {Physical Review A},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Maurice Tiotsop**  
University of Louisville School of Medicine  
Louisville, KY 40202, USA  
<!-- Add email if desired -->
