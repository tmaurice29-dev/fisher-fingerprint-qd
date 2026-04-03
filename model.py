"""
model.py
========
Physics-conditioned dual-stream CNN for Fisher-information fingerprint
extraction from noisy quantum dot images.

Architecture
------------
  Stream 1 (Image)     : 4 × ConvBlock → GlobalAvgPool → feature vector
  Stream 2 (Potential) : 2-layer MLP   → geometry embedding
  Fusion               : Concat + Cross-attention layer
  Heads (×3)           : FC → cos_coeffs | sin_coeffs | AF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class CrossAttentionFusion(nn.Module):
    """
    Single-head cross-attention: image features attend to geometry embedding.
    Query = image,  Key = Value = geometry embedding.
    """
    def __init__(self, img_dim: int, geom_dim: int, out_dim: int):
        super().__init__()
        self.Wq = nn.Linear(img_dim,  out_dim)
        self.Wk = nn.Linear(geom_dim, out_dim)
        self.Wv = nn.Linear(geom_dim, out_dim)
        self.scale = out_dim ** -0.5

    def forward(self, img_feat: torch.Tensor,
                geom_feat: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(img_feat)                   # (B, out_dim)
        K = self.Wk(geom_feat)                  # (B, out_dim)
        V = self.Wv(geom_feat)                  # (B, out_dim)
        # dot-product attention (single token)
        attn = (Q * K).sum(dim=-1, keepdim=True) * self.scale
        attn = torch.sigmoid(attn)              # soft gate
        fused = Q + attn * V                    # residual fusion
        return fused                            # (B, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class FisherCNN(nn.Module):
    """
    Dual-stream physics-conditioned CNN.

    Parameters
    ----------
    grid_size    : int   - input image spatial size (e.g. 64)
    descriptor_dim : int - length of Fourier boundary descriptor (2K+1)
    M            : int   - number of Fourier modes to predict
    dropout_rate : float - MC-dropout rate (applied at inference too)
    """

    def __init__(
        self,
        grid_size:      int   = 64,
        descriptor_dim: int   = 7,    # 4 (one-hot) + 3 (ω₀, α, R_dot)
        M:              int   = 6,
        dropout_rate:   float = 0.1,
    ):
        super().__init__()
        self.M            = M
        self.dropout_rate = dropout_rate

        # ── Stream 1: Image CNN ──────────────────────────────────────────────
        # 4 × ConvBlock halves spatial dims each time
        # 64 → 32 → 16 → 8 → 4
        self.img_stream = nn.Sequential(
            ConvBlock(1,   32),   # 64 → 32
            ConvBlock(32,  64),   # 32 → 16
            ConvBlock(64,  128),  # 16 → 8
            ConvBlock(128, 128),  # 8  → 4
        )
        # spatial size after 4 pools: grid_size // 16
        spatial_out  = grid_size // 16
        img_feat_dim = 128 * spatial_out * spatial_out

        self.img_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # ── Stream 2: Potential MLP ──────────────────────────────────────────
        self.pot_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # ── Fusion: cross-attention ──────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            img_dim=256, geom_dim=128, out_dim=128
        )

        # shared latent after fusion
        self.latent = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # ── Output heads ─────────────────────────────────────────────────────
        # Head 1: cosine Fourier coefficients
        self.head_cos = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, M),
            nn.Softplus(),   # enforce nonnegativity [Frieden 1999]
        )
        # Head 2: sine Fourier coefficients
        self.head_sin = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, M),
        )
        # Head 3: anisotropy index A_F ∈ [0,1]
        # With sector-sum A_F values in [0.35, 0.75], Sigmoid is appropriate:
        # it maps the full real line to (0,1) with good gradients in this range.
        self.head_AF = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor,
                descriptor: torch.Tensor):
        """
        Parameters
        ----------
        image      : (B, 1, H, W)  normalised degraded STM image
        descriptor : (B, 2K+1)     Fourier boundary descriptor

        Returns
        -------
        cos_pred : (B, M)
        sin_pred : (B, M)
        AF_pred  : (B, 1)
        """
        # Stream 1
        img_feat = self.img_stream(image)
        img_feat = self.img_fc(img_feat)       # (B, 256)

        # Stream 2
        pot_feat = self.pot_mlp(descriptor)    # (B, 128)

        # Fusion
        z = self.fusion(img_feat, pot_feat)    # (B, 128)
        z = self.latent(z)                     # (B, 128)

        # Heads
        cos_pred = self.head_cos(z)            # (B, M)
        sin_pred = self.head_sin(z)            # (B, M)
        AF_pred  = self.head_AF(z)             # (B, 1)

        return cos_pred, sin_pred, AF_pred

    def predict_with_uncertainty(
        self,
        image: torch.Tensor,
        descriptor: torch.Tensor,
        n_passes: int = 50,
    ):
        """
        Monte-Carlo dropout uncertainty estimation.
        Keeps dropout active during inference.

        Returns
        -------
        means  : dict {'cos', 'sin', 'AF'}  — mean predictions
        stds   : dict {'cos', 'sin', 'AF'}  — predictive std (uncertainty)
        """
        self.train()   # activate dropout
        cos_samples, sin_samples, AF_samples = [], [], []

        with torch.no_grad():
            for _ in range(n_passes):
                c, s, a = self.forward(image, descriptor)
                cos_samples.append(c)
                sin_samples.append(s)
                AF_samples.append(a)

        self.eval()

        cos_stack = torch.stack(cos_samples)  # (n_passes, B, M)
        sin_stack = torch.stack(sin_samples)
        AF_stack  = torch.stack(AF_samples)   # (n_passes, B, 1)

        means = {
            'cos': cos_stack.mean(0),
            'sin': sin_stack.mean(0),
            'AF' : AF_stack.mean(0),
        }
        stds = {
            'cos': cos_stack.std(0),
            'sin': sin_stack.std(0),
            'AF' : AF_stack.std(0),
        }
        return means, stds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = FisherCNN(grid_size=64, descriptor_dim=7, M=6)
    print(f"Total parameters: {model.count_parameters():,}")

    # Forward pass smoke test
    B = 4
    img  = torch.randn(B, 1, 64, 64)
    desc = torch.randn(B, 7)
    cos_, sin_, AF_ = model(img, desc)
    print(f"cos_pred shape: {cos_.shape}")   # (4, 6)
    print(f"sin_pred shape: {sin_.shape}")   # (4, 6)
    print(f"AF_pred  shape: {AF_.shape}")    # (4, 1)
    print("Model smoke test passed.")
