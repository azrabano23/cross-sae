"""
Top-k sparse autoencoder (Makhzani & Frey 2013; Gao et al. 2024 "Top-k SAEs").

Used identically on BOTH domains:
  - model side: cached ViT/CLIP activations over a stimulus set
  - brain side: per-stimulus EEG/fMRI response vectors (the *brain-side SAE* that
    SAE-BrainMap (arXiv:2506.11123) does not train — one of our differentiators).

Top-k is chosen over L1 SAEs because the sparsity level is exact and directly
comparable across the two domains, which matters when we later match features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        # Tie decoder init to encoder transpose, unit-norm dictionary columns.
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self._normalize_decoder()

    def _normalize_decoder(self):
        w = self.decoder.weight
        self.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x - self.b_pre)
        # Top-k activation: keep the k largest positive pre-activations per row.
        topv, topi = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topi, F.relu(topv))
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z) + self.b_pre
        return x_hat, z

    def loss(self, x: torch.Tensor):
        x_hat, z = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        return recon, z


def train_sae(acts: torch.Tensor, d_hidden: int, k: int, seed: int = 0,
              epochs: int = 200, lr: float = 1e-3, batch: int = 512,
              device: str = "cpu") -> TopKSAE:
    """Train a Top-k SAE on a (n_samples, d_in) activation matrix.

    `seed` is exposed deliberately: re-training with different seeds is how the
    stability layer (crosssae.stability) quantifies SAE non-determinism, which
    the 2025 literature shows is severe (~30% feature overlap across seeds).
    """
    torch.manual_seed(seed)
    acts = acts.to(device).float()
    n, d_in = acts.shape
    sae = TopKSAE(d_in, d_hidden, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            recon, _ = sae.loss(acts[idx])
            opt.zero_grad()
            recon.backward()
            opt.step()
            with torch.no_grad():
                sae._normalize_decoder()
    return sae


@torch.no_grad()
def encode_dataset(sae: TopKSAE, acts: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Return the (n_samples, d_hidden) feature-activation matrix."""
    return sae.encode(acts.to(device).float()).cpu()
