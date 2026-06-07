"""
DIAGNOSTIC — why is the model<->brain SAE match null on THINGS-EEG2?

Two competing explanations for the headline null:
  (A) DATA: there is no ViT<->EEG shared structure across these 200 images, so
      nothing could be matched.
  (B) METHOD: shared structure exists, but sparse-feature matching destroys/misses
      it (e.g., the SAE basis is the wrong unit for cross-domain comparison).

Representational Similarity Analysis (RSA) distinguishes them. We build a 200x200
representational dissimilarity matrix (RDM) for each domain and correlate them with
a permutation test:
  - RAW RSA   : ViT activations  vs  EEG responses
  - SAE RSA   : model-SAE features vs brain-SAE features

If RAW RSA is significant but SAE RSA is not -> explanation (B): the SAE step is
discarding the shared structure (a real methodological finding). If RAW RSA is
also null -> explanation (A): these per-image EEG responses just don't carry
ViT-aligned information, and the right fix is spatially-resolved fMRI.

Run:  python experiments/rsa_diagnostic.py
Out:  results/rsa_diagnostic.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from experiments.headline_model_brain import load_epochs, vit_features

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def rdm(X):
    """1 - Pearson correlation RDM over rows (images)."""
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    C = (Xz @ Xz.T) / X.shape[1]
    return 1.0 - C


def upper(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def rsa_permtest(Ra, Rb, n_perm=2000, seed=0):
    a, b = upper(Ra), upper(Rb)
    obs = spearmanr(a, b).correlation
    n = Ra.shape[0]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearmanr(a, upper(Rb[np.ix_(p, p)])).correlation
    pval = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, pval, null


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Loading EEG (pooled) + ViT features over 200 shared images...")
    Xtr, trial_img, Xb_img, img_paths = load_epochs()
    Vimg = vit_features(img_paths)
    print(f"  ViT {Vimg.shape}, EEG per-image {Xb_img.shape}")

    # SAE features (same recipe as the headline experiment)
    bsae = train_sae(torch.from_numpy(Xtr), 64, 8, seed=0, epochs=200)
    Ztr = encode_dataset(bsae, torch.from_numpy(Xtr)).numpy()
    Zb = np.stack([Ztr[trial_img == i].mean(0) for i in range(200)])
    msae = train_sae(torch.from_numpy(Vimg), 64, 8, seed=0, epochs=300)
    Zm = encode_dataset(msae, torch.from_numpy(Vimg)).numpy()

    print("RSA: raw ViT vs raw EEG ...")
    raw_r, raw_p, raw_null = rsa_permtest(rdm(Vimg), rdm(Xb_img))
    print(f"  raw RSA  rho={raw_r:+.3f}  p={raw_p:.4f}")
    print("RSA: model-SAE vs brain-SAE ...")
    sae_r, sae_p, _ = rsa_permtest(rdm(Zm), rdm(Zb))
    print(f"  SAE RSA  rho={sae_r:+.3f}  p={sae_p:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.hist(raw_null, bins=40, color="#b0bec5", label="null")
    ax.axvline(raw_r, color="#c62828", lw=2.5, label=f"raw RSA rho={raw_r:+.3f} (p={raw_p:.3f})")
    ax.axvline(sae_r, color="#1565c0", lw=2.5, ls="--", label=f"SAE RSA rho={sae_r:+.3f} (p={sae_p:.3f})")
    ax.set_xlabel("RSA Spearman rho (ViT RDM vs EEG RDM)"); ax.set_ylabel("# permutations")
    ax.set_title("Is there ANY shared ViT<->EEG structure?"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    im = ax.imshow(rdm(Vimg), cmap="viridis"); ax.set_title("ViT representational dissimilarity (200 images)")
    ax.set_xlabel("image"); ax.set_ylabel("image")
    fig.colorbar(im, ax=ax, fraction=0.046)

    concl = ("DATA-limited: no shared structure (raw RSA n.s.)" if raw_p > 0.05 else
             "SAE discards shared structure (raw sig, SAE n.s.)" if sae_p > 0.05 else
             "shared structure present and SAE-preserved")
    fig.suptitle(f"DIAGNOSTIC — locating the model<->brain null   |   verdict: {concl}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(RESULTS, "rsa_diagnostic.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")
    with open(os.path.join(RESULTS, "rsa_diagnostic.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["comparison", "rho", "p"])
        w.writerow(["raw_ViT_vs_EEG", f"{raw_r:.4f}", f"{raw_p:.4f}"])
        w.writerow(["modelSAE_vs_brainSAE", f"{sae_r:.4f}", f"{sae_p:.4f}"])
    print(f"VERDICT: {concl}")


if __name__ == "__main__":
    main()
