"""
EXPERIMENT — Is the cross-domain attenuation caused by SPARSITY, or just by
dimensionality reduction? And does more capacity recover the shared structure?

Background: raw ViT<->EEG representations share significant structure
(RSA rho=0.155, p=0.0005), but matching SAE features misses it. This experiment
asks WHY, with two controls a reviewer would require:

  1. EEG NOISE CEILING (split-half reliability of the EEG RDM): the maximum RSA
     any model can achieve given EEG measurement noise. Without it, "rho=0.15" is
     uninterpretable.
  2. DENSE PCA BASELINE at matched component counts: PCA reduces dimensionality
     WITHOUT sparsity. If SAE-RSA << PCA-RSA at matched capacity, the cost is
     sparsity specifically; if SAE ~= PCA, it is just dimensionality.

We sweep capacity for both bases and compare their cross-domain RSA to the raw
ceiling and the noise ceiling. SAEs are trained at multiple seeds (they are
non-deterministic) and reported as mean +/- std.

Real data only: THINGS-EEG2 sub-01 (all sessions) + pretrained ViT over the same
200 images. Reproducible: fixed seeds, permutation tests, no synthetic data.

Run:  python experiments/sae_vs_pca_rsa.py
Out:  results/sae_vs_pca_rsa.png / .csv
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
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from experiments.headline_model_brain import load_epochs, vit_features

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CAPACITIES = [8, 16, 32, 64]          # active dims (PCA n_comp; SAE k, with d_hidden=8*k)
SEEDS = [0, 1, 2]
N_PERM = 2000


def rdm(X):
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Xz @ Xz.T) / X.shape[1]


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def rsa(Ra, Rb):
    return spearmanr(upper(Ra), upper(Rb)).correlation


def rsa_p(Ra, Rb, n_perm=N_PERM, seed=0):
    obs = rsa(Ra, Rb); n = Ra.shape[0]; rng = np.random.default_rng(seed)
    a = upper(Ra)
    null = np.array([spearmanr(a, upper(Rb[np.ix_(p, p)])).correlation
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    return obs, (1 + np.sum(null >= obs)) / (1 + n_perm)


def eeg_noise_ceiling(Xtr, trial_img, n_splits=10, seed=0):
    """Split-half reliability of the EEG RDM (Spearman-Brown corrected) = ceiling."""
    rng = np.random.default_rng(seed)
    rels = []
    for _ in range(n_splits):
        A = np.zeros((200, Xtr.shape[1])); B = np.zeros((200, Xtr.shape[1]))
        for i in range(200):
            idx = np.where(trial_img == i)[0]; rng.shuffle(idx)
            h = len(idx) // 2
            A[i] = Xtr[idx[:h]].mean(0); B[i] = Xtr[idx[h:2*h]].mean(0)
        r = rsa(rdm(A), rdm(B))
        rels.append(2 * r / (1 + r))                # Spearman-Brown to full reliability
    return float(np.mean(rels)), float(np.std(rels))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Loading real EEG (all sessions) + ViT over 200 shared images...")
    Xtr, trial_img, Xb_img, img_paths = load_epochs()
    Vimg = vit_features(img_paths)
    print(f"  trials {Xtr.shape}, per-image EEG {Xb_img.shape}, ViT {Vimg.shape}")

    R_eeg, R_vit = rdm(Xb_img), rdm(Vimg)

    nc_mean, nc_std = eeg_noise_ceiling(Xtr, trial_img)
    raw_rho, raw_p = rsa_p(R_vit, R_eeg)
    print(f"  EEG noise ceiling (SB) = {nc_mean:.3f} +/- {nc_std:.3f}")
    print(f"  RAW ViT<->EEG RSA = {raw_rho:.3f} (p={raw_p:.4f})")

    # PCA baseline (dense reduction)
    print("PCA baseline across capacities...")
    pca_rho = []
    for c in CAPACITIES:
        Pm = PCA(n_components=c, random_state=0).fit_transform(Vimg)
        Pe = PCA(n_components=c, random_state=0).fit_transform(Xb_img)
        r = rsa(rdm(Pm), rdm(Pe)); pca_rho.append(r)
        print(f"  PCA c={c:3d}: RSA={r:+.3f}")

    # SAE sweep (sparse reduction), multiple seeds
    print("SAE sweep across capacities (multi-seed)...")
    Xtr_t = torch.from_numpy(Xtr); Vimg_t = torch.from_numpy(Vimg)
    sae_mean, sae_std = [], []
    for c in CAPACITIES:
        d = 8 * c
        rs = []
        for s in SEEDS:
            bsae = train_sae(Xtr_t, d, c, seed=s, epochs=120)
            Ztr = encode_dataset(bsae, Xtr_t).numpy()
            Zb = np.stack([Ztr[trial_img == i].mean(0) for i in range(200)])
            msae = train_sae(Vimg_t, d, c, seed=s, epochs=300)
            Zm = encode_dataset(msae, Vimg_t).numpy()
            rs.append(rsa(rdm(Zm), rdm(Zb)))
        sae_mean.append(float(np.mean(rs))); sae_std.append(float(np.std(rs)))
        print(f"  SAE k={c:3d} (d={d}): RSA={np.mean(rs):+.3f} +/- {np.std(rs):.3f}")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhspan(nc_mean - nc_std, nc_mean + nc_std, color="#c8e6c9", alpha=0.6,
               label=f"EEG noise ceiling ({nc_mean:.2f})")
    ax.axhline(raw_rho, color="#37474f", ls=":", lw=1.8, label=f"raw ViT<->EEG ({raw_rho:.2f}, p={raw_p:.3f})")
    ax.plot(CAPACITIES, pca_rho, "s-", color="#1565c0", lw=2, ms=7, label="PCA (dense)")
    ax.errorbar(CAPACITIES, sae_mean, yerr=sae_std, fmt="o-", color="#c62828",
                lw=2, ms=7, capsize=4, label="SAE (sparse, ±std over seeds)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(CAPACITIES); ax.set_xticklabels(CAPACITIES)
    ax.set_xlabel("active dimensions per sample (PCA components / SAE top-k)")
    ax.set_ylabel("cross-domain RSA  (model RDM vs brain RDM)")
    ax.set_title("Does the cross-domain structure survive a sparse vs dense basis?\n"
                 "real ViT-SAE <-> human-EEG (THINGS-EEG2, 200 shared images)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(RESULTS, "sae_vs_pca_rsa.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "sae_vs_pca_rsa.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capacity", "pca_rsa", "sae_rsa_mean", "sae_rsa_std"])
        for i, c in enumerate(CAPACITIES):
            w.writerow([c, f"{pca_rho[i]:.4f}", f"{sae_mean[i]:.4f}", f"{sae_std[i]:.4f}"])
        w.writerow(["raw_rho", f"{raw_rho:.4f}", "noise_ceiling", f"{nc_mean:.4f}"])

    # verdict
    best_sae = max(sae_mean); best_pca = max(pca_rho)
    print("\n=== VERDICT ===")
    print(f"raw={raw_rho:.3f} | best PCA={best_pca:.3f} | best SAE={best_sae:.3f} | ceiling={nc_mean:.3f}")
    if best_pca - best_sae > 0.03:
        print("Sparsity-specific cost: PCA preserves more cross-domain structure than SAE at matched capacity.")
    elif best_sae >= raw_rho - 0.02:
        print("Capacity recovers it: large SAEs approach the raw cross-domain RSA.")
    else:
        print("Both bases lose structure vs raw -> reduction itself (not only sparsity) attenuates cross-domain alignment.")


if __name__ == "__main__":
    main()
