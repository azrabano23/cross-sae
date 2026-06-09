"""
EXPERIMENT — is the SAE a faithful unit on HIGH-SNR brain data? (fMRI analog of §7)

FINDINGS §7 showed, on scalp EEG, that an SAE preserves cross-domain structure
once it has enough capacity (sparsity is not the culprit) — but EEG's ceiling is
only 0.214, so the test was ceiling-limited. THINGS-fMRI visual cortex has a 0.46
ceiling (§10), giving a much cleaner test of the same question, now WITHIN the
brain: does a sparse autoencoder on fMRI visual-cortex betas preserve the brain's
own representational geometry (its RDM) as well as dense PCA, up to the ceiling?

Method (cross-validated, unbiased):
  - split each image's 12 reps into half A / half B.
  - ceiling = RSA( raw-RDM(A), raw-RDM(B) )  [the reliability bound].
  - SAE  : train on within-session-z single trials; RSA( SAE-RDM(A), raw-RDM(B) ).
  - PCA  : fit on per-image(A); RSA( PCA-RDM(A), raw-RDM(B) ).
  - sweep capacity; SAE reported as mean +/- std over seeds.

Evaluating the SAE/PCA RDM against the HELD-OUT raw RDM means a faithful basis
approaches the ceiling and cannot exceed it by overfitting. Real data only
(cached THINGS-fMRI visual-ROI betas), reproducible, no images, no gated data.

Run:  python experiments/fmri_sae_capacity.py
Out:  results/fmri_sae_capacity.png / .csv
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
VIS_CACHE = os.path.join(ROOT, "data", "things_fmri", "visual_betas_cache.npz")
CAPS = [8, 16, 32, 64]
SEEDS = [0, 1]


def rdm(F):
    Fz = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Fz @ Fz.T) / F.shape[1]


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    d = np.load(VIS_CACHE)
    betas, img, sess = d["betas"].astype(np.float64), d["img"], d["sess"]
    n_img = int(img.max()) + 1
    # within-session z-score
    for s in np.unique(sess):
        mm = sess == s
        betas[mm] = (betas[mm] - betas[mm].mean(0)) / (betas[mm].std(0) + 1e-8)
    print(f"fMRI visual betas: {betas.shape}, {n_img} images")

    reps = [np.where(img == i)[0] for i in range(n_img)]
    rng = np.random.default_rng(0)
    # fixed A/B split of reps per image (same split for all methods)
    A_idx, B_idx = [], []
    for idx in reps:
        p = rng.permutation(idx); h = len(p) // 2
        A_idx.append(p[:h]); B_idx.append(p[h:])
    rawA = np.stack([betas[a].mean(0) for a in A_idx])
    rawB = np.stack([betas[b].mean(0) for b in B_idx])
    trA = betas[np.concatenate(A_idx)]                # single-trial betas in half A (SAE training)
    trA_img = np.concatenate([[i] * len(a) for i, a in enumerate(A_idx)])

    R_rawB = rdm(rawB)
    ceiling = spearmanr(upper(rdm(rawA)), upper(R_rawB)).correlation
    print(f"raw split-half ceiling = {ceiling:.3f}")

    pca_rsa, sae_mean, sae_std = [], [], []
    for c in CAPS:
        Pm = PCA(n_components=c, random_state=0).fit(rawA)
        pc = spearmanr(upper(rdm(Pm.transform(rawA))), upper(R_rawB)).correlation
        pca_rsa.append(pc)
        rs = []
        for s in SEEDS:
            sae = train_sae(torch.from_numpy(trA).float(), 8 * c, c, seed=s, epochs=200)
            Z = encode_dataset(sae, torch.from_numpy(trA).float()).numpy()
            saeA = np.stack([Z[trA_img == i].mean(0) for i in range(n_img)])
            rs.append(spearmanr(upper(rdm(saeA)), upper(R_rawB)).correlation)
        sae_mean.append(float(np.mean(rs))); sae_std.append(float(np.std(rs)))
        print(f"  c={c:3d}: PCA={pc:+.3f} | SAE={np.mean(rs):+.3f} +/- {np.std(rs):.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(ceiling, color="#2e7d32", ls="--", lw=1.8, label=f"raw split-half ceiling ({ceiling:.2f})")
    ax.plot(CAPS, pca_rsa, "s-", color="#1565c0", lw=2, ms=7, label="PCA (dense)")
    ax.errorbar(CAPS, sae_mean, yerr=sae_std, fmt="o-", color="#c62828", lw=2, ms=7,
                capsize=4, label="SAE (sparse, ±std)")
    ax.set_xscale("log", base=2); ax.set_xticks(CAPS); ax.set_xticklabels(CAPS)
    ax.set_xlabel("active dimensions (PCA comps / SAE top-k)")
    ax.set_ylabel("RSA to held-out raw fMRI RDM")
    ax.set_title("Is the SAE a faithful unit on high-SNR fMRI visual cortex?\n"
                 "(THINGS-fMRI sub-01, cross-validated; approaches the reliability ceiling)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fmri_sae_capacity.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")
    with open(os.path.join(RESULTS, "fmri_sae_capacity.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["capacity", "pca_rsa", "sae_rsa_mean", "sae_rsa_std", "ceiling"])
        for i, c in enumerate(CAPS):
            w.writerow([c, f"{pca_rsa[i]:.4f}", f"{sae_mean[i]:.4f}", f"{sae_std[i]:.4f}", f"{ceiling:.4f}"])

    best_sae, best_pca = max(sae_mean), max(pca_rsa)
    print("\n=== VERDICT ===")
    print(f"ceiling={ceiling:.2f} | best SAE={best_sae:.2f} | best PCA={best_pca:.2f}")
    if best_sae >= ceiling - 0.05:
        print("SAE recovers ~all reliable fMRI visual structure -> faithful unit on high-SNR brain data.")
    elif best_sae >= best_pca - 0.03:
        print("SAE matches dense PCA at capacity -> sparsity does not cost fMRI representational structure.")
    else:
        print("SAE underperforms PCA on fMRI -> the sparse basis loses brain structure here.")


if __name__ == "__main__":
    main()
