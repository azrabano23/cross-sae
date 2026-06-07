"""
MVP — REAL human-brain-side pipeline (the brain half of the cross-domain method).

Mirrors experiments/phase1_vit_sae.py exactly, but the activations come from a
human brain instead of a vision transformer:

    real fMRI (Haxby 2001, ventral-temporal cortex) -> brain-side Top-k SAE ->
    FDR-controlled matching of brain SAE features to the viewed object category.

Together with Phase 1 this gives BOTH domains on real data through the identical
knockoff engine: model-SAE<->concept and brain-SAE<->concept. The only remaining
step for the headline result is to join them on a shared-stimulus set (THINGS),
where the right-hand matrix becomes the *other domain's* SAE features.

Why Haxby: it is the canonical, clean, openly-fetchable shared-stimulus fMRI
dataset (8 object categories viewed in-scanner), one nilearn call, ~300MB.
THINGS-EEG2 is the scale-up target (its HuggingFace loader currently ships a
broken ClassLabel schema; tracked as a known issue).

Run:
    pip install nilearn
    python experiments/mvp_brain_sae.py
Outputs:
    results/mvp_brain_sae.png
    results/mvp_brain_concept_features.csv
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from crosssae.knockoffs import select_matches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

D_HIDDEN = 512
K = 32
SAE_EPOCHS = 400
FDR_Q = 0.2
MIN_ACTIVE_FRAC = 0.005


def load_haxby_brain():
    import pandas as pd
    from nilearn import datasets
    from nilearn.maskers import NiftiMasker

    h = datasets.fetch_haxby(subjects=[1])
    labels = pd.read_csv(h.session_target[0], sep=" ")
    y = labels["labels"].values

    # Mask to ventral-temporal cortex; standardize per-voxel.
    masker = NiftiMasker(mask_img=h.mask_vt[0], standardize="zscore_sample",
                         detrend=True, smoothing_fwhm=4)
    X = masker.fit_transform(h.func[0])          # (n_timepoints, n_voxels)

    # Drop the 'rest' condition; keep the 8 object categories.
    keep = y != "rest"
    return X[keep].astype(np.float32), y[keep]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Loading real Haxby fMRI (ventral-temporal cortex)...")
    X, y = load_haxby_brain()
    cats = sorted(set(y.tolist()))
    n, d_in = X.shape
    print(f"Brain responses: {X.shape}  |  {len(cats)} categories: {cats}")

    acts = torch.from_numpy(X)
    print(f"Training BRAIN-side Top-k SAE (d_hidden={D_HIDDEN}, k={K}) on real fMRI...")
    sae = train_sae(acts, d_hidden=D_HIDDEN, k=K, seed=0, epochs=SAE_EPOCHS)
    Z = encode_dataset(sae, acts).numpy()

    with torch.no_grad():
        x_hat, _ = sae(acts)
        r2 = 1.0 - ((acts - x_hat) ** 2).sum().item() / ((acts - acts.mean(0)) ** 2).sum().item()

    active_frac = (Z > 0).mean(0)
    keep = np.where(active_frac >= MIN_ACTIVE_FRAC)[0]
    Zk = Z[:, keep]
    print(f"Brain SAE: R^2={r2:.3f}, {len(keep)}/{D_HIDDEN} features active >= {MIN_ACTIVE_FRAC:.0%}")

    print(f"FDR-controlled matching of BRAIN features <-> viewed category (q={FDR_Q})...")
    rows, sig_counts = [], np.zeros(len(cats), dtype=int)
    concept_feat = np.zeros((len(cats), len(keep)))
    for c, cname in enumerate(cats):
        yc = (y == cname).astype(float)
        sel, W = select_matches(Zk, yc, q=FDR_Q, rng=np.random.default_rng(200 + c))
        sig_counts[c] = len(sel)
        concept_feat[c, sel] = 1
        best = int(keep[sel[np.argmax(W[sel])]]) if len(sel) else -1
        rows.append((cname, len(sel), best, float(W[sel].max()) if len(sel) else 0.0))
        print(f"  {cname:12s}: {len(sel):2d} significant brain-SAE features")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    order = np.argsort(-sig_counts)
    ax.barh([cats[i] for i in order][::-1], sig_counts[order][::-1], color="#7b1fa2")
    ax.set_xlabel("# FDR-significant brain-SAE features (q=%.2f)" % FDR_Q)
    ax.set_title("Brain-SAE features matched to viewed category\n(real human fMRI, knockoff-controlled)")

    ax = axes[1]
    used = np.where(concept_feat.sum(0) > 0)[0]
    ax.imshow(concept_feat[:, used], aspect="auto", cmap="Purples", interpolation="nearest")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_xlabel("brain-SAE feature (significant for >=1 category)")
    ax.set_title("Category <-> brain-SAE-feature matches\n(purple = significant at FDR %.2f)" % FDR_Q)

    fig.suptitle(
        "MVP — real human-brain SAE feature<->category matching with FDR control "
        f"(Haxby fMRI, SAE R^2={r2:.2f}, {len(keep)} live features, n={n})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(RESULTS, "mvp_brain_sae.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved figure -> {out_png}")

    out_csv = os.path.join(RESULTS, "mvp_brain_concept_features.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n_significant_features", "top_feature_id", "top_W"])
        for r in rows:
            w.writerow(r)
    print(f"Saved table  -> {out_csv}")
    print(f"\nMVP complete: real human fMRI -> brain SAE (R^2={r2:.2f}) -> "
          f"{int(sig_counts.sum())} FDR-controlled feature<->category matches across {len(cats)} categories.")


if __name__ == "__main__":
    main()
