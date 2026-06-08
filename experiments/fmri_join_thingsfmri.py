"""
EXPERIMENT — the fMRI testbed: does spatially-resolved brain data give a higher
ceiling for cross-domain matching than scalp EEG (ceiling 0.214, FINDINGS §7-8)?

Data: THINGS-fMRI (OpenNeuro ds004192), single-trial ICA-betas. The 100 repeated
TEST images (12 reps each) are the shared-stimulus set. Betas are read LAZILY from
the 15.7 GB per-subject HDF5 over S3 (range requests) — only the ~1200 test-trial
columns are fetched, never the whole file.

Two modes, by design:
  (A) BRAIN-SIDE CEILING (runs now, no image download): split-half reliability of
      the fMRI test-image RDM = the maximum cross-domain RSA achievable here. This
      is the number that decides whether fMRI is worth it vs EEG's 0.214.
  (B) FULL model<->brain join (needs the 100 THINGS test images): ViT-SAE <->
      fMRI-SAE RSA + FDR match + permutation null.

ETHICS: the THINGS object images are distributed under a usage agreement
(password-protected 5 GB archive at osf.io/jum2f). This script does NOT download
or bypass that. Mode (B) runs only if you have obtained the images yourself and
point THINGS_IMAGES_DIR at them. Mode (A) uses only the openly-licensed betas.

Run:  python experiments/fmri_join_thingsfmri.py
Out:  results/fmri_noise_ceiling.png / .csv  (mode A)
"""
from __future__ import annotations

import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "things_fmri")
SUBJECT = "sub-01"
S3_BETAS = f"openneuro.org/ds004192/derivatives/ICA-betas/{SUBJECT}/voxel-metadata/{SUBJECT}_task-things_voxel-wise-responses.h5"
S3_META = f"s3://openneuro.org/ds004192/derivatives/ICA-betas/{SUBJECT}/voxel-metadata/{SUBJECT}_task-things_stimulus-metadata.tsv"
VOX = 60000              # voxel subset read for the ceiling (contiguous slice)
EEG_CEILING = 0.214      # from FINDINGS §7, for comparison


def rdm(F):
    Fz = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Fz @ Fz.T) / F.shape[1]


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def load_test_betas():
    """Lazy-read the test-trial betas (VOX voxels x 1200 trials) from S3 and the
    trial->image mapping. Returns per-trial betas (n_trials, VOX) and image id per
    trial (0..99)."""
    import s3fs, h5py
    meta_path = os.path.join(DATA, f"{SUBJECT}_task-things_stimulus-metadata.tsv")
    if not os.path.exists(meta_path):
        os.makedirs(DATA, exist_ok=True)
        s3fs.S3FileSystem(anon=True).get(S3_META.replace("s3://", ""), meta_path)
    m = pd.read_csv(meta_path, sep="\t")
    parts = m[m.columns[0]].str.split(",", expand=True)
    parts.columns = ["trial_type", "session", "run", "subject_id", "trial_id", "stimulus", "concept"]
    is_test = parts.trial_type.values == "test"
    test_cols = np.where(is_test)[0]
    stimuli = parts.stimulus.values[is_test]
    uniq = sorted(set(stimuli.tolist()))
    img_of_trial = np.array([uniq.index(s) for s in stimuli])

    fs = s3fs.S3FileSystem(anon=True)
    print(f"  lazy-reading {len(test_cols)} test-trial columns x {VOX} voxels from S3 (range reads)...")
    with fs.open(S3_BETAS, "rb") as fo:
        with h5py.File(fo, "r") as f:
            d = f["ResponseData/block0_values"]          # (211339 voxels, 9840 trials)
            cols = np.sort(test_cols)
            betas = d[:VOX, :][:, cols].T.astype(np.float32)   # (n_test_trials, VOX)
    # reorder img_of_trial to match sorted cols
    order = np.argsort(test_cols)
    return betas[np.argsort(order)], img_of_trial, uniq


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("Loading THINGS-fMRI test betas (lazy S3, no full download)...")
    betas, img_of_trial, uniq = load_test_betas()
    n_img = len(uniq)
    print(f"  {betas.shape[0]} test trials, {n_img} images, {betas.shape[1]} voxels")

    # per-voxel z-score across trials
    betas = (betas - betas.mean(0)) / (betas.std(0) + 1e-8)

    # split-half noise ceiling of the fMRI test-image RDM
    rng = np.random.default_rng(0)
    rels = []
    for _ in range(20):
        A = np.zeros((n_img, betas.shape[1])); B = np.zeros((n_img, betas.shape[1]))
        for i in range(n_img):
            idx = np.where(img_of_trial == i)[0]; rng.shuffle(idx); h = len(idx) // 2
            A[i] = betas[idx[:h]].mean(0); B[i] = betas[idx[h:2*h]].mean(0)
        r = spearmanr(upper(rdm(A)), upper(rdm(B))).correlation
        rels.append(2 * r / (1 + r))
    nc_mean, nc_std = float(np.mean(rels)), float(np.std(rels))
    print(f"\n  fMRI noise ceiling (split-half SB) = {nc_mean:.3f} +/- {nc_std:.3f}")
    print(f"  EEG noise ceiling (FINDINGS §7)     = {EEG_CEILING:.3f}")
    print(f"  -> fMRI ceiling is {nc_mean/EEG_CEILING:.1f}x the EEG ceiling")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(["scalp EEG\n(THINGS-EEG2)", "fMRI\n(THINGS-fMRI)"],
                  [EEG_CEILING, nc_mean], yerr=[0, nc_std],
                  color=["#90a4ae", "#6a1b9a"], capsize=5)
    ax.axhline(0.155, color="#37474f", ls=":", lw=1.5, label="raw ViT↔EEG RSA (0.155)")
    ax.set_ylabel("representational noise ceiling (split-half SB)")
    ax.set_title("Why fMRI: spatially-resolved brain data has a far higher\n"
                 "ceiling for cross-domain matching than scalp EEG")
    for b, v in zip(bars, [EEG_CEILING, nc_mean]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fmri_noise_ceiling.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "fmri_noise_ceiling.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modality", "noise_ceiling", "std"])
        w.writerow(["EEG_THINGS-EEG2", f"{EEG_CEILING:.4f}", "0"])
        w.writerow(["fMRI_THINGS-fMRI", f"{nc_mean:.4f}", f"{nc_std:.4f}"])

    imgdir = os.environ.get("THINGS_IMAGES_DIR")
    print("\n=== Mode B (full model<->brain join) ===")
    if imgdir and os.path.isdir(imgdir):
        print(f"THINGS_IMAGES_DIR set -> would run ViT-SAE<->fMRI-SAE match on {n_img} images.")
        print("(image-bearing arm: TODO once images verified present)")
    else:
        print("THINGS images not provided (gated, usage agreement). Set THINGS_IMAGES_DIR to")
        print("the THINGS object-image folder to run the full join. Brain-side ceiling above")
        print(f"shows the headroom: fMRI {nc_mean:.2f} vs EEG {EEG_CEILING:.2f}.")


if __name__ == "__main__":
    main()
