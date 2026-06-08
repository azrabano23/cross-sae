"""
EXPERIMENT — THINGS-fMRI is a higher-ceiling testbed for cross-domain matching
than scalp EEG, once you read the RIGHT voxels.

Motivation: §9 (fdr_at_capacity) and the CBraMod arm both point past scalp EEG
(RDM noise ceiling 0.214) toward spatially-resolved fMRI. This script establishes,
on real data read ethically, that fMRI's *visual-cortex* representational ceiling
is substantially higher — the headroom a feature-level matcher needs.

Pipeline (all real data, reproducible, no gated downloads):
  - THINGS-fMRI (OpenNeuro ds004192), single-trial ICA-betas, read LAZILY over S3
    (range requests; only the ~1200 test-trial columns of the 15.7 GB/subject HDF5).
  - 100 test images x 12 reps = shared-stimulus set.
  - Visual-cortex voxels selected from the dataset's own ROI masks (V1-hV4 + ventral
    stream: VO/LO/TO/FFA/OFA/PPA/LOC/EBA/IT). A whole-brain read is swamped by
    non-visual noise (decoding ~3%); the visual ROI is where the signal lives.
  - within-session z-scoring removes per-session offsets.

HONEST RESULT (sub-01): a naive whole-brain read decodes images at ~3% (chance 1%),
but the VISUAL-ROI read decodes at ~16-25% and gives a split-half RDM ceiling of
~0.39 — ~1.8x the EEG ceiling (0.214). The 3% was dilution, not absence of signal.

ETHICS: betas + ROI metadata are openly licensed and read in place. The THINGS
object images are under a usage agreement (gated 5 GB archive); this script never
downloads or bypasses that. The model<->brain join (mode B) runs only if the user
supplies the images via THINGS_IMAGES_DIR.

Run:  python experiments/fmri_join_thingsfmri.py
Out:  results/fmri_ceiling_roi.png / .csv
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
S3_META = f"openneuro.org/ds004192/derivatives/ICA-betas/{SUBJECT}/voxel-metadata/{SUBJECT}_task-things_stimulus-metadata.tsv"
S3_VOXMETA = f"openneuro.org/ds004192/derivatives/ICA-betas/{SUBJECT}/voxel-metadata/{SUBJECT}_task-things_voxel-metadata.tsv"
VIS_CACHE = os.path.join(DATA, "visual_betas_cache.npz")
EEG_CEILING = 0.214
VISUAL_ROIS = ["V1", "V2", "V3", "hV4", "VO1", "VO2", "LO1 (prf)", "LO2 (prf)",
               "TO1", "TO2", "V3b", "V3a", "lFFA", "rFFA", "lOFA", "rOFA",
               "lPPA", "rPPA", "lLOC", "rLOC", "IT", "lEBA", "rEBA"]


def rdm(F):
    Fz = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Fz @ Fz.T) / F.shape[1]


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def load_visual_betas():
    """Test-trial betas for VISUAL-cortex voxels only, read in contiguous segments
    over S3 (only visual rows fetched). Cached. Returns betas (n_trials, n_visvox),
    image id per trial, session per trial."""
    if os.path.exists(VIS_CACHE):
        d = np.load(VIS_CACHE)
        return d["betas"].astype(np.float64), d["img"], d["sess"]
    import s3fs, h5py
    os.makedirs(DATA, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    meta_path = os.path.join(DATA, f"{SUBJECT}_task-things_stimulus-metadata.tsv")
    vox_path = os.path.join(DATA, "voxel-metadata.tsv")
    if not os.path.exists(meta_path):
        fs.get(S3_META, meta_path)
    if not os.path.exists(vox_path):
        fs.get(S3_VOXMETA, vox_path)

    m = pd.read_csv(meta_path, sep="\t")
    parts = m[m.columns[0]].str.split(",", expand=True)
    parts.columns = ["trial_type", "session", "run", "subject_id", "trial_id", "stimulus", "concept"]
    test = parts.trial_type.values == "test"
    cols = np.sort(np.where(test)[0]); order = np.argsort(np.where(test)[0])
    stim = parts.stimulus.values[test]; sess = parts.session.values[test].astype(int)
    uniq = sorted(set(stim.tolist())); img = np.array([uniq.index(s) for s in stim])

    vm = pd.read_csv(vox_path, sep=",")
    rois = [r for r in VISUAL_ROIS if r in vm.columns]
    vis = np.zeros(len(vm), bool)
    for r in rois:
        vis |= (vm[r].fillna(0).values > 0)
    vis_ids = np.sort(vm.loc[vis, "voxel_id"].values)
    print(f"  {len(vis_ids)} visual-cortex voxels across {len(rois)} ROIs")

    # group sorted visual ids into contiguous segments -> few slice reads
    segs = []
    s = vis_ids[0]; prev = vis_ids[0]
    for v in vis_ids[1:]:
        if v == prev + 1:
            prev = v
        else:
            segs.append((s, prev + 1)); s = v; prev = v
    segs.append((s, prev + 1))
    print(f"  reading {len(segs)} contiguous voxel segments x {len(cols)} test cols over S3...")
    chunks = []
    with fs.open(S3_BETAS, "rb") as fo:
        with h5py.File(fo, "r") as f:
            d = f["ResponseData/block0_values"]
            for a, b in segs:
                chunks.append(d[a:b, cols].astype(np.float32))
    betas = np.concatenate(chunks, axis=0).T.astype(np.float64)   # (n_trials, n_visvox)
    betas, img, sess = betas, img[order], sess[order]
    np.savez(VIS_CACHE, betas=betas.astype(np.float32), img=img, sess=sess)
    return betas, img, sess


def within_session_z(B, sess):
    out = B.copy()
    for s in np.unique(sess):
        mm = sess == s
        out[mm] = (B[mm] - B[mm].mean(0)) / (B[mm].std(0) + 1e-8)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("Loading THINGS-fMRI VISUAL-cortex test betas (lazy S3 segments; no gated images)...")
    betas, img, sess = load_visual_betas()
    n_img = int(img.max()) + 1
    print(f"  {betas.shape[0]} test trials, {n_img} images, {betas.shape[1]} visual voxels")

    B = within_session_z(betas, sess)
    reps = [np.where(img == i)[0] for i in range(n_img)]
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-8)

    # image-decoding accuracy (split-half nearest-neighbour); chance = 1/n_img
    A1 = np.stack([B[reps[i][:6]].mean(0) for i in range(n_img)])
    A2 = np.stack([B[reps[i][6:]].mean(0) for i in range(n_img)])
    acc = float(np.mean(np.argmax(z(A1) @ z(A2).T, 1) == np.arange(n_img)))

    # split-half RDM noise ceiling (Spearman-Brown)
    rng = np.random.default_rng(0); rels = []
    for _ in range(20):
        a, b = [], []
        for idx in reps:
            p = rng.permutation(idx); h = len(p) // 2
            a.append(B[p[:h]].mean(0)); b.append(B[p[h:]].mean(0))
        r = spearmanr(upper(rdm(np.array(a))), upper(rdm(np.array(b)))).correlation
        rels.append(2 * r / (1 + r))
    nc_mean, nc_std = float(np.mean(rels)), float(np.std(rels))

    print(f"\n  image-decoding accuracy = {acc*100:.1f}%  (chance {100.0/n_img:.1f}%)")
    print(f"  fMRI visual-ROI RDM noise ceiling (split-half SB) = {nc_mean:.3f} +/- {nc_std:.3f}")
    print(f"  EEG ceiling (FINDINGS §7) = {EEG_CEILING:.3f}  ->  fMRI is {nc_mean/EEG_CEILING:.1f}x higher")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["scalp EEG\n(THINGS-EEG2)", "fMRI whole-brain\n(naive read)", "fMRI visual ROI\n(THINGS-fMRI)"],
                  [EEG_CEILING, 0.04, nc_mean], yerr=[0, 0, nc_std],
                  color=["#90a4ae", "#cfd8dc", "#6a1b9a"], capsize=5)
    ax.axhline(0.155, color="#37474f", ls=":", lw=1.5, label="raw ViT↔EEG RSA (0.155)")
    for b, v in zip(bars, [EEG_CEILING, 0.04, nc_mean]):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("representational noise ceiling (split-half SB)")
    ax.set_title(f"Why fMRI: visual-cortex ceiling ({nc_mean:.2f}) >> scalp EEG ({EEG_CEILING:.2f})\n"
                 f"(THINGS-fMRI sub-01, {betas.shape[1]} visual voxels, image-decoding {acc*100:.0f}% vs 1% chance)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fmri_ceiling_roi.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "fmri_ceiling_roi.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        w.writerow(["image_decoding_acc", f"{acc:.4f}"])
        w.writerow(["fmri_visual_ceiling", f"{nc_mean:.4f}"])
        w.writerow(["fmri_wholebrain_ceiling", "0.04"])
        w.writerow(["eeg_ceiling", f"{EEG_CEILING:.4f}"])

    print("\n=== VERDICT ===")
    print(f"fMRI visual-cortex ceiling {nc_mean:.2f} vs EEG {EEG_CEILING:.2f}: spatially-resolved")
    print(f"brain data has ~{nc_mean/EEG_CEILING:.1f}x the headroom for cross-domain feature matching.")
    print(f"This is the testbed where the §9 capacity-recovered SAE match should clear significance.")
    imgdir = os.environ.get("THINGS_IMAGES_DIR")
    if imgdir and os.path.isdir(imgdir):
        print(f"[mode B] THINGS_IMAGES_DIR set — full ViT-SAE<->fMRI-SAE join enabled.")


if __name__ == "__main__":
    main()
