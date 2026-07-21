"""
EXPERIMENT — do unsupervised fMRI-SAE features recover known visual areas?

If a sparse autoencoder trained on fMRI visual-cortex betas is finding real units
of brain representation, its features should be ANATOMICALLY coherent: a feature's
spatial pattern (decoder weights over voxels) should concentrate in specific visual
ROIs (V1, hV4, FFA, PPA, ...) rather than smearing across all of cortex. This is
the anatomical version of "are SAE features real?" — and needs no images.

Method:
  - train an fMRI-SAE on within-session-z visual-cortex betas.
  - each SAE feature has a decoder column = its weight over the visual voxels.
  - assign each voxel its visual ROI (dataset's own masks); for each feature compute
    ROI concentration = fraction of its (absolute) decoder mass in its top ROI.
  - compare to a null: shuffle the voxel<->ROI assignment and recompute.
If real features concentrate in single ROIs far above the shuffled null, the SAE
has recovered the visual area structure without supervision.

Real data only (cached THINGS-fMRI visual betas + dataset ROI masks). No images.

Run:  python experiments/fmri_sae_roi.py
Out:  results/fmri_sae_roi.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "things_fmri")
VIS_CACHE = os.path.join(DATA, "visual_betas_cache.npz")
VOXMETA = os.path.join(DATA, "voxel-metadata.tsv")
META = os.path.join(DATA, "sub-01_task-things_stimulus-metadata.tsv")
# coarse visual-area groups (each maps several fine masks to one label)
ROI_GROUPS = {
    "V1": ["V1"], "V2": ["V2"], "V3": ["V3", "V3a", "V3b"], "hV4": ["hV4"],
    "ventral (VO)": ["VO1", "VO2"], "lateral (LO/TO)": ["LO1 (prf)", "LO2 (prf)", "TO1", "TO2", "lLOC", "rLOC"],
    "FFA": ["lFFA", "rFFA"], "OFA": ["lOFA", "rOFA"], "PPA": ["lPPA", "rPPA"],
    "EBA": ["lEBA", "rEBA"], "IT": ["IT"],
}
D_HID, K = 256, 16


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    d = np.load(VIS_CACHE)
    betas, sess = d["betas"].astype(np.float64), d["sess"]
    for s in np.unique(sess):
        mm = sess == s
        betas[mm] = (betas[mm] - betas[mm].mean(0)) / (betas[mm].std(0) + 1e-8)

    # recover the visual voxel order used to build the cache (sorted visual voxel_ids)
    vm = pd.read_csv(VOXMETA, sep=",")
    all_rois = [r for g in ROI_GROUPS.values() for r in g]
    vis_mask = np.zeros(len(vm), bool)
    for r in all_rois:
        if r in vm.columns:
            vis_mask |= (vm[r].fillna(0).values > 0)
    vis_rows = np.where(vis_mask)[0]
    vis_ids = vm.loc[vis_mask, "voxel_id"].values
    sort = np.argsort(vis_ids)
    vis_rows = vis_rows[sort]                          # matches cache voxel order
    assert len(vis_rows) == betas.shape[1], (len(vis_rows), betas.shape[1])

    # assign each visual voxel a coarse ROI group (first group it belongs to)
    group_names = list(ROI_GROUPS.keys())
    vox_group = np.full(betas.shape[1], -1)
    for gi, (g, masks) in enumerate(ROI_GROUPS.items()):
        m = np.zeros(len(vis_rows), bool)
        for r in masks:
            if r in vm.columns:
                m |= (vm.iloc[vis_rows][r].fillna(0).values > 0)
        vox_group[(vox_group == -1) & m] = gi
    valid = vox_group >= 0
    print(f"{betas.shape[1]} visual voxels; {valid.sum()} assigned to {len(group_names)} ROI groups")

    print("Training fMRI-SAE on visual-cortex betas...")
    sae = train_sae(torch.from_numpy(betas).float(), D_HID, K, seed=0, epochs=200)
    W = sae.decoder.weight.detach().numpy()           # (n_voxels, d_hidden)
    W = np.abs(W)

    # per-feature ROI concentration: fraction of decoder mass in its top ROI group
    def concentration(weights, groups):
        conc, top = [], []
        for f in range(weights.shape[1]):
            w = weights[:, f]
            mass = np.array([w[groups == gi].sum() for gi in range(len(group_names))])
            tot = mass.sum()
            if tot <= 0:
                continue
            conc.append(mass.max() / tot); top.append(int(mass.argmax()))
        return np.array(conc), np.array(top)

    Wv = W[valid]; gv = vox_group[valid]
    real_conc, real_top = concentration(Wv, gv)
    rng = np.random.default_rng(0)
    null_conc = np.concatenate([concentration(Wv, rng.permutation(gv))[0] for _ in range(5)])
    print(f"feature ROI concentration: real mean={real_conc.mean():.3f} | shuffled null={null_conc.mean():.3f}")

    # which ROI groups are most represented among selective features
    selective = real_conc > np.quantile(null_conc, 0.95)
    top_counts = np.bincount(real_top[selective], minlength=len(group_names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.hist(null_conc, bins=25, color="#b0bec5", density=True, label="shuffled voxel→ROI null")
    ax.hist(real_conc, bins=25, color="#6a1b9a", density=True, alpha=0.7, label="real SAE features")
    ax.axvline(real_conc.mean(), color="#6a1b9a", lw=2)
    ax.set_xlabel("feature ROI concentration (mass in top visual area)")
    ax.set_ylabel("density"); ax.set_title("Unsupervised fMRI-SAE features are ROI-concentrated")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    order = np.argsort(-top_counts)
    ax.barh([group_names[i] for i in order][::-1], top_counts[order][::-1], color="#1565c0")
    ax.set_xlabel(f"# ROI-selective SAE features (conc > 95th-pct null)")
    ax.set_title("Which visual areas the selective features map to")

    fig.suptitle("Do unsupervised fMRI-SAE features recover known visual areas? (THINGS-fMRI sub-01)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULTS, "fmri_sae_roi.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")
    with open(os.path.join(RESULTS, "fmri_sae_roi.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        w.writerow(["real_mean_concentration", f"{real_conc.mean():.4f}"])
        w.writerow(["null_mean_concentration", f"{null_conc.mean():.4f}"])
        w.writerow(["n_selective_features", int(selective.sum())])
        for i in order:
            if top_counts[i] > 0:
                w.writerow([f"selective_in_{group_names[i]}", int(top_counts[i])])

    print("\n=== VERDICT ===")
    lift = real_conc.mean() / null_conc.mean()
    print(f"Real features are {lift:.2f}x more ROI-concentrated than the shuffled null "
          f"({int(selective.sum())} strongly-selective features). Unsupervised SAE features "
          f"{'recover' if lift > 1.2 else 'do NOT clearly recover'} the visual-area structure.")


if __name__ == "__main__":
    main()
