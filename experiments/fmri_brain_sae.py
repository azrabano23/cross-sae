"""
EXPERIMENT (§11) — does the SAE basis PRESERVE the rich fMRI visual structure?
The prerequisite for moving the cross-domain join to the high-ceiling fMRI testbed.

On scalp EEG the brain-side SAE was capacity-limited and only marginally preserved the
ViT-aligned structure (§6/§7, ceiling 0.214). The whole reason to move to THINGS-fMRI
is its ~2x higher visual-cortex ceiling (~0.46, §10/fmri_join_thingsfmri.py). But a
higher ceiling is only useful if the SAE — Tarjuman's brain-side dictionary — actually
RETAINS the image-discriminative geometry on this substrate. This script tests exactly
that, using only the openly-licensed, already-cached visual-cortex betas (NO gated
THINGS images needed — this is a brain-side-only analysis).

For SAE (sparse) and PCA (dense) at matched capacity k in {8,16,32,64}, on the 100
test images (12 reps each), we measure two preservation metrics against the raw
visual-voxel betas:
  1. image-decoding accuracy (split-half nearest-neighbour; chance = 1%)
  2. RDM preservation: Spearman(SAE/PCA RDM, raw-betas RDM) — does the low-dim basis
     keep the representational geometry?
plus the SAE reconstruction R². Reported vs the raw decoding and the split-half ceiling.

If SAE decoding/RDM-preservation tracks PCA and stays high on fMRI (unlike the marginal
EEG case), the brain-side dictionary is viable on the high-ceiling substrate and the
full ViT-SAE<->fMRI-SAE join is green-lit (pending user-supplied THINGS_IMAGES_DIR).

Run:  python experiments/fmri_brain_sae.py
Out:  results/fmri_brain_sae.png / .csv
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
from crosssae.sae import TopKSAE, train_sae, encode_dataset
from experiments.fmri_join_thingsfmri import load_visual_betas, within_session_z, rdm, upper

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
EEG_CEILING = 0.214
CAPACITIES = [8, 16, 32, 64]
SEEDS = [0, 1]


def z(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def decode_acc(per_img_a, per_img_b):
    """split-half nearest-neighbour image decoding; chance = 1/n_img."""
    n = per_img_a.shape[0]
    sim = z(per_img_a) @ z(per_img_b).T
    return float(np.mean(np.argmax(sim, 1) == np.arange(n)))


def halves_per_image(F_trials, reps, rng):
    """Return two (n_img, d) matrices: independent split-half per-image means."""
    a, b = [], []
    for idx in reps:
        p = rng.permutation(idx); h = len(p) // 2
        a.append(F_trials[p[:h]].mean(0)); b.append(F_trials[p[h:2 * h]].mean(0))
    return np.array(a), np.array(b)


def noise_ceiling(B, reps, n_rep=20, seed=0):
    rng = np.random.default_rng(seed); rels = []
    for _ in range(n_rep):
        a, b = halves_per_image(B, reps, rng)
        r = spearmanr(upper(rdm(a)), upper(rdm(b))).correlation
        rels.append(2 * r / (1 + r))
    return float(np.mean(rels)), float(np.std(rels))


def sae_r2(sae, X):
    with torch.no_grad():
        xh, _ = sae.forward(torch.from_numpy(X.astype(np.float32)))
        xh = xh.numpy()
    ss_res = ((X - xh) ** 2).sum(); ss_tot = ((X - X.mean(0)) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Loading THINGS-fMRI visual-cortex betas (cached; no gated images)...")
    betas, img, sess = load_visual_betas()
    n_img = int(img.max()) + 1
    B = within_session_z(betas, sess)
    reps = [np.where(img == i)[0] for i in range(n_img)]
    print(f"  {B.shape[0]} trials, {n_img} images, {B.shape[1]} visual voxels")

    rng0 = np.random.default_rng(0)
    # raw references
    A1raw, A2raw = halves_per_image(B, reps, np.random.default_rng(1))
    raw_dec = decode_acc(A1raw, A2raw)
    nc_mean, nc_std = noise_ceiling(B, reps)
    raw_img = np.stack([B[reps[i]].mean(0) for i in range(n_img)])
    R_raw = rdm(raw_img)
    print(f"  RAW: decoding {raw_dec*100:.1f}% (chance {100/n_img:.0f}%), "
          f"RDM ceiling {nc_mean:.3f} (vs EEG {EEG_CEILING:.3f}, {nc_mean/EEG_CEILING:.1f}x)")

    Bt = torch.from_numpy(B.astype(np.float32))
    rows = []
    for c in CAPACITIES:
        d = 8 * c
        sae_dec, sae_rdm, sae_r2s = [], [], []
        for s in SEEDS:
            sae = train_sae(Bt, d, c, seed=s, epochs=120)
            Ztr = encode_dataset(sae, Bt).numpy()
            a, b = halves_per_image(Ztr, reps, np.random.default_rng(1))
            sae_dec.append(decode_acc(a, b))
            Zimg = np.stack([Ztr[reps[i]].mean(0) for i in range(n_img)])
            sae_rdm.append(spearmanr(upper(rdm(Zimg)), upper(R_raw)).correlation)
            sae_r2s.append(sae_r2(sae, B))
        # PCA control at matched capacity
        P = PCA(n_components=c, random_state=0).fit_transform(B)
        Pa = np.stack([P[reps[i][:6]].mean(0) for i in range(n_img)])
        Pb = np.stack([P[reps[i][6:]].mean(0) for i in range(n_img)])
        pca_dec = decode_acc(Pa, Pb)
        Pimg = np.stack([P[reps[i]].mean(0) for i in range(n_img)])
        pca_rdm = spearmanr(upper(rdm(Pimg)), upper(R_raw)).correlation
        row = dict(k=c, sae_dec=float(np.mean(sae_dec)), sae_dec_s=float(np.std(sae_dec)),
                   sae_rdm=float(np.mean(sae_rdm)), sae_r2=float(np.mean(sae_r2s)),
                   pca_dec=pca_dec, pca_rdm=float(pca_rdm))
        rows.append(row)
        print(f"  k={c:3d}: SAE dec {row['sae_dec']*100:4.1f}% rdm {row['sae_rdm']:.3f} "
              f"R2 {row['sae_r2']:.2f} | PCA dec {pca_dec*100:4.1f}% rdm {pca_rdm:.3f}")

    # --- figure: decoding + RDM preservation vs capacity ----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ks = [r["k"] for r in rows]
    ax1.axhline(raw_dec, color="#37474f", ls=":", lw=1.6, label=f"raw voxels ({raw_dec*100:.0f}%)")
    ax1.axhline(100 / n_img / 100, color="gray", lw=0.8)
    ax1.errorbar(ks, [r["sae_dec"] for r in rows], yerr=[r["sae_dec_s"] for r in rows],
                 fmt="o-", color="#6a1b9a", lw=2, ms=7, capsize=4, label="SAE (sparse)")
    ax1.plot(ks, [r["pca_dec"] for r in rows], "s--", color="#b39ddb", lw=1.6, ms=6, label="PCA (dense)")
    ax1.set_xscale("log", base=2); ax1.set_xticks(ks); ax1.set_xticklabels(ks)
    ax1.set_xlabel("capacity (SAE top-k / PCA comps)"); ax1.set_ylabel("image-decoding accuracy")
    ax1.set_title("Does the basis keep images DECODABLE?"); ax1.legend(frameon=False, fontsize=8)

    ax2.axhspan(nc_mean - nc_std, nc_mean + nc_std, color="#c8e6c9", alpha=0.6,
                label=f"raw RDM ceiling ({nc_mean:.2f})")
    ax2.plot(ks, [r["sae_rdm"] for r in rows], "o-", color="#6a1b9a", lw=2, ms=7, label="SAE RDM vs raw RDM")
    ax2.plot(ks, [r["pca_rdm"] for r in rows], "s--", color="#b39ddb", lw=1.6, ms=6, label="PCA RDM vs raw RDM")
    ax2.set_xscale("log", base=2); ax2.set_xticks(ks); ax2.set_xticklabels(ks)
    ax2.set_xlabel("capacity (SAE top-k / PCA comps)"); ax2.set_ylabel("Spearman(basis RDM, raw RDM)")
    ax2.set_title("Does the basis keep the GEOMETRY?"); ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("§11 — fMRI brain-side SAE preservation (THINGS-fMRI sub-01, 100 images, "
                 f"{B.shape[1]} visual voxels)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS, "fmri_brain_sae.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "fmri_brain_sae.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "sae_dec", "sae_dec_std", "sae_rdm_pres", "sae_r2", "pca_dec", "pca_rdm_pres"])
        for r in rows:
            w.writerow([r["k"], f"{r['sae_dec']:.4f}", f"{r['sae_dec_s']:.4f}",
                        f"{r['sae_rdm']:.4f}", f"{r['sae_r2']:.3f}",
                        f"{r['pca_dec']:.4f}", f"{r['pca_rdm']:.4f}"])
        w.writerow(["raw_dec", f"{raw_dec:.4f}", "ceiling", f"{nc_mean:.4f}", "", "", ""])

    best = max(rows, key=lambda r: r["sae_dec"])
    print("\n=== VERDICT ===")
    print(f"raw voxel decoding {raw_dec*100:.0f}% | best SAE decoding {best['sae_dec']*100:.0f}% "
          f"(k={best['k']}), RDM-preservation {best['sae_rdm']:.2f}")
    if best["sae_dec"] >= 0.5 * raw_dec and best["sae_rdm"] >= 0.5:
        print("SAE PRESERVES fMRI visual structure -> brain-side dictionary is viable on the")
        print("high-ceiling substrate; the ViT-SAE<->fMRI-SAE join is green-lit (needs images).")
    else:
        print("SAE loses substantial structure even on fMRI -> report honestly; the dictionary")
        print("basis, not the substrate, is then the bottleneck (escalate SAE width/training).")


if __name__ == "__main__":
    main()
