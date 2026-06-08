"""
HEADLINE JOIN (fMRI) — real ViT-SAE <-> human-fMRI-SAE FDR matching on the
high-ceiling THINGS-fMRI visual cortex (ceiling 0.46, vs EEG 0.214).

This is the experiment the whole project was building toward, on the brain
substrate where it has the best chance: spatially-resolved visual-cortex fMRI.

  THINGS images --ViT--> activations --SAE--> model features  (100 x p_model)
  fMRI visual-ROI betas        --SAE--> brain features  (100 x p_brain)
  cross_domain_match(model, brain, FDR q) + permutation null + RSA-vs-ceiling

Brain side is fully prepared (cached visual-ROI betas from
experiments/fmri_join_thingsfmri.py). The MODEL side needs the 100 THINGS test
images, which are under THINGS's research-use agreement and are NOT bundled here.

  >>> Provide them yourself and run:
        THINGS_IMAGES_DIR=/path/to/THINGS/object_images \
            python experiments/fmri_model_brain_join.py
  The script finds the 100 test images by filename (recursively), reports coverage,
  and runs the join on whatever it finds. Images are never copied into the repo.

Caveat (honest): n=100 images is small for knockoffs, so candidate features are
capped well below n on each side; this is a first-subject pilot, not a population
result. The 0.46 ceiling is what gives it a chance the EEG join (0.214) did not.

Out:  results/fmri_model_brain_join.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from crosssae.matching import cross_domain_match
from experiments.headline_model_brain import vit_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "things_fmri")
VIS_CACHE = os.path.join(DATA, "visual_betas_cache.npz")
META = os.path.join(DATA, "sub-01_task-things_stimulus-metadata.tsv")

FDR_Q = 0.2
N_PERM = 20
MAX_FEATS = 25          # candidates must be << n=100 for a well-posed knockoff filter
D_HID, K = 256, 16


def rdm(F):
    Fz = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Fz @ Fz.T) / F.shape[1]


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def test_stimulus_names():
    m = pd.read_csv(META, sep="\t")
    parts = m[m.columns[0]].str.split(",", expand=True)
    parts.columns = ["trial_type", "session", "run", "subject_id", "trial_id", "stimulus", "concept"]
    test = parts.trial_type.values == "test"
    return sorted(set(parts.stimulus.values[test].tolist()))    # == the `img` index order


def live(Z):
    frac = (Z > 0).mean(0)
    idx = np.where(frac >= 0.05)[0]
    if len(idx) > MAX_FEATS:
        idx = idx[np.argsort(-frac[idx])[:MAX_FEATS]]
    return idx


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    if not os.path.exists(VIS_CACHE):
        print("Brain-side cache missing. Run experiments/fmri_join_thingsfmri.py first.")
        return
    d = np.load(VIS_CACHE)
    betas, img, sess = d["betas"].astype(np.float64), d["img"], d["sess"]
    uniq = test_stimulus_names()
    n_img = len(uniq)
    print(f"Brain side: {betas.shape[0]} trials, {n_img} images, {betas.shape[1]} visual voxels")

    imgdir = os.environ.get("THINGS_IMAGES_DIR")
    if not imgdir or not os.path.isdir(imgdir):
        print("\nTHINGS_IMAGES_DIR not set. The brain side is ready; provide the 100 THINGS")
        print("test images (research-use agreement) and re-run:")
        print("  THINGS_IMAGES_DIR=/path/to/THINGS python experiments/fmri_model_brain_join.py")
        print("\nExpected test image filenames (first 5):", uniq[:5])
        return

    # locate the 100 test images by filename
    index = {os.path.basename(p): p for p in glob.glob(os.path.join(imgdir, "**", "*.jpg"), recursive=True)}
    paths = [index.get(name) for name in uniq]
    found = [i for i, p in enumerate(paths) if p]
    print(f"Found {len(found)}/{n_img} THINGS test images in {imgdir}")
    if len(found) < 30:
        print("Too few images found — check THINGS_IMAGES_DIR points at the object-image folders.")
        return

    keep_img = np.array(found)
    Vimg = vit_features([paths[i] for i in keep_img])           # (n_found, 384)

    # brain SAE on single trials of kept images, encode per-image averages
    keep_mask = np.isin(img, keep_img)
    remap = {old: new for new, old in enumerate(keep_img)}
    Xtr = betas[keep_mask]
    tr_img = np.array([remap[i] for i in img[keep_mask]])
    tr_sess = sess[keep_mask]
    # within-session z
    for s in np.unique(tr_sess):
        mm = tr_sess == s
        Xtr[mm] = (Xtr[mm] - Xtr[mm].mean(0)) / (Xtr[mm].std(0) + 1e-8)

    print(f"Training brain-side SAE on {Xtr.shape[0]} fMRI trials...")
    bsae = train_sae(torch.from_numpy(Xtr).float(), D_HID, K, seed=0, epochs=200)
    Ztr = encode_dataset(bsae, torch.from_numpy(Xtr).float()).numpy()
    Zb = np.stack([Ztr[tr_img == i].mean(0) for i in range(len(keep_img))])
    print(f"Training model-side SAE on {len(keep_img)} ViT activations...")
    msae = train_sae(torch.from_numpy(Vimg).float(), D_HID, K, seed=0, epochs=300)
    Zm = encode_dataset(msae, torch.from_numpy(Vimg).float()).numpy()

    Zm, Zb = Zm[:, live(Zm)], Zb[:, live(Zb)]
    n = Zm.shape[0]
    print(f"  live features: model {Zm.shape[1]}, brain {Zb.shape[1]}; n={n} images")

    # raw RSA for context: ViT RDM vs fMRI RDM, both in keep_img order
    brain_per_img = np.stack([betas[img == i].mean(0) for i in keep_img])
    raw_rsa = spearmanr(upper(rdm(Vimg)), upper(rdm(brain_per_img))).correlation

    # FDR cross-domain match + permutation null
    print(f"FDR cross-domain match (q={FDR_Q})...")
    M, stats = cross_domain_match(Zm, Zb, q=FDR_Q, seed=0)
    real = stats["total_matches"]
    null = []
    for pi in range(N_PERM):
        perm = np.random.default_rng(900 + pi).permutation(n)
        _, sp = cross_domain_match(Zm, Zb[perm], q=FDR_Q, seed=0)
        null.append(sp["total_matches"])
    null = np.array(null)
    p = (1 + np.sum(null >= real)) / (1 + N_PERM)
    print(f"  raw ViT<->fMRI RSA = {raw_rsa:+.3f}")
    print(f"  FDR matches: real {real} vs null {null.mean():.1f} (max {null.max()}) -> p={p:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(null, bins=range(0, max(int(null.max()), real) + 2), color="#b0bec5", label="permutation null")
    ax.axvline(real, color="#6a1b9a", lw=2.5, label=f"real = {real} (p={p:.3f})")
    ax.set_xlabel("# FDR-significant ViT-SAE <-> fMRI-SAE matches")
    ax.set_ylabel("# permutations")
    ax.set_title(f"Model<->brain join on fMRI visual cortex (ceiling 0.46)\n"
                 f"THINGS-fMRI sub-01, n={n} images, raw RSA={raw_rsa:.2f}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fmri_model_brain_join.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")
    with open(os.path.join(RESULTS, "fmri_model_brain_join.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n_images", "raw_rsa", "real_matches", "null_mean", "p"])
        w.writerow([n, f"{raw_rsa:.4f}", real, f"{null.mean():.2f}", f"{p:.3f}"])
    verdict = "ABOVE CHANCE" if p < 0.05 else "not above chance"
    print(f"\nVERDICT: ViT-SAE<->fMRI-SAE match {real} vs null {null.mean():.1f}, p={p:.3f} -> {verdict}.")


if __name__ == "__main__":
    main()
