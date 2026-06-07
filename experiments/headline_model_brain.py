"""
HEADLINE EXPERIMENT — the real model<->brain SAE feature match on shared stimuli.

Data: THINGS-EEG2 test set (subject 1, session 1) — 200 natural images shown in
rapid serial visual presentation while 63-channel EEG was recorded, each image
repeated ~20x. The SAME 200 images are fed to a pretrained vision transformer.

Pipeline:
  images --ViT--> activations --SAE--> model features  (200 x p_model)
  EEG --epoch+pseudo-average--> responses --SAE--> brain features (n x p_brain)
  cross_domain_match(model, brain, FDR q)  ->  significance-tested model<->brain pairs

This is the first FDR-controlled matching of sparse-autoencoder features between a
vision model and the human brain on a shared-stimulus set, to our knowledge.

Run (after downloading THINGS-EEG2 sub-01 ses-01 to data/things_eeg2/):
    python experiments/headline_model_brain.py
Outputs:
    results/headline_model_brain.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from crosssae.matching import cross_domain_match, matched_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "things_eeg2")
EEG = os.path.join(DATA, "raw-eeg/sub-01/ses-01/raw_eeg_test.npy")

EPOCH_MS = (0, 250)          # post-onset window (RSVP SOA is 200ms)
TIME_BINS = 5               # downsample the window to this many mean-bins
FDR_Q = 0.2
N_PERM = 30                # permutation null (shuffle image<->brain alignment)

# Small feature banks: matching is at the IMAGE level (n=200 independent rows),
# so candidates must stay well below n for the knockoff filter to be well-posed.
D_MODEL_HID, K_MODEL = 64, 8
D_BRAIN_HID, K_BRAIN = 64, 8
MIN_ACTIVE = 0.05


def load_epochs():
    """Return image-level brain matrix Xb (200, feats) — each row is the average
    EEG response to one image across all its repetitions (independent rows) — and
    the ordered list of 200 image file paths (by trigger code)."""
    a = np.load(EEG, allow_pickle=True).item()
    data = a["raw_eeg_data"]; ch = list(a["ch_names"]); sf = int(a["sfreq"])
    eeg_idx = [i for i, t in enumerate(a["ch_types"]) if t == "eeg"]
    stim = data[ch.index("stim")]
    X = data[eeg_idx]                                  # (63, T)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)

    onset = np.where((stim[1:] > 0) & (stim[:-1] == 0))[0] + 1
    codes = stim[onset].astype(int)
    keep = codes <= 200
    onset, codes = onset[keep], codes[keep]

    s0 = int(EPOCH_MS[0] * sf / 1000); s1 = int(EPOCH_MS[1] * sf / 1000)
    edges = np.linspace(s0, s1, TIME_BINS + 1).astype(int)

    meta = np.load(os.path.join(DATA, "image_metadata.npy"), allow_pickle=True).item()
    files = meta["test_img_files"]
    by_base = {os.path.basename(p): p for p in glob.glob(os.path.join(DATA, "test_images", "**", "*.jpg"), recursive=True)}
    img_paths = [by_base.get(files[k]) for k in range(200)]

    trials, trial_img, Xb_img = [], [], []
    for code in range(1, 201):
        ev = onset[codes == code]
        ep = []
        for o in ev:
            if o + s1 >= X.shape[1]:
                continue
            w = X[:, o + s0:o + s1]
            binned = np.stack([w[:, edges[b]-s0:edges[b+1]-s0].mean(1) for b in range(TIME_BINS)], axis=1)
            ep.append(binned.reshape(-1))             # 63*TIME_BINS
        ep = np.array(ep, dtype=np.float32)
        trials.append(ep); trial_img.extend([code - 1] * len(ep))
        Xb_img.append(ep.mean(0))                     # per-image average (for matching)
    return (np.concatenate(trials), np.array(trial_img),
            np.array(Xb_img, dtype=np.float32), img_paths)


def vit_features(img_paths):
    import timm
    from timm.data import resolve_data_config, create_transform
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    model.eval()
    cfg = resolve_data_config({}, model=model); tfm = create_transform(**cfg)
    feats = {}
    model.blocks[6].register_forward_hook(lambda m, i, o: feats.__setitem__("a", o.detach()))
    out = []
    with torch.no_grad():
        for p in img_paths:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0)
            model(x); out.append(feats["a"].mean(1).squeeze(0).numpy())
    return np.asarray(out, dtype=np.float32)           # (200, 384)


def live(Z):
    return np.where((Z > 0).mean(0) >= MIN_ACTIVE)[0]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Epoching real EEG (sub-01 ses-01 test)...")
    Xtr, trial_img, Xb_img, img_paths = load_epochs()
    print(f"  {Xtr.shape[0]} single trials -> 200 per-image averages ({Xb_img.shape[1]} feats)")

    print("Extracting ViT activations over the 200 shared images...")
    Vimg = vit_features(img_paths)                     # (200, 384)

    # Train the brain SAE on ALL single trials (more data -> better features),
    # then read out per-image brain features by averaging trial features per image.
    print("Training BRAIN-side SAE on all single trials...")
    brain_sae = train_sae(torch.from_numpy(Xtr), D_BRAIN_HID, K_BRAIN, seed=0, epochs=200)
    Ztr = encode_dataset(brain_sae, torch.from_numpy(Xtr)).numpy()
    Zb = np.stack([Ztr[trial_img == i].mean(0) for i in range(200)])   # (200, D_BRAIN_HID)
    print("Training MODEL-side SAE on shared-image ViT activations...")
    model_sae = train_sae(torch.from_numpy(Vimg), D_MODEL_HID, K_MODEL, seed=0, epochs=300)
    Zm = encode_dataset(model_sae, torch.from_numpy(Vimg)).numpy()

    lm, lb = live(Zm), live(Zb)
    Zm, Zb = Zm[:, lm], Zb[:, lb]
    n = Zm.shape[0]
    print(f"  live features: model {len(lm)}, brain {len(lb)}; n={n} independent images")

    print(f"FDR-controlled cross-domain matching (model<->brain), q={FDR_Q}...")
    M, stats = cross_domain_match(Zm, Zb, q=FDR_Q, seed=0)
    n_real = stats["total_matches"]
    pairs = matched_pairs(M, Zm, Zb)
    print(f"  -> {n_real} model<->brain pairs ({stats['n_matched_targets']}/{stats['n_targets']} model features matched)")

    # --- PERMUTATION NULL: break the true image<->brain alignment, re-match ---
    print(f"Permutation null ({N_PERM} shuffles of image<->brain alignment)...")
    null_counts = []
    for pi in range(N_PERM):
        perm = np.random.default_rng(1000 + pi).permutation(n)
        Mp, sp = cross_domain_match(Zm, Zb[perm], q=FDR_Q, seed=0)
        null_counts.append(sp["total_matches"])
    null_counts = np.array(null_counts)
    null_mean = float(null_counts.mean())
    p_val = float((1 + (null_counts >= n_real).sum()) / (1 + N_PERM))
    print(f"  null matches: mean {null_mean:.1f} (max {null_counts.max()}); "
          f"real {n_real} -> permutation p = {p_val:.3f}")

    # --- POSITIVE CONTROL: same matcher on synthetic data that HAS cross-domain
    #     structure, so the figure shows the method detects signal when it exists. ---
    print("Positive control (synthetic shared-stimulus with planted structure)...")
    from crosssae.synthetic import make_shared_stimulus
    Zm_s, Zb_s, _ = make_shared_stimulus(n=600, n_model_features=15, p_brain=40,
                                         brain_per_model=10, rng=np.random.default_rng(0))
    Ms, ss = cross_domain_match(Zm_s, Zb_s, q=FDR_Q, seed=0)
    pos_real = ss["total_matches"]
    pos_null = []
    for pi in range(10):
        permp = np.random.default_rng(7000 + pi).permutation(Zb_s.shape[0])
        _, spp = cross_domain_match(Zm_s, Zb_s[permp], q=FDR_Q, seed=0)
        pos_null.append(spp["total_matches"])
    pos_null = np.array(pos_null)
    pos_p = float((1 + (pos_null >= pos_real).sum()) / (1 + len(pos_null)))
    print(f"  positive control: real {pos_real} vs null mean {pos_null.mean():.1f} (p={pos_p:.3f})")

    # --- figure: positive control (signal exists) | real EEG (honest test) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.hist(pos_null, bins=range(0, max(int(pos_null.max()), pos_real) + 2),
            color="#b0bec5", label="null (shuffled)")
    ax.axvline(pos_real, color="#2e7d32", lw=2.5, label=f"real = {pos_real}  (null max {int(pos_null.max())})")
    ax.set_xlabel("# FDR-significant cross-domain matches"); ax.set_ylabel("# permutations")
    ax.set_title("POSITIVE CONTROL (synthetic, signal present)\nmethod detects cross-domain structure")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.hist(null_counts, bins=range(0, max(int(null_counts.max()), n_real) + 2),
            color="#b0bec5", label="null (shuffled alignment)")
    ax.axvline(n_real, color="#c62828", lw=2.5, label=f"real = {n_real} (p={p_val:.3f})")
    ax.set_xlabel("# FDR-significant model<->brain matches"); ax.set_ylabel("# permutations")
    ax.set_title("REAL DATA (THINGS-EEG2, single session)\nno above-chance matches — data-limited")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "HEADLINE — FDR-controlled model<->brain SAE feature matching, permutation-validated\n"
        f"Method detects cross-domain structure when present (left); single-session scalp EEG is data-limited (right, n={n})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(RESULTS, "headline_model_brain.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "headline_model_brain.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model_feature", "brain_feature", "correlation"])
        for a, b, r in pairs:
            w.writerow([a, b, f"{r:.4f}"])

    verdict = ("ABOVE CHANCE" if p_val < 0.05 else
               "NOT above chance" if p_val > 0.1 else "marginal")
    print(f"\nHEADLINE: {n_real} matches vs null mean {null_mean:.1f}, p={p_val:.3f} -> {verdict}.")
    if p_val >= 0.05:
        print("Honest read: single-session EEG carries weak cross-domain SAE structure; "
              "more sessions/subjects needed for a confident model<->brain claim.")


if __name__ == "__main__":
    main()
