"""
PHASE 1 — REAL model-side pipeline, end to end, on a real pretrained ViT.

This is the model-side analog of the eventual brain experiment. Instead of
matching ViT-SAE features to *brain* features, we match them to *semantic
concepts* (real class labels) under the same FDR-controlled procedure. It proves
the full pipeline works on real activations:

    real images -> pretrained ViT activations -> Top-k SAE -> FDR-controlled
    matching of SAE features to concepts (knockoffs) -> named, significance-tested
    feature<->concept correspondences.

Swapping the concept indicators for brain-side SAE features is the only change
needed to run the headline cross-domain (model<->brain) experiment.

Run:
    pip install timm torchvision
    python experiments/phase1_vit_sae.py
Outputs:
    results/phase1_vit_sae.png
    results/phase1_concept_features.csv
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
from crosssae.sae import TopKSAE, train_sae, encode_dataset
from crosssae.knockoffs import select_matches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "raw")

N_IMAGES = 2000
BLOCK = 6                 # which ViT block to read
D_HIDDEN = 512
K = 48
SAE_EPOCHS = 400
FDR_Q = 0.2
MIN_ACTIVE_FRAC = 0.005   # drop near-dead SAE features before matching
ACTS_CACHE = os.path.join(RESULTS, "phase1_acts.npz")


def get_vit_activations():
    import timm
    import torchvision
    from torchvision import transforms

    print("Loading pretrained ViT (vit_small_patch16_224)...")
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    print("Loading CIFAR-10 (real images, 10 semantic concepts)...")
    ds = torchvision.datasets.CIFAR10(root=DATA, train=False, download=True, transform=tfm)
    classes = ds.classes
    idx = np.random.default_rng(0).choice(len(ds), size=N_IMAGES, replace=False)

    feats = {}
    handle = model.blocks[BLOCK].register_forward_hook(lambda m, i, o: feats.__setitem__("a", o.detach()))

    acts, labels = [], []
    batch = 64
    print(f"Caching block-{BLOCK} activations over {N_IMAGES} images...")
    with torch.no_grad():
        for b0 in range(0, len(idx), batch):
            chunk = idx[b0:b0 + batch]
            x = torch.stack([ds[i][0] for i in chunk])
            model(x)
            acts.append(feats["a"].mean(dim=1).cpu())   # mean-pool tokens -> (B, d)
            labels.extend([ds[i][1] for i in chunk])
            print(f"  {min(b0 + batch, len(idx))}/{len(idx)}", end="\r")
    handle.remove()
    print()
    return torch.cat(acts).float(), np.array(labels), classes


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    torch.manual_seed(0)

    if os.path.exists(ACTS_CACHE):
        print(f"Loading cached ViT activations from {ACTS_CACHE}")
        cached = np.load(ACTS_CACHE, allow_pickle=True)
        acts = torch.from_numpy(cached["acts"]).float()
        labels = cached["labels"]
        classes = list(cached["classes"])
    else:
        acts, labels, classes = get_vit_activations()
        np.savez(ACTS_CACHE, acts=acts.numpy(), labels=labels, classes=np.array(classes))
    n, d_in = acts.shape
    print(f"Activations: {acts.shape}")

    print(f"Training Top-k SAE (d_hidden={D_HIDDEN}, k={K}) on real ViT activations...")
    sae = train_sae(acts, d_hidden=D_HIDDEN, k=K, seed=0, epochs=SAE_EPOCHS)
    Z = encode_dataset(sae, acts).numpy()         # (n, D_HIDDEN) feature activations

    # Reconstruction sanity (variance explained).
    with torch.no_grad():
        x_hat, _ = sae(acts)
        ss_res = ((acts - x_hat) ** 2).sum().item()
        ss_tot = ((acts - acts.mean(0)) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot

    # Drop near-dead features before matching.
    active_frac = (Z > 0).mean(0)
    keep = np.where(active_frac >= MIN_ACTIVE_FRAC)[0]
    Zk = Z[:, keep]
    print(f"SAE: R^2={r2:.3f}, {len(keep)}/{D_HIDDEN} features active >= {MIN_ACTIVE_FRAC:.0%}")

    # FDR-controlled matching: which SAE features are significantly associated
    # with each semantic concept? (model-side stand-in for brain-side matching)
    print(f"Running FDR-controlled feature<->concept matching (q={FDR_Q})...")
    rows = []
    sig_counts = np.zeros(len(classes), dtype=int)
    concept_feat = np.zeros((len(classes), len(keep)))
    for c, cname in enumerate(classes):
        y = (labels == c).astype(float)
        sel, W = select_matches(Zk, y, q=FDR_Q, rng=np.random.default_rng(100 + c))
        sig_counts[c] = len(sel)
        concept_feat[c, sel] = 1
        # record the strongest matched feature per concept
        if len(sel):
            best = sel[np.argmax(W[sel])]
            rows.append((cname, len(sel), int(keep[best]), float(W[best])))
        else:
            rows.append((cname, 0, -1, 0.0))
        print(f"  {cname:12s}: {len(sel):2d} significant SAE features")

    # --- artifact figure ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    order = np.argsort(-sig_counts)
    ax.barh([classes[i] for i in order][::-1], sig_counts[order][::-1], color="#1f4e79")
    ax.set_xlabel("# FDR-significant SAE features (q=%.2f)" % FDR_Q)
    ax.set_title("ViT-SAE features matched to each concept\n(real ViT, real images, knockoff-controlled)")

    ax = axes[1]
    # show concept x feature significance for features selected by >=1 concept
    used = np.where(concept_feat.sum(0) > 0)[0]
    M = concept_feat[:, used]
    im = ax.imshow(M, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("SAE feature (significant for >=1 concept)")
    ax.set_title("Concept <-> SAE-feature matches\n(blue = significant at FDR %.2f)" % FDR_Q)

    fig.suptitle(
        "PHASE 1 — real ViT-SAE feature<->concept matching with FDR control "
        f"(SAE R^2={r2:.2f}, {len(keep)} live features, n={n} images)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(RESULTS, "phase1_vit_sae.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved figure -> {out_png}")

    out_csv = os.path.join(RESULTS, "phase1_concept_features.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "n_significant_features", "top_feature_id", "top_W"])
        for r in rows:
            w.writerow(r)
    print(f"Saved table  -> {out_csv}")
    print(f"\nPHASE 1 complete: real ViT -> SAE (R^2={r2:.2f}) -> {int(sig_counts.sum())} "
          f"FDR-controlled feature<->concept matches across {len(classes)} concepts.")


if __name__ == "__main__":
    main()
