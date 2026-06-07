"""
Real-data loaders for the model side (ViT/CLIP over images) and the brain side
(THINGS-EEG2 shared-stimulus responses).

These are intentionally thin and dependency-gated: the synthetic demo
(experiments/demo_fdr_matching.py) runs with zero extra installs, while the real
pipeline activates once you install the heavier deps:

    pip install datasets timm pillow

Why THINGS: the SAME 16,740 images were shown to humans (EEG/MEG/fMRI) AND can be
fed to a vision transformer -> a shared-stimulus design, which is exactly what
makes per-feature cross-domain matching well-posed.

  THINGS-EEG2 : https://huggingface.co/datasets/gasparyanartur/things-eeg2
  THINGS-MEG  : https://openneuro.org/datasets/ds004212
"""
from __future__ import annotations

import numpy as np


def load_vit_activations(images, layer: str = "blocks.6", model_name: str = "vit_base_patch16_224"):
    """Cache mean-pooled ViT activations at a given block for a list of PIL images.

    Returns a (n_images, d_model) float array. Requires `timm` + `torch`.
    """
    import torch
    import timm
    from timm.data import resolve_data_config, create_transform

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    cfg = resolve_data_config({}, model=model)
    tfm = create_transform(**cfg)

    feats = {}
    def hook(_m, _i, o):
        feats["act"] = o.detach()
    dict(model.named_modules())[layer].register_forward_hook(hook)

    out = []
    with torch.no_grad():
        for img in images:
            x = tfm(img.convert("RGB")).unsqueeze(0)
            model(x)
            a = feats["act"]                      # (1, tokens, d) for a ViT block
            out.append(a.mean(dim=1).squeeze(0).numpy())
    return np.asarray(out)


def load_things_eeg2(subject: int = 1, window=(0.05, 0.30), split: str = "test"):
    """Load THINGS-EEG2 responses as a (n_conditions, d_channels*time) matrix
    aligned to the image conditions. Requires `datasets`.

    The returned `image_ids` let you align brain responses to the SAME images fed
    to `load_vit_activations`, giving the shared-stimulus design the matching
    procedure needs. Window is in seconds post-stimulus over visual ERP.
    """
    from datasets import load_dataset

    ds = load_dataset("gasparyanartur/things-eeg2", split=split)
    ds = ds.filter(lambda r: r.get("subject", subject) == subject)
    X, image_ids = [], []
    for r in ds:
        eeg = np.asarray(r["eeg"])                # (channels, time)
        X.append(eeg.reshape(-1))
        image_ids.append(r["image_id"])
    return np.asarray(X), image_ids


def synthetic_fallback(n=3000, d_model=128, d_brain=80, seed=0):
    """Zero-dependency stand-in so the pipeline is end-to-end runnable offline.
    See crosssae.synthetic for the planted-ground-truth version used in the demo.
    """
    from .synthetic import make_planted_matching
    rng = np.random.default_rng(seed)
    X_brain, targets = make_planted_matching(n=n, p_brain=d_brain, rng=rng)
    return X_brain, targets
