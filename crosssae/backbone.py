"""
Brain-side representation backbones — the Tarjuman extension to cross-sae.

cross-sae's headline finding (FINDINGS.md §6) is a *controlled negative*: raw
ViT<->EEG representations share significant structure (RSA rho=+0.155, p=0.0005),
but training a Top-k SAE directly on *raw* binned EEG attenuates that shared
structure below significance (rho=+0.067, p=0.058). The sparse basis, learned on
raw scalp EEG, is not the right canonical unit for cross-domain matching.

Tarjuman's central, falsifiable hypothesis follows directly:

    A neural FOUNDATION MODEL latent space is a learned, denoised, semantically
    organized representation of the EEG. Training the brain-side SAE on FM latents
    (instead of raw EEG) should RECOVER the shared cross-domain structure that the
    raw-EEG SAE discards — making FM-latent SAE features a viable interlingua for
    FDR-controlled model<->brain matching.

This module provides the brain-side backbone abstraction that lets us test that
hypothesis as a clean, controlled comparison:

    raw EEG  --[backbone]-->  brain representation  --[Top-k SAE]-->  brain features

Three backbones, sharing one interface, so the ONLY thing that varies between
arms of the experiment is the representation the SAE sees:

  * IdentityBackbone   passthrough  == the current raw-EEG baseline (known null)
  * PCABackbone        linear embedding control (cheap, offline, no extra deps)
                       — rules out "any dimensionality reduction helps"
  * EEGFoundationBackbone   a pretrained EEG foundation model (CBraMod / LaBraM)
                       — the hypothesis arm; the learned, biologically-pretrained
                       latent space

HONESTY BOUNDARY (do not soften — mirrors crosssae.knockoffs):
  - Identity and PCA run fully offline today. The FM arm REQUIRES real pretrained
    weights; if they are absent it is SKIPPED with a loud message, never silently
    replaced by a random embedding (an untrained embedding would prove nothing and
    would be dishonest as an "FM" result).
  - The CBraMod/LaBraM adapters below are an INTEGRATION SCAFFOLD: the tensor
    shapes / sampling-rate handling are written to the released checkpoints' public
    interface but MUST be validated against the actual weights before any result is
    reported. They are marked accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
#  Interface                                                                    #
# --------------------------------------------------------------------------- #
class BrainBackbone:
    """Maps a brain-side input to a representation the SAE is trained on.

    Two accepted input layouts (a backbone declares which it needs via
    `wants_raw_windows`):
      - flattened features : (n_trials, d)            e.g. binned EEG, 63*bins
      - raw windows        : (n_trials, n_channels, n_time)  for FM tokenizers
    Returns (n_trials, d_out) float32.
    """

    name = "base"
    wants_raw_windows = False

    def fit(self, X: np.ndarray) -> "BrainBackbone":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
#  1. Identity — the raw-EEG baseline (reproduces the known null)               #
# --------------------------------------------------------------------------- #
class IdentityBackbone(BrainBackbone):
    name = "identity"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  2. PCA — linear-embedding control                                            #
# --------------------------------------------------------------------------- #
class PCABackbone(BrainBackbone):
    """Linear control: if the FM arm beats *this*, the gain is not merely from
    moving to a lower-dim, decorrelated space — it is from the FM's learned,
    nonlinear, biologically-pretrained structure."""

    name = "pca"

    def __init__(self, n_components: int = 128, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten
        self._pca = None

    def fit(self, X: np.ndarray) -> "PCABackbone":
        from sklearn.decomposition import PCA

        k = int(min(self.n_components, X.shape[0], X.shape[1]))
        self._pca = PCA(n_components=k, whiten=self.whiten, random_state=0).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.transform(X).astype(np.float32)


# --------------------------------------------------------------------------- #
#  3. EEG foundation model — the hypothesis arm                                 #
# --------------------------------------------------------------------------- #
@dataclass
class FMConfig:
    """Where to find / how to run the pretrained EEG foundation model."""
    model: str = "cbramod"          # "cbramod" | "labram"
    weights_path: str | None = None  # local checkpoint; None -> try HF hub id below
    hf_id: str | None = None
    sfreq_in: int = 1000             # THINGS-EEG2 raw sampling rate (Hz)
    sfreq_model: int = 200           # CBraMod operates at 200 Hz, 1 s patches
    patch_seconds: float = 1.0
    pool: str = "mean"               # how to pool (channels, patches, d) -> (d,)
    device: str = "cpu"


class EEGFoundationBackbone(BrainBackbone):
    """Encode raw EEG windows with a pretrained EEG foundation model.

    INTEGRATION SCAFFOLD — the forward-pass shape handling targets CBraMod's public
    interface (Wang et al., ICLR 2025; patches of 200 samples @ 200 Hz, criss-cross
    transformer over a (batch, channels, patches, patch_len) tensor). Validate the
    exact I/O against the released checkpoint before reporting any number.

    Raises a clear error (never a silent fallback) if weights are unavailable.
    """

    name = "eeg-fm"
    wants_raw_windows = True

    def __init__(self, cfg: FMConfig | None = None):
        self.cfg = cfg or FMConfig()
        self._model = None
        self._torch = None

    # -- weight loading -------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "EEGFoundationBackbone needs torch. `pip install torch`."
            ) from e
        self._torch = torch

        path = self.cfg.weights_path
        if path is None:
            raise RuntimeError(
                "No EEG-FM weights provided. Tarjuman will NOT fabricate an FM arm "
                "with an untrained network.\n"
                "  Get CBraMod:  https://github.com/wjq-learning/CBraMod  (ICLR 2025)\n"
                "  then pass FMConfig(weights_path='/path/to/cbramod.pth').\n"
                "  LaBraM:       https://github.com/935963004/LaBraM"
            )
        # The concrete nn.Module is defined by the checkpoint's repo; we load the
        # released architecture and state dict. Kept thin + explicit on purpose.
        self._model = self._build_arch()
        state = torch.load(path, map_location=self.cfg.device)
        state = state.get("model", state)
        self._model.load_state_dict(state, strict=False)
        self._model.eval().to(self.cfg.device)

    def _build_arch(self):  # pragma: no cover - depends on external repo
        if self.cfg.model == "cbramod":
            try:
                from cbramod import CBraMod  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Install the CBraMod package/repo so `from cbramod import "
                    "CBraMod` works, or vendor its model.py."
                ) from e
            return CBraMod()
        if self.cfg.model == "labram":
            try:
                from labram import labram_base_patch200_200  # type: ignore
            except Exception as e:
                raise RuntimeError("Install LaBraM to use model='labram'.") from e
            return labram_base_patch200_200()
        raise ValueError(f"unknown FM model {self.cfg.model!r}")

    # -- preprocessing --------------------------------------------------------
    def _to_patches(self, X: np.ndarray):
        """(n, channels, time @ sfreq_in) -> (n, channels, patches, patch_len @ sfreq_model)."""
        import numpy as np
        from math import gcd

        torch = self._torch
        n, c, t = X.shape
        # resample time axis sfreq_in -> sfreq_model via linear interpolation
        t_new = int(round(t * self.cfg.sfreq_model / self.cfg.sfreq_in))
        xt = torch.from_numpy(np.ascontiguousarray(X)).float().reshape(n * c, 1, t)
        xt = torch.nn.functional.interpolate(xt, size=t_new, mode="linear",
                                             align_corners=False)
        xt = xt.reshape(n, c, t_new)
        patch_len = int(self.cfg.patch_seconds * self.cfg.sfreq_model)
        n_patches = t_new // patch_len
        xt = xt[:, :, : n_patches * patch_len].reshape(n, c, n_patches, patch_len)
        # per-patch standardization (CBraMod expects roughly unit-scale patches)
        m = xt.mean(-1, keepdim=True)
        s = xt.std(-1, keepdim=True) + 1e-8
        return (xt - m) / s

    # -- public API -----------------------------------------------------------
    @property
    def torch_no_grad(self):
        return self._torch.no_grad()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n_trials, n_channels, n_time) raw windows -> (n_trials, d_out)."""
        self._load()
        if X.ndim != 3:
            raise ValueError(
                "EEGFoundationBackbone needs raw windows (n, channels, time); "
                f"got shape {X.shape}. Use load_epochs_raw() (see experiments)."
            )
        patches = self._to_patches(X)
        outs = []
        with self.torch_no_grad:
            for i in range(0, patches.shape[0], 64):
                feats = self._model(patches[i:i + 64].to(self.cfg.device))
                # CBraMod returns (batch, channels, patches, d_model). Pool to (b, d).
                if hasattr(feats, "ndim") and feats.ndim == 4:
                    feats = feats.mean(dim=(1, 2)) if self.cfg.pool == "mean" else \
                            feats.amax(dim=(1, 2))
                elif feats.ndim == 3:
                    feats = feats.mean(dim=1)
                outs.append(feats.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Factory                                                                       #
# --------------------------------------------------------------------------- #
def get_backbone(name: str, **kw) -> BrainBackbone:
    name = name.lower()
    if name in ("identity", "raw"):
        return IdentityBackbone()
    if name == "pca":
        return PCABackbone(**kw)
    if name in ("eeg-fm", "fm", "cbramod", "labram"):
        cfg = kw.pop("cfg", None) or FMConfig(model="labram" if name == "labram"
                                              else "cbramod", **kw)
        return EEGFoundationBackbone(cfg)
    raise ValueError(f"unknown backbone {name!r}")
