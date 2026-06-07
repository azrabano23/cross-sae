"""
Gaussian Model-X knockoffs for FDR-controlled cross-domain SAE feature matching.

This is the statistical heart of the project. Given a single "target" feature
(e.g. one model-side SAE feature) as the response y, and a bank of candidate
features (e.g. all brain-side SAE features) as covariates X, we want to select
the subset of candidates genuinely associated with the target while controlling
the false discovery rate (FDR) at a user-chosen level q.

HONESTY NOTE (do not delete — this is the project's integrity boundary):
Model-X knockoffs give *exact, finite-sample* FDR control only when the joint
distribution of the covariates is KNOWN (Candès et al., 2018, JRSSB). Here we
use second-order (Gaussian, mean+covariance-matched) knockoffs estimated from
data. On sparse, non-negative, heavy-tailed SAE latents this is a known
misspecification, so the guarantee is APPROXIMATE / ASYMPTOTIC, not exact
(see Barber, Candès & Samworth, 2020; Fan et al., 2025, arXiv:2502.05969).
Validating that control actually holds empirically on real SAE/brain latents is
itself a contribution of this project, not an assumption we get for free.

References:
  Candès, Fan, Janson, Lv (2018). "Panning for gold: Model-X knockoffs for
    high-dimensional controlled variable selection." JRSSB 80(3):551-577.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import eigvalsh, inv, cholesky
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _nearest_psd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix (clip eigenvalues)."""
    mat = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def gaussian_knockoffs(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Construct second-order (equicorrelated) Gaussian Model-X knockoffs.

    The knockoff matrix X_tilde satisfies, to second order, the pairwise
    exchangeability property: swapping any feature with its knockoff leaves the
    first two moments of (X, X_tilde) unchanged, while X_tilde is conditionally
    independent of the response given X.

    Parameters
    ----------
    X : (n, p) array of covariates.
    rng : numpy Generator for reproducibility.

    Returns
    -------
    X_tilde : (n, p) knockoff matrix.
    """
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    # Work in correlation scale for numerical stability, rescale at the end.
    sd = Xc.std(axis=0) + 1e-8
    Z = Xc / sd
    Sigma = np.corrcoef(Z, rowvar=False)
    Sigma = _nearest_psd(Sigma)

    # Equicorrelated construction: s_j = s for all j, s = min(1, 2*lambda_min).
    lam_min = float(np.min(eigvalsh(Sigma)))
    s_val = min(1.0, max(0.0, 2.0 * lam_min))
    s = np.full(p, s_val)
    Sinv = inv(Sigma)

    # Conditional mean: mu_tilde = X - (X - mu) Sigma^{-1} diag(s)
    diag_s = np.diag(s)
    cond_mean = Z - Z @ (Sinv @ diag_s)
    # Conditional covariance: V = 2 diag(s) - diag(s) Sigma^{-1} diag(s)
    V = 2.0 * diag_s - diag_s @ Sinv @ diag_s
    V = _nearest_psd(V)
    L = cholesky(V)

    noise = rng.standard_normal((n, p)) @ L.T
    Z_tilde = cond_mean + noise
    # Rescale back to the original feature scale and re-add the mean.
    return Z_tilde * sd + mu


def _lasso_importance(X: np.ndarray, X_tilde: np.ndarray, y: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Lasso signed-max (LSM) knockoff statistic W_j = |b_j| - |b_{j+p}|.

    Fit a single Lasso on the augmented design [X, X_tilde] -> y. The statistic
    is positive when the real feature is more important than its knockoff.
    """
    p = X.shape[1]
    Xaug = np.hstack([X, X_tilde])
    # Standardize columns so the L1 penalty is applied fairly.
    col_sd = Xaug.std(axis=0) + 1e-8
    Xaug = (Xaug - Xaug.mean(axis=0)) / col_sd
    seed = int(rng.integers(0, 2**31 - 1))
    model = LassoCV(cv=3, n_alphas=30, max_iter=5000, random_state=seed, n_jobs=1)
    model.fit(Xaug, (y - y.mean()) / (y.std() + 1e-8))
    coef = np.abs(model.coef_)
    return coef[:p] - coef[p:]


def knockoff_threshold(W: np.ndarray, q: float, offset: int = 1) -> float:
    """Knockoff(+) selection threshold for target FDR level q.

    offset=1 is the knockoff+ variant (conservative, finite-sample valid under
    the model assumptions); offset=0 is the plain knockoff filter.
    """
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    for t in ts:
        denom = max(1, int(np.sum(W >= t)))
        ratio = (offset + int(np.sum(W <= -t))) / denom
        if ratio <= q:
            return float(t)
    return float("inf")


def select_matches(X: np.ndarray, y: np.ndarray, q: float = 0.1,
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """FDR-controlled selection of candidate features associated with target y.

    Returns
    -------
    selected : int array of selected candidate indices.
    W : the per-candidate knockoff statistics (for inspection / plotting).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X_tilde = gaussian_knockoffs(X, rng)
    W = _lasso_importance(X, X_tilde, y, rng)
    tau = knockoff_threshold(W, q, offset=1)
    selected = np.where(W >= tau)[0]
    return selected, W
