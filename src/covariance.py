"""Covariance estimation — Ledoit–Wolf shrinkage (implemented in NumPy).

The sample covariance matrix is unbiased but noisy: with N assets and T
observations its eigenvalues are systematically spread too wide, which is
why mean–variance optimisers built on it concentrate in whatever asset
happened to look quiet. Ledoit & Wolf (2004, *A Well-Conditioned
Estimator of Large-Dimensional Covariance Matrices*) shrink the sample
matrix S toward a scaled identity target F = m·I, where m = tr(S)/N:

    Σ_LW = (1 − δ)·S + δ·F,      δ ∈ [0, 1]

with the shrinkage intensity δ estimated from the data itself by
minimising the expected Frobenius loss. The estimator is implemented
here from the paper's formulas — no scikit-learn required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    annualise: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Ledoit–Wolf shrunk covariance of daily ``returns``.

    Parameters
    ----------
    returns:
        DataFrame of daily simple returns (T × N).
    annualise:
        Multiply the result by 252 (shrinkage intensity is scale-free).

    Returns
    -------
    (covariance, delta):
        The shrunk covariance as a DataFrame aligned to ``returns``
        columns, and the estimated shrinkage intensity δ ∈ [0, 1].
        δ = 0 means the sample matrix was kept; δ = 1 means full
        shrinkage to the scaled-identity target.
    """
    X = returns.to_numpy(dtype=float)
    T, N = X.shape
    if T < 3:
        raise ValueError("need at least 3 return observations")
    if N < 1:
        raise ValueError("returns must have at least one column")

    X = X - X.mean(axis=0, keepdims=True)

    # Sample covariance with the 1/T normalisation used by Ledoit–Wolf
    S = (X.T @ X) / T

    # Target: scaled identity with the same trace as S
    m = float(np.trace(S) / N)
    F = m * np.eye(N)

    # d² = ||S − F||²_F / N  (squared distance of S from the target)
    d2 = float(np.sum((S - F) ** 2) / N)
    if d2 <= 0:  # S already is a scaled identity
        cov = S * TRADING_DAYS if annualise else S
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns), 1.0

    # b̄² = (1 / (T²·N)) · Σ_t ||x_t x_tᵀ − S||²_F
    # Using tr((xxᵀ − S)²) = (xᵀx)² − 2·xᵀSx + ||S||²_F per observation.
    norms4 = np.sum(X**2, axis=1) ** 2
    xSx = np.einsum("ti,ij,tj->t", X, S, X)
    s_frob2 = float(np.sum(S**2))
    per_obs = norms4 - 2.0 * xSx + s_frob2
    b_bar2 = float(np.sum(per_obs) / (T**2 * N))

    b2 = min(b_bar2, d2)
    delta = float(np.clip(b2 / d2, 0.0, 1.0))

    shrunk = (1.0 - delta) * S + delta * F
    if annualise:
        shrunk = shrunk * TRADING_DAYS
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), delta
