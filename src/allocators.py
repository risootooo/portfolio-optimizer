"""Advanced portfolio allocators.

Beyond Markowitz max-Sharpe / min-volatility (in
:mod:`portfolio_optimizer`), this module implements the three allocation
frameworks a modern quant desk actually reaches for:

- **Risk Parity** — equal risk contribution (ERC). Each asset contributes
  the same share of total portfolio variance. Solved with Spinu's convex
  log-barrier reformulation, so the result is a true optimum, not a
  fixed-point heuristic.
- **Hierarchical Risk Parity (HRP)** — López de Prado (2016). Cluster
  assets on their correlation distance, quasi-diagonalise the covariance
  matrix, then allocate by recursive bisection with inverse-variance
  weighting inside clusters. No matrix inversion, no expected returns —
  markedly more stable out of sample than Markowitz.
- **Black–Litterman** — start from market-equilibrium implied returns,
  blend in investor views (P, Q) with uncertainty Ω, and optimise the
  posterior. Views move the answer smoothly instead of hijacking it.

All allocators return long-only, fully-invested weight vectors and use
only NumPy / SciPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

try:  # src/ on sys.path (CLI + tests)
    from portfolio_optimizer import PortfolioStats, portfolio_stats
except ImportError:  # package import
    from .portfolio_optimizer import PortfolioStats, portfolio_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _clean_weights(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    if total <= 0:
        raise ValueError("allocator produced a zero weight vector")
    return w / total


def equal_weight(n_assets: int) -> np.ndarray:
    """The 1/N benchmark."""
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return np.full(n_assets, 1.0 / n_assets)


def risk_contributions(weights, cov) -> np.ndarray:
    """Absolute risk contributions: RC_i = w_i · (Σw)_i, Σ RC_i = σ_p²."""
    w = _as_array(weights)
    cov_arr = _as_array(cov)
    return w * (cov_arr @ w)


def risk_contribution_shares(weights, cov) -> np.ndarray:
    """Risk contributions as shares of total variance (sum to 1)."""
    rc = risk_contributions(weights, cov)
    total = rc.sum()
    if total <= 0:
        raise ValueError("portfolio variance must be positive")
    return rc / total


# ---------------------------------------------------------------------------
# Risk Parity (equal risk contribution)
# ---------------------------------------------------------------------------

def risk_parity_portfolio(cov, risk_budgets=None) -> np.ndarray:
    """Risk-parity weights: every asset contributes equally to variance.

    Solves Spinu's convex reformulation

        minimise  ½·xᵀΣx − Σ_i b_i·log(x_i)     over x > 0,

    whose first-order condition is exactly x_i·(Σx)_i = b_i; normalising
    x to sum 1 then gives a portfolio whose risk-contribution shares
    equal the budgets ``b`` (uniform by default).
    """
    cov_arr = _as_array(cov)
    n = cov_arr.shape[0]
    if cov_arr.shape != (n, n):
        raise ValueError("cov must be square")
    if risk_budgets is None:
        budgets = np.full(n, 1.0 / n)
    else:
        budgets = _as_array(risk_budgets)
        if budgets.shape != (n,) or (budgets <= 0).any():
            raise ValueError("risk_budgets must be a positive vector of length n")
        budgets = budgets / budgets.sum()

    def objective(x: np.ndarray) -> float:
        return 0.5 * float(x @ cov_arr @ x) - float(budgets @ np.log(x))

    def gradient(x: np.ndarray) -> np.ndarray:
        return cov_arr @ x - budgets / x

    # Start from the inverse-volatility portfolio (already close)
    ivp = 1.0 / np.sqrt(np.diag(cov_arr))
    x0 = ivp / ivp.sum()
    result = minimize(objective, x0, jac=gradient, method="L-BFGS-B",
                      bounds=[(1e-10, None)] * n)
    if not result.success and result.fun is None:  # pragma: no cover
        raise RuntimeError("risk parity optimisation failed")
    return _clean_weights(result.x)


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------

def _cluster_variance(cov_arr: np.ndarray, indices: list[int]) -> float:
    """Variance of the inverse-variance portfolio over a cluster."""
    sub = cov_arr[np.ix_(indices, indices)]
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_portfolio(cov) -> np.ndarray:
    """Hierarchical Risk Parity weights (López de Prado, 2016).

    Steps: (1) correlation → distance d_ij = √(½(1−ρ_ij)); (2) single-
    linkage hierarchical clustering; (3) quasi-diagonal ordering from
    the dendrogram; (4) recursive bisection — at each split, capital is
    divided between the two child clusters in inverse proportion to
    their inverse-variance-portfolio variances.
    """
    cov_arr = _as_array(cov)
    n = cov_arr.shape[0]
    if cov_arr.shape != (n, n):
        raise ValueError("cov must be square")
    if n == 1:
        return np.array([1.0])

    vols = np.sqrt(np.diag(cov_arr))
    corr = cov_arr / np.outer(vols, vols)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = [int(i) for i in leaves_list(link)]  # quasi-diagonal order

    weights = np.ones(n)
    clusters: list[list[int]] = [order]
    while clusters:
        # Split every cluster of size > 1 into two halves
        clusters = [
            part
            for cluster in clusters
            for part in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2:])
            if len(cluster) > 1
        ]
        for i in range(0, len(clusters) - 1, 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov_arr, left)
            var_right = _cluster_variance(cov_arr, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= 1.0 - alpha
    return _clean_weights(weights)


# ---------------------------------------------------------------------------
# Black–Litterman
# ---------------------------------------------------------------------------

def black_litterman_posterior(
    cov,
    market_weights,
    P=None,
    Q=None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    omega=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black–Litterman posterior expected returns.

    Equilibrium returns are implied by reverse optimisation,
    Π = λ·Σ·w_mkt. Views (P·μ = Q) are blended with confidence Ω
    (default: Ω = diag(P·(τΣ)·Pᵀ), the Idzorek-style proportional
    choice). Returns ``(posterior_mu, posterior_cov, equilibrium_mu)``,
    all on the same (annualised) scale as ``cov``.
    """
    cov_arr = _as_array(cov)
    n = cov_arr.shape[0]
    w_mkt = _as_array(market_weights)
    if w_mkt.shape != (n,):
        raise ValueError("market_weights length must match cov")

    equilibrium = risk_aversion * (cov_arr @ w_mkt)
    if P is None or Q is None:
        return equilibrium.copy(), cov_arr.copy(), equilibrium

    P_arr = np.atleast_2d(_as_array(P))
    Q_arr = _as_array(Q).flatten()
    if P_arr.shape[1] != n or P_arr.shape[0] != len(Q_arr):
        raise ValueError("P must be (n_views × n_assets) and Q of length n_views")

    tau_cov = tau * cov_arr
    if omega is None:
        omega_arr = np.diag(np.diag(P_arr @ tau_cov @ P_arr.T))
    else:
        omega_arr = _as_array(omega)

    tau_cov_inv = np.linalg.pinv(tau_cov)
    omega_inv = np.linalg.pinv(omega_arr)
    middle_inv = np.linalg.pinv(tau_cov_inv + P_arr.T @ omega_inv @ P_arr)
    posterior_mu = middle_inv @ (tau_cov_inv @ equilibrium + P_arr.T @ omega_inv @ Q_arr)
    posterior_cov = cov_arr + middle_inv
    return posterior_mu, posterior_cov, equilibrium


def black_litterman_portfolio(
    cov,
    market_weights,
    P=None,
    Q=None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    omega=None,
    risk_free_rate: float = 0.0,
) -> PortfolioStats:
    """Mean–variance optimum under Black–Litterman posterior returns.

    Maximises wᵀμ_BL − (λ/2)·wᵀΣ_post·w over long-only, fully-invested
    weights (SLSQP), returning full :class:`PortfolioStats` evaluated on
    the posterior moments.
    """
    posterior_mu, posterior_cov, _ = black_litterman_posterior(
        cov, market_weights, P=P, Q=Q, tau=tau,
        risk_aversion=risk_aversion, omega=omega,
    )
    n = len(posterior_mu)

    def neg_utility(w: np.ndarray) -> float:
        return -float(w @ posterior_mu - 0.5 * risk_aversion * (w @ posterior_cov @ w))

    result = minimize(
        neg_utility, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    w = _clean_weights(result.x if result.x is not None else np.full(n, 1.0 / n))
    stats = portfolio_stats(w, posterior_mu, posterior_cov, risk_free_rate)
    return stats
