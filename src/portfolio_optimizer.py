"""
Portfolio Optimizer — Modern Portfolio Theory toolkit.

Features
--------
- Synthetic (offline) price / return generation with a seeded RNG
- Optional live download via pandas (yfinance-compatible interface is left
  as an extension point — the core never *requires* the internet)
- Expected returns and covariance estimation
- Portfolio statistics: return, volatility, Sharpe ratio
- Random portfolio generation, efficient-frontier sampling
- Optimisation: maximum-Sharpe and minimum-volatility portfolios
  (SciPy SLSQP when available, NumPy random-search + refinement fallback)
- Monte Carlo simulation of portfolio wealth paths

Conventions
-----------
- Prices are daily closes, returns are simple daily returns.
- Annualisation uses 252 trading days.
- Risk-free rate defaults to 0.0 and is expressed as an annual rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only without scipy
    minimize = None  # type: ignore[assignment]
    _HAS_SCIPY = False

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data generation / loading
# ---------------------------------------------------------------------------

def generate_synthetic_prices(
    tickers: list[str] | None = None,
    n_days: int = 504,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic daily closing prices with a seeded RNG.

    Each asset follows geometric Brownian motion with a distinct drift and
    volatility, plus a shared market factor so assets are correlated —
    which is what makes diversification / optimisation non-trivial.

    Works fully offline and is deterministic for a given ``seed``.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM"]
    if n_days < 2:
        raise ValueError("n_days must be >= 2")
    if not tickers:
        raise ValueError("tickers must be non-empty")

    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    # Per-asset annual drift / vol, drawn once (deterministic given seed)
    annual_mu = rng.uniform(0.04, 0.18, size=n_assets)
    annual_sigma = rng.uniform(0.12, 0.35, size=n_assets)

    daily_mu = annual_mu / TRADING_DAYS
    daily_sigma = annual_sigma / np.sqrt(TRADING_DAYS)

    # Shared market factor + idiosyncratic shocks -> correlated returns
    market = rng.standard_normal(n_days)
    idio = rng.standard_normal((n_days, n_assets))
    loadings = rng.uniform(0.3, 0.7, size=n_assets)
    shocks = loadings * market[:, None] + np.sqrt(1.0 - loadings**2) * idio

    log_returns = daily_mu + daily_sigma * shocks
    prices = start_price * np.exp(np.cumsum(log_returns, axis=0))
    # Prepend the starting price row so day-0 is exactly start_price-scaled
    prices = np.vstack([np.full((1, n_assets), float(start_price)), prices])

    index = pd.bdate_range("2023-01-02", periods=len(prices))
    return pd.DataFrame(prices, index=index, columns=tickers)


def download_prices(
    tickers: list[str],
    start: str = "2023-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily closes with yfinance if it is installed.

    This is an optional convenience wrapper. The rest of the toolkit —
    and the entire test-suite / CLI demo — works offline via
    :func:`generate_synthetic_prices`, so no network access is required.

    Raises
    ------
    ImportError
        If ``yfinance`` is not installed.
    """
    try:
        import yfinance as yf  # local import: optional dependency
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError(
            "download_prices requires the optional 'yfinance' package. "
            "Install it with `pip install yfinance`, or use "
            "generate_synthetic_prices() for offline work."
        ) from exc

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return prices.dropna(how="all")


# ---------------------------------------------------------------------------
# Returns & risk model
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price DataFrame (first NaN row dropped)."""
    if prices.shape[0] < 2:
        raise ValueError("prices must have at least 2 rows")
    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("no returns could be computed from prices")
    return returns


def expected_returns(returns: pd.DataFrame, annualise: bool = True) -> pd.Series:
    """Mean return per asset (annualised by default)."""
    mu = returns.mean()
    return mu * TRADING_DAYS if annualise else mu


def covariance_matrix(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Sample covariance matrix of returns (annualised by default)."""
    cov = returns.cov()
    return cov * TRADING_DAYS if annualise else cov


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

@dataclass
class PortfolioStats:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_return": float(self.expected_return),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
        }


def _validate_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float).flatten()
    if w.shape != (n_assets,):
        raise ValueError(f"weights must have length {n_assets}, got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    return w


def portfolio_stats(
    weights: np.ndarray,
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
) -> PortfolioStats:
    """Expected return, volatility and Sharpe ratio for a weight vector.

    Maths (annualised inputs):
        return     = wᵀ μ
        variance   = wᵀ Σ w
        volatility = sqrt(variance)
        Sharpe     = (return − rf) / volatility
    """
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    w = _validate_weights(weights, len(mu_arr))
    if cov_arr.shape != (len(mu_arr), len(mu_arr)):
        raise ValueError("covariance matrix shape does not match mu")

    port_return = float(w @ mu_arr)
    variance = float(w @ cov_arr @ w)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (port_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    return PortfolioStats(port_return, volatility, float(sharpe), w)


# ---------------------------------------------------------------------------
# Random portfolios & efficient frontier
# ---------------------------------------------------------------------------

def generate_random_portfolios(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample random long-only, fully-invested portfolios.

    Returns a DataFrame with columns: return, volatility, sharpe,
    and one ``w_<i>`` column per asset weight.
    """
    mu_arr = np.asarray(mu, dtype=float)
    n_assets = len(mu_arr)
    rng = np.random.default_rng(seed)
    weights = rng.random((n_portfolios, n_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    rows = []
    for w in weights:
        s = portfolio_stats(w, mu_arr, cov, risk_free_rate)
        rows.append([s.expected_return, s.volatility, s.sharpe_ratio, *w])
    cols = ["return", "volatility", "sharpe"] + [f"w_{i}" for i in range(n_assets)]
    return pd.DataFrame(rows, columns=cols)


def efficient_frontier(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    n_points: int = 25,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute efficient-frontier points (min vol for target returns).

    Falls back to a sorted random-portfolio frontier approximation when
    SciPy is unavailable.
    """
    mu_arr = np.asarray(mu, dtype=float)
    min_ret, max_ret = float(mu_arr.min()), float(mu_arr.max())
    targets = np.linspace(min_ret, max_ret, n_points)

    if not _HAS_SCIPY:
        # Approximation: bin random portfolios by return, keep min vol
        sampled = generate_random_portfolios(mu_arr, cov, n_portfolios=8000,
                                             risk_free_rate=risk_free_rate)
        sampled = sampled.sort_values("volatility")
        frontier = []
        for t in targets:
            near = sampled[np.abs(sampled["return"] - t) < (max_ret - min_ret) / n_points + 1e-9]
            if not near.empty:
                best = near.iloc[0]
                frontier.append({"target_return": t, "return": best["return"],
                                 "volatility": best["volatility"],
                                 "sharpe": best["sharpe"]})
        return pd.DataFrame(frontier)

    cov_arr = np.asarray(cov, dtype=float)
    n = len(mu_arr)
    rows = []
    for target in targets:
        res = _solve(
            lambda w: float(w @ cov_arr @ w),
            n,
            extra_constraints=[{"type": "eq", "fun": lambda w, t=target: float(w @ mu_arr - t)}],
        )
        if res is None:
            continue
        s = portfolio_stats(res, mu_arr, cov_arr, risk_free_rate)
        rows.append({"target_return": float(target), "return": s.expected_return,
                     "volatility": s.volatility, "sharpe": s.sharpe_ratio})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def _solve(objective, n_assets: int, extra_constraints=None):
    """SLSQP solve with long-only / fully-invested constraints, or fallback."""
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    bounds = [(0.0, 1.0)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)

    if _HAS_SCIPY:
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                          constraints=constraints)
        if result.success or result.fun is not None:
            w = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
            total = w.sum()
            return w / total if total > 0 else x0
        return None

    # Fallback: random search + local refinement (no extra constraints support
    # beyond return-target-free objectives used by the public optimisers)
    if extra_constraints:
        return None
    rng = np.random.default_rng(123)
    best_w, best_val = x0, objective(x0)
    candidates = rng.random((20000, n_assets))
    candidates /= candidates.sum(axis=1, keepdims=True)
    for w in candidates:
        val = objective(w)
        if val < best_val:
            best_w, best_val = w, val
    # Refinement: perturb the incumbent, keep improvements
    for scale in (0.10, 0.05, 0.02, 0.01):
        for _ in range(4000):
            w = np.clip(best_w + rng.normal(0, scale, size=n_assets), 0, None)
            if w.sum() == 0:
                continue
            w /= w.sum()
            val = objective(w)
            if val < best_val:
                best_w, best_val = w, val
    return best_w


def max_sharpe_portfolio(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
) -> PortfolioStats:
    """Long-only portfolio maximising the Sharpe ratio."""
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(float(w @ cov_arr @ w), 1e-18))
        return -float((w @ mu_arr - risk_free_rate) / vol)

    w = _solve(neg_sharpe, len(mu_arr))
    if w is None:  # pragma: no cover
        w = np.full(len(mu_arr), 1.0 / len(mu_arr))
    return portfolio_stats(w, mu_arr, cov_arr, risk_free_rate)


def min_volatility_portfolio(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
) -> PortfolioStats:
    """Long-only portfolio minimising variance."""
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)

    def variance(w: np.ndarray) -> float:
        return float(w @ cov_arr @ w)

    w = _solve(variance, len(mu_arr))
    if w is None:  # pragma: no cover
        w = np.full(len(mu_arr), 1.0 / len(mu_arr))
    return portfolio_stats(w, mu_arr, cov_arr, risk_free_rate)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def monte_carlo_simulation(
    weights: np.ndarray,
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    n_simulations: int = 1000,
    n_days: int = 252,
    initial_value: float = 10_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate portfolio wealth paths from annualised mu / cov.

    Returns a DataFrame of shape (n_days + 1, n_simulations); row 0 is
    ``initial_value`` for every path.
    """
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    w = _validate_weights(weights, len(mu_arr))

    daily_mu = mu_arr / TRADING_DAYS
    daily_cov = cov_arr / TRADING_DAYS
    # Nearest PSD matrix guards against tiny negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(daily_cov)
    daily_cov_psd = eigvecs @ np.diag(np.clip(eigvals, 0, None)) @ eigvecs.T

    rng = np.random.default_rng(seed)
    asset_returns = rng.multivariate_normal(
        daily_mu, daily_cov_psd, size=(n_simulations, n_days)
    )
    port_daily = asset_returns @ w
    wealth = initial_value * np.cumprod(1.0 + port_daily, axis=1)
    wealth = np.hstack([np.full((n_simulations, 1), float(initial_value)), wealth])
    return pd.DataFrame(wealth.T)


def summarise_simulation(wealth: pd.DataFrame, initial_value: float = 10_000.0) -> dict[str, float]:
    """Percentile summary of final wealth from a Monte Carlo run."""
    final = wealth.iloc[-1].to_numpy(dtype=float)
    return {
        "initial_value": float(initial_value),
        "median_final": float(np.median(final)),
        "p5_final": float(np.percentile(final, 5)),
        "p95_final": float(np.percentile(final, 95)),
        "prob_loss": float(np.mean(final < initial_value)),
    }
