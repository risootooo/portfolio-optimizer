"""Walk-forward backtest — do the fancy allocators actually earn it?

Protocol (expanding window, no look-ahead):

1. Start with ``train_size`` days of history.
2. Estimate expected returns and a Ledoit–Wolf-shrunk covariance on
   *everything seen so far*.
3. Compute each strategy's weights from that window only.
4. Hold those weights for the next ``step`` days and record the
   out-of-sample returns.
5. Expand the window and repeat until the data runs out.

Strategies compared: Equal Weight (1/N), Max-Sharpe, Min-Volatility,
Risk Parity and HRP — all long-only, fully invested, rebalanced only at
window boundaries. Reported metrics: annual return, annual volatility,
Sharpe, Sortino, maximum drawdown and Calmar, all from the realised
out-of-sample return stream (never from in-sample fitted moments).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # src/ on sys.path (CLI + tests)
    from portfolio_optimizer import (
        TRADING_DAYS,
        compute_returns,
        covariance_matrix,
        expected_returns,
        generate_synthetic_prices,
        max_sharpe_portfolio,
        min_volatility_portfolio,
    )
    from allocators import equal_weight, hrp_portfolio, risk_parity_portfolio
    from covariance import ledoit_wolf_shrinkage
except ImportError:  # package import
    from .portfolio_optimizer import (
        TRADING_DAYS,
        compute_returns,
        covariance_matrix,
        expected_returns,
        generate_synthetic_prices,
        max_sharpe_portfolio,
        min_volatility_portfolio,
    )
    from .allocators import equal_weight, hrp_portfolio, risk_parity_portfolio
    from .covariance import ledoit_wolf_shrinkage

STRATEGIES = ["Equal Weight", "Max-Sharpe", "Min-Volatility", "Risk Parity", "HRP"]


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the cumulative wealth curve (≤ 0)."""
    wealth = (1.0 + returns).cumprod()
    peaks = wealth.cummax()
    return float((wealth / peaks - 1.0).min())


def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> dict[str, float]:
    """Annualised performance metrics from a daily return series."""
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    if n == 0:
        raise ValueError("returns series is empty")
    total_growth = float((1.0 + r).prod())
    annual_return = total_growth ** (TRADING_DAYS / n) - 1.0
    annual_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0

    excess = r - risk_free_rate / TRADING_DAYS
    sharpe = (float(excess.mean()) * TRADING_DAYS / annual_vol) if annual_vol > 0 else 0.0
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = (float(excess.mean()) * TRADING_DAYS / downside) if downside > 0 else 0.0

    mdd = max_drawdown(r)
    calmar = annual_return / abs(mdd) if mdd < 0 else 0.0
    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    metrics: pd.DataFrame                      # strategy × metric
    growth: pd.DataFrame                       # cumulative growth of $10,000
    returns: pd.DataFrame                      # out-of-sample daily returns
    weights_history: dict = field(default_factory=dict)  # strategy → [weight vectors]


def _strategy_weights(name: str, mu: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    n = len(mu)
    if name == "Equal Weight":
        return equal_weight(n)
    if name == "Max-Sharpe":
        return max_sharpe_portfolio(mu, cov, risk_free_rate=0.02).weights
    if name == "Min-Volatility":
        return min_volatility_portfolio(mu, cov, risk_free_rate=0.02).weights
    if name == "Risk Parity":
        return risk_parity_portfolio(cov)
    if name == "HRP":
        return hrp_portfolio(cov)
    raise KeyError(name)  # pragma: no cover


def run_backtest(
    prices: pd.DataFrame | None = None,
    seed: int = 42,
    n_days: int = 756,
    train_size: int = 252,
    step: int = 63,
    risk_free_rate: float = 0.02,
    initial_value: float = 10_000.0,
) -> BacktestResult:
    """Expanding-window walk-forward backtest on synthetic seeded data."""
    if prices is None:
        prices = generate_synthetic_prices(n_days=n_days, seed=seed)
    rets = compute_returns(prices)
    if len(rets) <= train_size:
        raise ValueError("not enough history for the initial training window")

    segments: dict[str, list[pd.Series]] = {name: [] for name in STRATEGIES}
    weights_history: dict[str, list[np.ndarray]] = {name: [] for name in STRATEGIES}

    for start in range(train_size, len(rets), step):
        train = rets.iloc[:start]
        test = rets.iloc[start:start + step]
        mu = expected_returns(train)
        cov, _delta = ledoit_wolf_shrinkage(train)  # shrunk, annualised
        test_arr = test.to_numpy(dtype=float)
        for name in STRATEGIES:
            w = _strategy_weights(name, mu, cov)
            weights_history[name].append(w)
            segments[name].append(pd.Series(test_arr @ w, index=test.index))

    oos_returns = pd.DataFrame(
        {name: pd.concat(parts) for name, parts in segments.items()}
    )
    metrics = pd.DataFrame(
        {name: performance_metrics(oos_returns[name], risk_free_rate)
         for name in STRATEGIES}
    ).T
    growth = initial_value * (1.0 + oos_returns).cumprod()
    growth = pd.concat(
        [pd.DataFrame(initial_value, index=[oos_returns.index[0] - pd.tseries.frequencies.to_offset("1B")],
                      columns=growth.columns), growth]
    )
    return BacktestResult(metrics=metrics, growth=growth, returns=oos_returns,
                          weights_history=weights_history)


def format_metrics_table(metrics: pd.DataFrame) -> str:
    """Human-readable comparison table for the terminal / README."""
    df = metrics.copy()
    pct_cols = ["annual_return", "annual_volatility", "max_drawdown"]
    lines = [
        f"{'Strategy':<18} {'Return':>8} {'Vol':>8} {'Sharpe':>7} "
        f"{'Sortino':>8} {'MaxDD':>8} {'Calmar':>7}",
        "-" * 72,
    ]
    for name, row in df.iterrows():
        lines.append(
            f"{name:<18} {row['annual_return']:>7.2%} {row['annual_volatility']:>7.2%} "
            f"{row['sharpe']:>7.3f} {row['sortino']:>8.3f} "
            f"{row['max_drawdown']:>7.2%} {row['calmar']:>7.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # python src/backtest.py
    result = run_backtest()
    print("Walk-forward backtest — expanding window, Ledoit–Wolf covariance")
    print(f"Out-of-sample days: {len(result.returns)}  "
          f"Strategies: {', '.join(STRATEGIES)}\n")
    print(format_metrics_table(result.metrics))
