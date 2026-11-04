#!/usr/bin/env python3
"""Minimal library-usage example (offline). Run: python examples/quickstart.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio_optimizer import (
    compute_returns,
    covariance_matrix,
    expected_returns,
    generate_synthetic_prices,
    max_sharpe_portfolio,
)

prices = generate_synthetic_prices(n_days=504, seed=42)
returns = compute_returns(prices)
mu = expected_returns(returns)
cov = covariance_matrix(returns)

best = max_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
for ticker, w in zip(prices.columns, best.weights):
    print(f"{ticker:<6} {w:6.2%}")
print(f"Return {best.expected_return:.2%} | Vol {best.volatility:.2%} | Sharpe {best.sharpe_ratio:.3f}")
