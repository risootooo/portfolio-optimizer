#!/usr/bin/env python3
"""CLI demo — runs end-to-end OFFLINE on synthetic data.

Usage:
    python main.py [--seed 42] [--portfolios 5000] [--simulations 1000]

Prints the max-Sharpe and min-volatility portfolios (weights, expected
return, volatility, Sharpe), an efficient-frontier sample, and a Monte
Carlo summary. No internet connection is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

from portfolio_optimizer import (  # noqa: E402
    compute_returns,
    covariance_matrix,
    efficient_frontier,
    expected_returns,
    generate_random_portfolios,
    generate_synthetic_prices,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    monte_carlo_simulation,
    summarise_simulation,
)

RISK_FREE_RATE = 0.02


def _print_portfolio(title: str, tickers: list[str], stats) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for ticker, w in zip(tickers, stats.weights):
        if w > 0.0005:
            print(f"  {ticker:<6} {w:6.2%}")
    print(f"  Expected annual return : {stats.expected_return:6.2%}")
    print(f"  Annual volatility      : {stats.volatility:6.2%}")
    print(f"  Sharpe ratio (rf=2%)   : {stats.sharpe_ratio:6.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio Optimizer demo (offline, synthetic data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--portfolios", type=int, default=5000)
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM"]
    print("Portfolio Optimizer — Modern Portfolio Theory demo")
    print("Data: synthetic (seeded, offline) — no internet required")
    print(f"Assets: {', '.join(tickers)}")

    prices = generate_synthetic_prices(tickers=tickers, n_days=504, seed=args.seed)
    returns = compute_returns(prices)
    mu = expected_returns(returns)
    cov = covariance_matrix(returns)
    print(f"History: {len(prices)} trading days, {len(returns)} return observations")

    random_ports = generate_random_portfolios(mu, cov, n_portfolios=args.portfolios,
                                              risk_free_rate=RISK_FREE_RATE, seed=args.seed)
    print(f"\nRandom portfolios sampled: {len(random_ports):,} "
          f"(vol range {random_ports['volatility'].min():.2%}–{random_ports['volatility'].max():.2%})")

    best = max_sharpe_portfolio(mu, cov, risk_free_rate=RISK_FREE_RATE)
    _print_portfolio("Maximum-Sharpe portfolio", tickers, best)

    min_vol = min_volatility_portfolio(mu, cov, risk_free_rate=RISK_FREE_RATE)
    _print_portfolio("Minimum-volatility portfolio", tickers, min_vol)

    frontier = efficient_frontier(mu, cov, n_points=10, risk_free_rate=RISK_FREE_RATE)
    if not frontier.empty:
        print("\nEfficient frontier (sample)")
        print("-----------------------------")
        print(frontier[["return", "volatility", "sharpe"]].round(4).to_string(index=False))

    wealth = monte_carlo_simulation(best.weights, mu, cov,
                                    n_simulations=args.simulations, n_days=252,
                                    initial_value=10_000.0, seed=args.seed)
    summary = summarise_simulation(wealth, initial_value=10_000.0)
    print("\nMonte Carlo — max-Sharpe portfolio, 1 year, $10,000 invested")
    print("--------------------------------------------------------------")
    print(f"  Simulations            : {args.simulations:,}")
    print(f"  Median final value     : ${summary['median_final']:,.2f}")
    print(f"  5th percentile         : ${summary['p5_final']:,.2f}")
    print(f"  95th percentile        : ${summary['p95_final']:,.2f}")
    print(f"  Probability of loss    : {summary['prob_loss']:6.2%}")
    print("\nDisclaimer: Educational use only — not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
