# Portfolio Optimizer

A Modern Portfolio Theory (MPT) toolkit in Python: estimate expected returns and risk from price history, explore thousands of random portfolios, compute the efficient frontier, and find the **maximum-Sharpe** and **minimum-volatility** portfolios — plus Monte Carlo simulation of future wealth paths.

It runs **fully offline** out of the box. A seeded synthetic-data generator produces realistic, correlated asset prices, so the demo and the entire test suite work with no internet connection and no API keys. An optional `yfinance` wrapper is included for live data when you want it.

> **Disclaimer: Educational use only — not investment advice.** Past / simulated performance does not predict future returns. Do your own research and consider speaking to a licensed financial adviser before investing.

---

## What it does

| Capability | Function |
|---|---|
| Synthetic prices (offline, seeded) | `generate_synthetic_prices()` |
| Live prices (optional, needs `yfinance`) | `download_prices()` |
| Daily returns | `compute_returns()` |
| Expected returns (annualised) | `expected_returns()` |
| Covariance matrix (annualised) | `covariance_matrix()` |
| Portfolio return / vol / Sharpe | `portfolio_stats()` |
| Random portfolio sampling | `generate_random_portfolios()` |
| Efficient frontier | `efficient_frontier()` |
| Maximum-Sharpe portfolio | `max_sharpe_portfolio()` |
| Minimum-volatility portfolio | `min_volatility_portfolio()` |
| Monte Carlo wealth paths | `monte_carlo_simulation()` / `summarise_simulation()` |

## The maths

For weights $w$, expected asset returns $\mu$ and covariance matrix $\Sigma$ (all annualised, 252 trading days):

- **Portfolio return:** $R_p = w^T \mu$
- **Portfolio variance:** $\sigma_p^2 = w^T \Sigma w$, volatility $\sigma_p = \sqrt{\sigma_p^2}$
- **Sharpe ratio:** $S = \dfrac{R_p - R_f}{\sigma_p}$ — excess return per unit of risk, with risk-free rate $R_f$

**Efficient frontier:** for each target return, the frontier portfolio is the one with the lowest possible volatility. Portfolios on the frontier are *efficient* — no other portfolio offers more return for the same risk, or less risk for the same return.

**Optimisation:** long-only ($0 \le w_i \le 1$), fully invested ($\sum w_i = 1$). Solved with SciPy SLSQP when SciPy is available; otherwise a NumPy random-search + local-refinement solver is used automatically, so the toolkit still runs in minimal environments.

**Monte Carlo:** daily asset returns are drawn from a multivariate normal distribution with the estimated daily $\mu$ / $\Sigma$ (projected to the nearest positive-semidefinite matrix), compounded into wealth paths. The summary reports median, 5th and 95th percentile final values and the probability of loss.

## Installation

```bash
git clone <this-repo>
cd portfolio-optimizer
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt        # numpy, pandas, scipy
pip install -r requirements-dev.txt    # + pytest, for the tests
```

Requires Python 3.10+.

## Quick start — CLI demo (offline)

```bash
python main.py
```

Sample output (verified run, synthetic data, seed 42 — reproducible exactly):

```
Portfolio Optimizer — Modern Portfolio Theory demo
Data: synthetic (seeded, offline) — no internet required
Assets: AAPL, MSFT, GOOGL, AMZN, JPM, XOM
History: 505 trading days, 504 return observations

Random portfolios sampled: 5,000 (vol range 13.58%–25.08%)

Maximum-Sharpe portfolio
------------------------
  AMZN   66.64%
  JPM    33.36%
  Expected annual return : 20.49%
  Annual volatility      : 17.99%
  Sharpe ratio (rf=2%)   :  1.028

Minimum-volatility portfolio
----------------------------
  AAPL    0.18%
  GOOGL  57.06%
  AMZN   12.39%
  JPM    26.80%
  XOM     3.58%
  Expected annual return :  9.94%
  Annual volatility      : 13.37%
  Sharpe ratio (rf=2%)   :  0.594

Monte Carlo — max-Sharpe portfolio, 1 year, $10,000 invested
--------------------------------------------------------------
  Simulations            : 1,000
  Median final value     : $12,097.18
  5th percentile         : $9,071.42
  95th percentile        : $16,068.49
  Probability of loss    : 15.50%
```

(The demo also prints a 10-point efficient-frontier table between the
minimum-volatility portfolio and the Monte Carlo section.)

Options: `python main.py --seed 7 --portfolios 10000 --simulations 2000`

## Usage as a library

```python
import sys
sys.path.insert(0, "src")

from portfolio_optimizer import (
    generate_synthetic_prices, compute_returns, expected_returns,
    covariance_matrix, max_sharpe_portfolio, min_volatility_portfolio,
)

prices  = generate_synthetic_prices(n_days=504, seed=42)   # or your own price DataFrame
returns = compute_returns(prices)
mu      = expected_returns(returns)        # annualised
cov     = covariance_matrix(returns)       # annualised

best = max_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
print(best.weights, best.expected_return, best.volatility, best.sharpe_ratio)

calm = min_volatility_portfolio(mu, cov, risk_free_rate=0.02)
```

To use real market data, pass any price DataFrame (columns = tickers, rows = dates) to `compute_returns()` — or install the optional extra and call `download_prices(["AAPL", "MSFT"], start="2023-01-01")` (`pip install yfinance`).

## Tests

```bash
python -m pytest tests/ -v
```

12 tests cover: return calculation against hand-computed values, synthetic-data determinism, covariance shape / symmetry / positive variances, portfolio-statistics maths against closed-form values, optimiser constraints (weights sum to 1, non-negative) and optimality vs. an equal-weight benchmark, random-portfolio normalisation, frontier monotonicity, and Monte Carlo shape / percentile ordering. All tests run offline.

## Project structure

```
portfolio-optimizer/
├── main.py                        # CLI demo (offline, synthetic data)
├── requirements.txt               # numpy, pandas, scipy
├── requirements-dev.txt           # + pytest
├── src/
│   ├── __init__.py                # public API re-exports
│   └── portfolio_optimizer.py     # core module (data, stats, optimisation, MC)
├── tests/
│   └── test_portfolio_optimizer.py
├── examples/
│   └── quickstart.py
├── LICENSE
└── README.md
```

## Limitations

- Mean-variance optimisation is sensitive to estimation error in $\mu$ and $\Sigma$; the "optimal" portfolio is optimal *for the estimates*, not for the future.
- Normal-return Monte Carlo understates tail risk (real returns are fat-tailed).
- No transaction costs, taxes, slippage, shorting, or leverage are modelled.
- Synthetic data is for development and education — it is not market data.

## License

MIT — see [LICENSE](LICENSE).

---

*Educational use only — not investment advice.*
