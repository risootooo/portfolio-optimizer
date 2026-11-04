"""Tests for the Portfolio Optimizer. All offline — synthetic data only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio_optimizer import (
    compute_returns,
    covariance_matrix,
    efficient_frontier,
    expected_returns,
    generate_random_portfolios,
    generate_synthetic_prices,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    monte_carlo_simulation,
    portfolio_stats,
    summarise_simulation,
)

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def prices():
    return generate_synthetic_prices(tickers=TICKERS, n_days=252, seed=7)


@pytest.fixture(scope="module")
def returns(prices):
    return compute_returns(prices)


@pytest.fixture(scope="module")
def mu_cov(returns):
    return expected_returns(returns), covariance_matrix(returns)


# 1 — returns calculation -------------------------------------------------
def test_compute_returns_values():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0]})
    rets = compute_returns(prices)
    assert len(rets) == 2
    assert rets.iloc[0]["A"] == pytest.approx(0.10)
    assert rets.iloc[1]["A"] == pytest.approx(-0.10)
    assert rets.iloc[0]["B"] == pytest.approx(0.0)
    assert not rets.isna().any().any()


def test_synthetic_prices_deterministic_and_positive(prices):
    again = generate_synthetic_prices(tickers=TICKERS, n_days=252, seed=7)
    pd.testing.assert_frame_equal(prices, again)
    assert (prices > 0).all().all()
    assert list(prices.columns) == TICKERS


# 2 — covariance shape / properties --------------------------------------
def test_covariance_shape_and_symmetry(returns, mu_cov):
    _, cov = mu_cov
    assert cov.shape == (len(TICKERS), len(TICKERS))
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)
    # Variances (diagonal) must be positive for stochastic synthetic data
    assert (np.diag(cov.to_numpy()) > 0).all()


def test_expected_returns_shape(returns):
    mu = expected_returns(returns)
    assert isinstance(mu, pd.Series)
    assert len(mu) == len(TICKERS)
    assert list(mu.index) == TICKERS


# 3 — portfolio statistics maths ------------------------------------------
def test_portfolio_stats_known_values():
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    stats = portfolio_stats([0.5, 0.5], mu, cov, risk_free_rate=0.0)
    assert stats.expected_return == pytest.approx(0.15)
    # variance = 0.25*0.04 + 0.25*0.09 = 0.0325
    assert stats.volatility == pytest.approx(np.sqrt(0.0325))
    assert stats.sharpe_ratio == pytest.approx(0.15 / np.sqrt(0.0325))


def test_portfolio_stats_single_asset():
    mu = np.array([0.12, 0.08])
    cov = np.array([[0.05, 0.01], [0.01, 0.03]])
    stats = portfolio_stats([1.0, 0.0], mu, cov, risk_free_rate=0.02)
    assert stats.expected_return == pytest.approx(0.12)
    assert stats.volatility == pytest.approx(np.sqrt(0.05))
    assert stats.sharpe_ratio == pytest.approx((0.12 - 0.02) / np.sqrt(0.05))


def test_portfolio_stats_rejects_wrong_length(mu_cov):
    mu, cov = mu_cov
    with pytest.raises(ValueError):
        portfolio_stats([0.5, 0.5], mu, cov)


# 4 — optimisers -----------------------------------------------------------
def test_max_sharpe_weights_sum_to_one_and_non_negative(mu_cov):
    mu, cov = mu_cov
    stats = max_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
    assert stats.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (stats.weights >= -1e-9).all()
    assert stats.volatility > 0
    # Max-Sharpe must beat (or match) an equal-weight portfolio
    eq = portfolio_stats(np.full(len(TICKERS), 1 / len(TICKERS)), mu, cov, 0.02)
    assert stats.sharpe_ratio >= eq.sharpe_ratio - 1e-6


def test_min_volatility_weights_and_volatility(mu_cov):
    mu, cov = mu_cov
    stats = min_volatility_portfolio(mu, cov, risk_free_rate=0.02)
    assert stats.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (stats.weights >= -1e-9).all()
    eq = portfolio_stats(np.full(len(TICKERS), 1 / len(TICKERS)), mu, cov, 0.02)
    assert stats.volatility <= eq.volatility + 1e-9


# 5 — random portfolios / frontier / Monte Carlo ---------------------------
def test_random_portfolios_weights_normalised(mu_cov):
    mu, cov = mu_cov
    df = generate_random_portfolios(mu, cov, n_portfolios=200, seed=1)
    weight_cols = [c for c in df.columns if c.startswith("w_")]
    assert len(weight_cols) == len(TICKERS)
    np.testing.assert_allclose(df[weight_cols].sum(axis=1), 1.0, atol=1e-9)
    assert (df[weight_cols] >= 0).all().all()
    assert (df["volatility"] > 0).all()


def test_efficient_frontier_is_sorted_by_return(mu_cov):
    mu, cov = mu_cov
    frontier = efficient_frontier(mu, cov, n_points=10, risk_free_rate=0.02)
    assert not frontier.empty
    assert frontier["return"].is_monotonic_increasing
    assert (frontier["volatility"] > 0).all()


def test_monte_carlo_simulation_shape_and_summary(mu_cov):
    mu, cov = mu_cov
    weights = np.array([0.5, 0.3, 0.2])
    wealth = monte_carlo_simulation(weights, mu, cov, n_simulations=100,
                                    n_days=50, initial_value=10_000.0, seed=3)
    assert wealth.shape == (51, 100)
    assert (wealth.iloc[0] == 10_000.0).all()
    summary = summarise_simulation(wealth, initial_value=10_000.0)
    assert summary["p5_final"] <= summary["median_final"] <= summary["p95_final"]
    assert 0.0 <= summary["prob_loss"] <= 1.0
