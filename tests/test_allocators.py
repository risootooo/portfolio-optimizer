"""Tests for the advanced allocators, covariance shrinkage and backtest."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio_optimizer import (
    compute_returns,
    covariance_matrix,
    expected_returns,
    generate_synthetic_prices,
    portfolio_stats,
)
from covariance import ledoit_wolf_shrinkage
from allocators import (
    black_litterman_portfolio,
    black_litterman_posterior,
    equal_weight,
    hrp_portfolio,
    risk_contribution_shares,
    risk_parity_portfolio,
)
from backtest import (
    STRATEGIES,
    format_metrics_table,
    max_drawdown,
    performance_metrics,
    run_backtest,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(scope="module")
def returns():
    prices = generate_synthetic_prices(tickers=TICKERS, n_days=300, seed=11)
    return compute_returns(prices)


@pytest.fixture(scope="module")
def cov(returns):
    return covariance_matrix(returns)


# --- Ledoit–Wolf shrinkage ----------------------------------------------
def test_ledoit_wolf_delta_in_unit_interval(returns):
    _, delta = ledoit_wolf_shrinkage(returns)
    assert 0.0 <= delta <= 1.0


def test_ledoit_wolf_symmetric_psd_and_trace_preserved(returns, cov):
    shrunk, _ = ledoit_wolf_shrinkage(returns)
    arr = shrunk.to_numpy()
    np.testing.assert_allclose(arr, arr.T, atol=1e-12)
    assert np.linalg.eigvalsh(arr).min() > 0  # positive definite
    # Target has the same trace as the Ledoit–Wolf sample matrix (1/T
    # normalisation — pandas .cov() uses 1/(T-1), hence the manual S here)
    X = returns.to_numpy()
    X = X - X.mean(axis=0, keepdims=True)
    S_lw = (X.T @ X) / len(X) * 252
    assert np.trace(arr) == pytest.approx(np.trace(S_lw), rel=1e-8)
    assert list(shrunk.columns) == TICKERS


def test_ledoit_wolf_improves_conditioning_on_noisy_data():
    rng = np.random.default_rng(0)
    # Few observations, many assets -> sample covariance is ill-conditioned
    noisy = pd.DataFrame(rng.normal(0, 0.02, size=(40, 12)),
                         columns=[f"A{i}" for i in range(12)])
    sample = covariance_matrix(noisy).to_numpy()
    shrunk, delta = ledoit_wolf_shrinkage(noisy)
    assert delta > 0
    assert np.linalg.cond(shrunk.to_numpy()) <= np.linalg.cond(sample)


# --- Risk Parity -----------------------------------------------------------
def test_risk_parity_equal_risk_contributions(cov):
    w = risk_parity_portfolio(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= 0).all()
    shares = risk_contribution_shares(w, cov)
    np.testing.assert_allclose(shares, np.full(len(TICKERS), 0.25), atol=0.02)


def test_risk_parity_custom_budgets(cov):
    budgets = np.array([0.4, 0.3, 0.2, 0.1])
    w = risk_parity_portfolio(cov, risk_budgets=budgets)
    shares = risk_contribution_shares(w, cov)
    np.testing.assert_allclose(shares, budgets, atol=0.02)


def test_equal_weight():
    np.testing.assert_allclose(equal_weight(4), np.full(4, 0.25))


# --- HRP -------------------------------------------------------------------
def test_hrp_weights_valid(cov):
    w = hrp_portfolio(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= 0).all()
    assert len(w) == len(TICKERS)


def test_hrp_favours_low_volatility_asset():
    # Two uncorrelated assets: one calm, one wild. HRP must favour the calm one.
    cov2 = np.array([[0.01, 0.0], [0.0, 0.16]])
    w = hrp_portfolio(cov2)
    assert w[0] > w[1]
    # Inverse-variance split for a single pair: w ∝ (1/0.01, 1/0.16)
    np.testing.assert_allclose(w, [16 / 17, 1 / 17], atol=0.02)


def test_hrp_deterministic(cov):
    np.testing.assert_allclose(hrp_portfolio(cov), hrp_portfolio(cov))


# --- Black–Litterman ---------------------------------------------------------
def test_black_litterman_no_views_returns_equilibrium(cov):
    w_mkt = equal_weight(len(TICKERS))
    post_mu, post_cov, pi = black_litterman_posterior(cov, w_mkt)
    np.testing.assert_allclose(post_mu, pi)
    np.testing.assert_allclose(post_cov, cov.to_numpy())


def test_black_litterman_view_pulls_posterior(cov):
    w_mkt = equal_weight(len(TICKERS))
    P = np.zeros((1, len(TICKERS)))
    P[0, 0] = 1.0
    Q = np.array([0.30])  # bullish absolute view on asset 0
    post_mu, _, pi = black_litterman_posterior(cov, w_mkt, P=P, Q=Q)
    assert pi[0] < post_mu[0] < 0.30  # moves toward the view, not all the way


def test_black_litterman_portfolio_valid(cov):
    w_mkt = equal_weight(len(TICKERS))
    P = np.zeros((1, len(TICKERS)))
    P[0, 1] = 1.0
    stats = black_litterman_portfolio(cov, w_mkt, P=P, Q=np.array([0.18]))
    assert stats.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (stats.weights >= -1e-9).all()


# --- Backtest ----------------------------------------------------------------
def test_max_drawdown_known_path():
    # Wealth: 100 -> 110 -> 99, i.e. -10% from the peak
    rets = pd.Series([0.10, -0.10])
    assert max_drawdown(rets) == pytest.approx(-0.10)


def test_performance_metrics_constant_growth():
    rets = pd.Series(np.full(252, 0.001))
    m = performance_metrics(rets, risk_free_rate=0.0)
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert m["annual_return"] == pytest.approx(1.001**252 - 1.0, rel=1e-6)
    assert m["annual_volatility"] == pytest.approx(0.0, abs=1e-12)


@pytest.fixture(scope="module")
def backtest_result():
    prices = generate_synthetic_prices(n_days=378, seed=5)
    return run_backtest(prices=prices, train_size=252, step=63)


def test_backtest_covers_all_strategies(backtest_result):
    assert list(backtest_result.metrics.index) == STRATEGIES
    for col in ("annual_return", "annual_volatility", "sharpe",
                "sortino", "max_drawdown", "calmar"):
        assert col in backtest_result.metrics.columns
    assert np.isfinite(backtest_result.metrics.to_numpy()).all()


def test_backtest_growth_starts_at_initial_value(backtest_result):
    growth = backtest_result.growth
    assert (growth.iloc[0] == 10_000.0).all()
    assert (growth > 0).all().all()
    assert len(backtest_result.returns) == 126  # 378 - 252 out-of-sample days


def test_backtest_table_formats(backtest_result):
    table = format_metrics_table(backtest_result.metrics)
    for name in STRATEGIES:
        assert name in table
