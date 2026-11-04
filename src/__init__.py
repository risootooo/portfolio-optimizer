"""Portfolio Optimizer — Modern Portfolio Theory toolkit."""

from .portfolio_optimizer import (
    PortfolioStats,
    compute_returns,
    covariance_matrix,
    download_prices,
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
from .covariance import ledoit_wolf_shrinkage
from .allocators import (
    black_litterman_portfolio,
    black_litterman_posterior,
    equal_weight,
    hrp_portfolio,
    risk_contribution_shares,
    risk_contributions,
    risk_parity_portfolio,
)
from .backtest import (
    STRATEGIES,
    BacktestResult,
    format_metrics_table,
    max_drawdown,
    performance_metrics,
    run_backtest,
)

__all__ = [
    "PortfolioStats",
    "compute_returns",
    "covariance_matrix",
    "download_prices",
    "efficient_frontier",
    "expected_returns",
    "generate_random_portfolios",
    "generate_synthetic_prices",
    "max_sharpe_portfolio",
    "min_volatility_portfolio",
    "monte_carlo_simulation",
    "portfolio_stats",
    "summarise_simulation",
    "ledoit_wolf_shrinkage",
    "black_litterman_portfolio",
    "black_litterman_posterior",
    "equal_weight",
    "hrp_portfolio",
    "risk_contribution_shares",
    "risk_contributions",
    "risk_parity_portfolio",
    "STRATEGIES",
    "BacktestResult",
    "format_metrics_table",
    "max_drawdown",
    "performance_metrics",
    "run_backtest",
]
__version__ = "2.0.0"
