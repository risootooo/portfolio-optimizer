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
]
__version__ = "1.0.0"
