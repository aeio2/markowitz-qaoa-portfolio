"""Market data fetching and return/covariance estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def fetch_prices(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers.

    Returns a DataFrame indexed by date, one column per ticker, forward/back
    filled for any sparse trading holidays across exchanges.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers)
    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=0, how="any")
    if prices.empty:
        raise ValueError("No overlapping price history for the requested tickers.")
    return prices


def annualized_moments(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Compute annualized expected returns and covariance from daily prices.

    Expected returns use the mean of daily log returns, annualized by
    trading-day count; covariance is the sample covariance of daily log
    returns, annualized the same way. This is the standard Markowitz
    input estimation approach (historical-mean estimator).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean() * TRADING_DAYS
    sigma = log_returns.cov() * TRADING_DAYS
    return mu, sigma
