"""Shared Taiwan-market data-fetching layer: TWSE/TPEx candidate universe,
individual ticker price history, and a local disk cache under .cache/ so
repeated runs (CLI backtest, the portfolio tool, dev iteration) don't
re-hit TWSE/TPEx/Yahoo Finance every time.

Used by both tw_backtest_main.py (historical backtest) and
tw_portfolio_tool.py (the live portfolio tool).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from tw_universe import build_top_n_universe

CACHE_DIR = Path(__file__).parent.parent / ".cache"
MIN_HISTORY_ROWS = 1500  # ~6y of trading days; screens out recent IPOs


def fetch_yf_history(ticker: str, start: str, retries: int = 4) -> pd.Series | None:
    for attempt in range(retries):
        d = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if not d.empty:
            return d["Close"].iloc[:, 0]
        time.sleep(2 * (attempt + 1))
    return None


def cached_series(cache_name: str, ticker: str, fetch_start: str, refresh: bool) -> pd.Series:
    """Fetch-or-load a single ticker's close-price series, cached to
    .cache/{cache_name}.csv. Forward-filled: yfinance occasionally
    returns an isolated NaN for a single day even on liquid tickers,
    which silently breaks any `.asof()`-style lookup landing exactly on
    that date.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_name}.csv"
    if not refresh and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0].ffill()
    series = fetch_yf_history(ticker, fetch_start)
    if series is None:
        raise RuntimeError(f"Failed to fetch {ticker} after retries")
    series = series.ffill()
    series.to_csv(cache_file)
    return series


def build_candidate_prices(
    pool_size: int, fetch_start: str, refresh: bool, min_history_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (prices_df, pool_df) for the top-`pool_size` TWSE/TPEx
    candidates by market cap. Candidates lacking sufficient price history
    back to `fetch_start` are dropped and backfilled from the next-ranked
    names, so the pool is always `pool_size` liquid names that all have
    data over the full requested window. Does NOT include any benchmark
    tickers -- use `cached_series` for those separately.

    min_history_rows: if None, derived from `fetch_start` (require 90% of
    the calendar-implied trading days) -- this must scale with the
    requested window, not a fixed constant, or a short lookback (e.g. the
    live tool's ~2y) would wrongly reject every candidate including
    TSMC (this happened in testing: a constant calibrated for the ~6y
    backtest window rejected 100% of a 2y-window fetch).
    """
    if min_history_rows is None:
        calendar_days = (pd.Timestamp.today() - pd.Timestamp(fetch_start)).days
        min_history_rows = int(0.9 * calendar_days * (252 / 365.25))

    CACHE_DIR.mkdir(exist_ok=True)
    prices_cache = CACHE_DIR / f"tw_candidates_prices_{pool_size}_{fetch_start}.csv"
    pool_cache = CACHE_DIR / f"tw_candidates_pool_{pool_size}_{fetch_start}.csv"

    if not refresh and prices_cache.exists() and pool_cache.exists():
        prices = pd.read_csv(prices_cache, index_col=0, parse_dates=True)
        pool = pd.read_csv(pool_cache)
        return prices, pool

    print(f"  Fetching TWSE+TPEx universe and ranking top candidates (pool_size={pool_size})...")
    candidates = build_top_n_universe(n=pool_size + 15)  # headroom for history-gap dropouts

    frames: dict[str, pd.Series] = {}
    kept_rows = []
    for _, row in candidates.iterrows():
        if len(frames) >= pool_size:
            break
        series = fetch_yf_history(row["yf_ticker"], fetch_start)
        if series is not None and len(series) >= min_history_rows:
            frames[row["yf_ticker"]] = series
            kept_rows.append(row)
        else:
            print(f"  Skipping {row['yf_ticker']} ({row['name']}): insufficient history, using next-ranked candidate")

    if len(frames) < pool_size:
        raise RuntimeError(f"Only found {len(frames)} candidates with full history; widen the search (increase headroom).")

    prices = pd.DataFrame(frames).ffill()
    pool = pd.DataFrame(kept_rows).reset_index(drop=True)

    prices.to_csv(prices_cache)
    pool.to_csv(pool_cache, index=False)
    return prices, pool


def add_custom_ticker(prices: pd.DataFrame, pool: pd.DataFrame, yf_ticker: str, fetch_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and append a user-supplied ticker (not from the TWSE/TPEx
    ranked universe -- e.g. a stock they specifically want considered)
    to the candidate pool. Raises if the ticker has no data.
    """
    series = fetch_yf_history(yf_ticker, fetch_start)
    if series is None or series.empty:
        raise ValueError(f"No price data found for {yf_ticker}")
    prices = prices.copy()
    prices[yf_ticker] = series
    prices = prices.ffill()
    if yf_ticker not in pool["yf_ticker"].values:
        pool = pd.concat([pool, pd.DataFrame([{
            "code": yf_ticker.split(".")[0], "name": yf_ticker, "market": "custom",
            "yf_ticker": yf_ticker, "close": float(series.iloc[-1]),
            "market_cap": float("nan"), "turnover_value": float("nan"),
        }])], ignore_index=True)
    return prices, pool
