"""Build a Taiwan-listed stock universe (TWSE 上市 + TPEx 上櫃) and rank it
down to a liquidity/market-cap top-N candidate pool small enough to feed
into the cardinality-constrained QUBO/QAOA selection in qubo_portfolio.py.

Why a pre-filter is necessary: the QUBO formulation used here assigns one
binary (qubit) per candidate asset. TWSE+TPEx together list ~2000
securities -- solving "choose 10 of ~2000" would need ~2000 qubits, far
beyond any simulator (or current real quantum hardware). Narrowing to a
liquid, large-cap top-N (~20) first keeps the combinatorial selection
step tractable while still starting from the full market.

Data source: TWSE and TPEx official OpenAPI (public, no auth). Company
lists give shares outstanding (for market cap); daily quote endpoints
give close price and turnover value (for liquidity).
"""
from __future__ import annotations

import time

import pandas as pd
import requests

TWSE_COMPANY_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TWSE_QUOTE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_COMPANY_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
TPEX_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"


def _get_json(url: str, retries: int = 3):
    """These public endpoints occasionally drop large responses mid-stream;
    a plain retry clears it almost always."""
    last_err = None
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=30).json()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def _is_ordinary_code(code: str) -> bool:
    """Ordinary common-stock codes on TWSE/TPEx are 4 plain digits.
    ETFs (00xxxx), warrants, bonds, and other instrument codes have
    different patterns and are excluded by this check.
    """
    return len(code) == 4 and code.isdigit()


def _is_innovation_board(name: str) -> bool:
    """TWSE/TPEx Innovation Board ('創新板'/'戰略新板') listings carry a
    '-創' suffix on their abbreviated name. These are pre-revenue/thinly
    -floated companies whose (shares outstanding x price) market cap can
    be wildly disconnected from actual tradable liquidity -- e.g. one such
    name outranked TSMC's turnover-to-market-cap ratio by 4x on the low
    side despite a nominal NT$1.4T market cap. Excluded from a "liquid
    large cap" universe for that reason.
    """
    return "-創" in name


def fetch_twse_universe() -> pd.DataFrame:
    """TWSE (上市) ordinary stocks with market cap and turnover."""
    companies = _get_json(TWSE_COMPANY_URL)
    quotes = _get_json(TWSE_QUOTE_URL)

    comp_df = pd.DataFrame(companies)[["公司代號", "公司簡稱", "已發行普通股數或TDR原股發行股數"]]
    comp_df.columns = ["code", "name", "shares_outstanding"]
    comp_df["shares_outstanding"] = pd.to_numeric(comp_df["shares_outstanding"], errors="coerce")

    quote_df = pd.DataFrame(quotes)[["Code", "ClosingPrice", "TradeValue"]]
    quote_df.columns = ["code", "close", "turnover_value"]
    quote_df["close"] = pd.to_numeric(quote_df["close"], errors="coerce")
    quote_df["turnover_value"] = pd.to_numeric(quote_df["turnover_value"], errors="coerce")

    df = comp_df.merge(quote_df, on="code", how="inner")
    df = df[df["code"].apply(_is_ordinary_code)]
    df = df[~df["name"].apply(_is_innovation_board)]
    df["market_cap"] = df["shares_outstanding"] * df["close"]
    df["yf_ticker"] = df["code"] + ".TW"
    df["market"] = "TWSE"
    return df.dropna(subset=["close", "market_cap"])


def fetch_tpex_universe() -> pd.DataFrame:
    """TPEx (上櫃) ordinary stocks with market cap and turnover."""
    companies = _get_json(TPEX_COMPANY_URL)
    quotes = _get_json(TPEX_QUOTE_URL)

    comp_df = pd.DataFrame(companies)[["SecuritiesCompanyCode", "CompanyAbbreviation", "IssueShares"]]
    comp_df.columns = ["code", "name", "shares_outstanding"]
    comp_df["shares_outstanding"] = pd.to_numeric(comp_df["shares_outstanding"], errors="coerce")

    quote_df = pd.DataFrame(quotes)[["SecuritiesCompanyCode", "Close", "TransactionAmount"]]
    quote_df.columns = ["code", "close", "turnover_value"]
    quote_df["close"] = pd.to_numeric(quote_df["close"], errors="coerce")
    quote_df["turnover_value"] = pd.to_numeric(quote_df["turnover_value"], errors="coerce")

    df = comp_df.merge(quote_df, on="code", how="inner")
    df = df[df["code"].apply(_is_ordinary_code)]
    df = df[~df["name"].apply(_is_innovation_board)]
    df["market_cap"] = df["shares_outstanding"] * df["close"]
    df["yf_ticker"] = df["code"] + ".TWO"
    df["market"] = "TPEx"
    return df.dropna(subset=["close", "market_cap"])


def build_top_n_universe(n: int = 20, min_turnover_value: float = 10_000_000) -> pd.DataFrame:
    """Combine TWSE+TPEx, screen out illiquid names (turnover below
    min_turnover_value on the reference day, default NT$10M), then rank
    by market cap and take the top N -- a liquidity-gated, market-cap-
    ranked candidate pool small enough for QUBO/QAOA.
    """
    twse = fetch_twse_universe()
    tpex = fetch_tpex_universe()
    universe = pd.concat([twse, tpex], ignore_index=True)
    liquid = universe[universe["turnover_value"] >= min_turnover_value]
    top_n = liquid.sort_values("market_cap", ascending=False).head(n).reset_index(drop=True)
    return top_n[["code", "name", "market", "yf_ticker", "close", "market_cap", "turnover_value"]]
