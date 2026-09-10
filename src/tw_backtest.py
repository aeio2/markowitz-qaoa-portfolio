"""Out-of-sample, periodically-rebalanced backtest of the cardinality-
constrained QUBO stock selection on a Taiwan large-cap universe, compared
against 0050.TW (Taiwan 50 ETF) and the S&P 500 as an opportunity-cost
benchmark.

Design (see README for the full write-up and its limitations):
  - Candidate pool: a FIXED top-20 Taiwan large caps by market cap
    (src/tw_universe.py), screened for liquidity and full price history
    back to the backtest start. Fixed rather than re-derived at each
    rebalance date -- a documented simplification, not point-in-time
    liquidity ranking (see README "Known limitations").
  - At each rebalance date, mu/Sigma are estimated ONLY from a trailing
    lookback window ending at that date (no look-ahead on the return
    data itself) and fed into the same cardinality-constrained QUBO used
    in qubo_portfolio.py (choose 10 of 20, equal-weighted).
  - Selection uses the EXACT solver (NumPyMinimumEigensolver) at every
    rebalance date. At 20 qubits this is ~0.25s; QAOA at this size took
    over 10 minutes for a single solve in testing (vs ~200s at 10
    qubits) -- reusing it at every rebalance date across a multi-year
    backtest is not tractable on a laptop simulator. QAOA is instead run
    once, separately, as a same-Hamiltonian quality comparison (see
    tw_qaoa_comparison.py), not to drive a second backtest curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qubo_portfolio import build_cardinality_qubo
from quantum_solver import solve_exact


@dataclass
class RebalancePeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    selected: list[str]
    period_return: float
    objective: float


def rebalance_dates(start: str, end: str, freq: str | None = "YS") -> list[pd.Timestamp]:
    """Period boundaries. freq='YS' = annual, 'QS' = quarterly, 'MS' =
    monthly. freq=None = a single buy-and-hold period, no rebalancing at
    all -- just [start, end]. (Any freq coarser than the start/end window
    degrades to this automatically anyway, e.g. quarterly freq over a
    2-week window; None makes that explicit rather than incidental.)
    """
    if freq is None:
        return [pd.Timestamp(start), pd.Timestamp(end)]
    dates = pd.date_range(start=start, end=end, freq=freq).tolist()
    end_ts = pd.Timestamp(end)
    if not dates or dates[0] > pd.Timestamp(start):
        dates = [pd.Timestamp(start)] + dates
    if dates[-1] < end_ts:
        dates.append(end_ts)
    return dates


def _price_asof(prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Last available close on/before `date` for every column."""
    idx = prices.index[prices.index <= date]
    if len(idx) == 0:
        raise ValueError(f"No price data on or before {date}")
    return prices.loc[idx[-1]]


def periodic_win_loss(
    curve_a: pd.Series, curve_b: pd.Series, label_a: str, label_b: str, freq: str = "W-MON",
) -> pd.DataFrame:
    """Chop [curve_a, curve_b]'s common date range into periods (default
    weekly, Monday-anchored) and compare EACH period's return between the
    two curves -- which periods A actually beat B in, not just the
    single aggregate number over the whole window. A single blended
    average can look like "A wins" while hiding that the edge was
    concentrated in one period and reversed everywhere else (or the
    opposite) -- that's the point of this function.

    Both curves must already be portfolio value series (e.g. an
    equal-weighted index like tw_portfolio_tool.py's `portfolio_curve`,
    or equity_curve()'s output), indexed by date. Neither curve is
    re-optimized within a period -- this compares two FIXED baskets'
    behavior period by period, not a re-selected one.
    """
    start = max(curve_a.index.min(), curve_b.index.min())
    end = min(curve_a.index.max(), curve_b.index.max())
    if start >= end:
        return pd.DataFrame(columns=["period_start", "period_end", label_a, label_b, "winner", "diff"])

    boundaries = rebalance_dates(start.date().isoformat(), end.date().isoformat(), freq)
    frame_a, frame_b = curve_a.to_frame("v"), curve_b.to_frame("v")

    rows = []
    for i in range(len(boundaries) - 1):
        p0, p1 = boundaries[i], boundaries[i + 1]
        ret_a = float(_price_asof(frame_a, p1)["v"] / _price_asof(frame_a, p0)["v"] - 1)
        ret_b = float(_price_asof(frame_b, p1)["v"] / _price_asof(frame_b, p0)["v"] - 1)
        winner = label_a if ret_a > ret_b else (label_b if ret_b > ret_a else "Tie")
        rows.append({
            "period_start": p0, "period_end": p1,
            label_a: ret_a, label_b: ret_b, "winner": winner, "diff": ret_a - ret_b,
        })
    return pd.DataFrame(rows)


def run_backtest(
    prices: pd.DataFrame,
    lookback_years: float,
    rebalance_freq: str,
    budget: int,
    risk_factor: float,
    start: str,
    end: str,
) -> list[RebalancePeriod]:
    """Roll forward through rebalance dates; at each one, select `budget`
    of `prices.columns` using only data in the trailing lookback window,
    then hold equal-weighted until the next rebalance date.
    """
    dates = rebalance_dates(start, end, rebalance_freq)
    periods: list[RebalancePeriod] = []

    for i in range(len(dates) - 1):
        reb_date, next_date = dates[i], dates[i + 1]
        lookback_start = reb_date - pd.Timedelta(days=int(lookback_years * 365.25))
        window = prices.loc[(prices.index > lookback_start) & (prices.index <= reb_date)]
        if len(window) < 60:
            raise ValueError(f"Insufficient lookback data before {reb_date.date()} ({len(window)} rows)")

        log_ret = np.log(window / window.shift(1)).dropna()
        mu = (log_ret.mean() * 252).values
        sigma = (log_ret.cov() * 252).values

        qp, _ = build_cardinality_qubo(mu, sigma, budget=budget, risk_factor=risk_factor)
        run = solve_exact(qp)
        selected = [t for t, bit in zip(prices.columns, run.result.x) if round(bit) == 1]

        start_prices = _price_asof(prices[selected], reb_date)
        end_prices = _price_asof(prices[selected], next_date)
        period_return = float((end_prices / start_prices - 1).mean())  # equal-weighted

        periods.append(RebalancePeriod(reb_date, next_date, selected, period_return, run.result.fval))

    return periods


def run_equal_weight_baseline(
    prices: pd.DataFrame, rebalance_freq: str, start: str, end: str,
) -> list[RebalancePeriod]:
    """Control strategy: equal-weight ALL candidates (no QUBO selection),
    rebalanced on the same date grid. Isolates how much of the QUBO
    strategy's return comes from the candidate pool itself (which is
    built from today's market caps, a look-ahead simplification -- see
    README) versus the mean-variance selection step on top of it. If
    this control performs similarly to the QUBO-selected portfolio, the
    pool construction -- not the optimization -- explains the result.
    """
    dates = rebalance_dates(start, end, rebalance_freq)
    all_tickers = list(prices.columns)
    periods: list[RebalancePeriod] = []
    for i in range(len(dates) - 1):
        reb_date, next_date = dates[i], dates[i + 1]
        start_prices = _price_asof(prices, reb_date)
        end_prices = _price_asof(prices, next_date)
        period_return = float((end_prices / start_prices - 1).mean())
        periods.append(RebalancePeriod(reb_date, next_date, all_tickers, period_return, float("nan")))
    return periods


def equity_curve(periods: list[RebalancePeriod], base: float = 1.0) -> pd.Series:
    dates = [periods[0].start] + [p.end for p in periods]
    values = [base]
    for p in periods:
        values.append(values[-1] * (1 + p.period_return))
    return pd.Series(values, index=dates)


def benchmark_curve(prices: pd.Series, dates: list[pd.Timestamp], base: float = 1.0) -> pd.Series:
    """Buy-and-hold equity curve for a single price series, indexed to
    the same rebalance-date grid so it plots directly against
    `equity_curve`. Unhedged: for a foreign-currency asset (e.g. the
    S&P 500 in USD) this is the *local-currency* index return, not what
    a TWD-based investor actually realized -- see
    `fx_adjusted_benchmark_curve` for that.
    """
    p0 = _price_asof(prices.to_frame("p"), dates[0])["p"]
    vals = [base * (_price_asof(prices.to_frame("p"), d)["p"] / p0) for d in dates]
    return pd.Series(vals, index=dates)


def fx_adjusted_benchmark_curve(
    prices: pd.Series, fx: pd.Series, dates: list[pd.Timestamp], base: float = 1.0,
) -> pd.Series:
    """Buy-and-hold equity curve for a foreign-currency asset, converted
    into home-currency (TWD) terms -- the actual return a TWD-based
    investor would have realized, not just the local-currency index
    return. `fx` must be home-currency-per-foreign-currency (e.g.
    TWD=X / USDTWD: TWD per 1 USD) on the same convention as
    yfinance's 'TWD=X' ticker.

    value_TWD(t) = base * (price(t)/price(t0)) * (fx(t)/fx(t0))

    i.e. the foreign-currency return compounded with the currency's own
    move over the same window. This is the right number for "opportunity
    cost of investing abroad" framing -- the unhedged local-currency
    number in `benchmark_curve` is not.
    """
    p0 = _price_asof(prices.to_frame("p"), dates[0])["p"]
    fx0 = _price_asof(fx.to_frame("fx"), dates[0])["fx"]
    vals = []
    for d in dates:
        pt = _price_asof(prices.to_frame("p"), d)["p"]
        fxt = _price_asof(fx.to_frame("fx"), d)["fx"]
        vals.append(base * (pt / p0) * (fxt / fx0))
    return pd.Series(vals, index=dates)


def summarize(periods: list[RebalancePeriod]) -> str:
    lines = []
    for p in periods:
        lines.append(
            f"{p.start.date()} -> {p.end.date()}: {', '.join(t.replace('.TW','') for t in p.selected)}  "
            f"period_return={p.period_return:+.2%}  objective={p.objective:.4f}"
        )
    return "\n".join(lines)
