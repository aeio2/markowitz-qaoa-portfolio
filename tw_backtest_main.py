"""Run the Taiwan large-cap QUBO stock-selection backtest end-to-end:

  1. Build a top-N TWSE/TPEx candidate pool by market cap (liquidity-screened),
     dropping any candidate without enough price history for the backtest
     window and pulling in the next-ranked name instead.
  2. Roll forward through rebalance dates; at each one, select `--budget` of
     the pool using only trailing (out-of-sample) return data, via the exact
     QUBO/Ising solver.
  3. Chain realized period returns into an equity curve, with an equal-
     weight-all-candidates control to isolate the selection step's effect
     from the candidate pool's own construction.
  4. Compare against 0050.TW and ^GSPC (S&P 500) buy-and-hold over the same
     window.

Live TWSE/TPEx + yfinance data is cached under .cache/ (gitignored) so
repeated runs during development don't re-hit those APIs every time; pass
--refresh to force a refetch.

Usage:
    .venv/bin/python3 tw_backtest_main.py
    .venv/bin/python3 tw_backtest_main.py --rebalance YS --budget 8 --refresh
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from tw_data import build_candidate_prices, cached_series
from tw_backtest import (
    run_backtest, run_equal_weight_baseline, equity_curve, benchmark_curve,
    fx_adjusted_benchmark_curve, summarize, curve_metrics,
)
from quantum_solver import check_versions
from plotting import plot_backtest_equity_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool-size", type=int, default=20, help="Candidate pool size (= qubit count)")
    p.add_argument("--budget", type=int, default=10, help="Number of stocks to hold (K)")
    p.add_argument("--risk-factor", type=float, default=0.5, help="Risk-aversion coefficient q")
    p.add_argument("--lookback-years", type=float, default=2.0, help="Trailing window for mu/Sigma at each rebalance")
    p.add_argument("--rebalance", default="QS", choices=["YS", "QS", "MS"], help="YS=yearly, QS=quarterly, MS=monthly")
    p.add_argument("--start", default="2021-01-01", help="Backtest start date")
    p.add_argument("--end", default=None, help="Backtest end date (default: today)")
    p.add_argument("--refresh", action="store_true", help="Bypass the local cache and refetch everything")
    p.add_argument("--output", default="outputs/tw_backtest_equity_curves.png")
    p.add_argument("--curves-csv", default="outputs/tw_backtest_curves.csv",
                   help="Equity curves as CSV (read by the Streamlit app's backtest section)")
    return p.parse_args()




def main() -> None:
    args = parse_args()
    end = args.end or date.today().isoformat()

    version_warning = check_versions()
    if version_warning:
        print(f"WARNING: {version_warning}")

    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=int(args.lookback_years * 365.25) + 30)).date().isoformat()

    print(f"[1/4] Building candidate pool + price history (start={fetch_start} for lookback headroom)")
    tw_prices, pool = build_candidate_prices(args.pool_size, fetch_start, args.refresh)
    bm_0050 = cached_series("bm_0050", "0050.TW", fetch_start, args.refresh).ffill()
    sp500 = cached_series("bm_sp500", "^GSPC", fetch_start, args.refresh).ffill()
    fx = cached_series("fx_twdusd", "TWD=X", fetch_start, args.refresh).ffill()

    candidate_tickers = list(tw_prices.columns)
    candidates = tw_prices[candidate_tickers]

    print(f"\n[2/4] Rolling {args.rebalance} rebalance backtest, {args.lookback_years}y lookback, "
          f"budget={args.budget} of {len(candidate_tickers)}, {args.start} -> {end}")
    periods = run_backtest(
        candidates, lookback_years=args.lookback_years, rebalance_freq=args.rebalance,
        budget=args.budget, risk_factor=args.risk_factor, start=args.start, end=end,
    )
    print(summarize(periods))

    strategy_curve = equity_curve(periods)
    dates = list(strategy_curve.index)

    print(f"\n[3/4] Control: equal-weight all {len(candidate_tickers)} candidates, no QUBO selection")
    control_periods = run_equal_weight_baseline(candidates, args.rebalance, args.start, end)
    control_curve = equity_curve(control_periods)

    curve_0050 = benchmark_curve(bm_0050, dates)
    curve_sp500 = benchmark_curve(sp500, dates)
    curve_sp500_twd = fx_adjusted_benchmark_curve(sp500, fx, dates)

    print("\n[4/4] Cumulative return summary")
    fx_start, fx_end = fx.loc[fx.index <= dates[0]].iloc[-1], fx.loc[fx.index <= dates[-1]].iloc[-1]
    print(f"  TWD/USD: {fx_start:.2f} -> {fx_end:.2f} ({fx_end/fx_start-1:+.2%} TWD move vs USD over the window)")
    n_years = (dates[-1] - dates[0]).days / 365.25
    results = [
        (f"QUBO-selected ({args.budget} of {len(candidate_tickers)})", strategy_curve),
        (f"Equal-weight ALL {len(candidate_tickers)} (control)", control_curve),
        ("0050.TW", curve_0050),
        ("S&P 500 (raw USD, unhedged)", curve_sp500),
        ("S&P 500 (TWD-adjusted)", curve_sp500_twd),
    ]
    periods_per_year = {"YS": 1, "QS": 4, "MS": 12}[args.rebalance]
    for name, curve in results:
        m = curve_metrics(curve, periods_per_year)
        print(f"  {name:34s} total={m['total_return']:+8.2%}  annualized={m['annualized_return']:+8.2%}  "
              f"vol={m['annualized_vol']:.2%}  sharpe={m['sharpe']:.2f}")

    qubo_ann = (strategy_curve.iloc[-1] / strategy_curve.iloc[0]) ** (1 / n_years) - 1
    control_ann = (control_curve.iloc[-1] / control_curve.iloc[0]) ** (1 / n_years) - 1
    print(f"\n  Selection effect (QUBO annualized - control annualized): {qubo_ann - control_ann:+.2%}")
    print("  (Isolates the QUBO mean-variance selection's contribution from the")
    print("   candidate pool's own look-ahead-biased construction -- see README.)")

    out = pd.DataFrame({
        "qubo_strategy": strategy_curve,
        "equal_weight_control": control_curve,
        "0050_TW": curve_0050,
        "sp500": curve_sp500,
        "sp500_twd": curve_sp500_twd,
    })
    Path("outputs").mkdir(exist_ok=True)
    plot_backtest_equity_curves(out, args.output)
    print(f"\nSaved {args.output}")
    # The Streamlit app reads these precomputed curves instead of re-running
    # ~23 exact QUBO solves on every page load.
    out.to_csv(args.curves_csv, index_label="date")
    print(f"Saved {args.curves_csv}")


if __name__ == "__main__":
    main()
