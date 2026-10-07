"""Taiwan top-20 out-of-sample backtest: QUBO selection vs. equal-weight control.

Reads the precomputed equity curves written by tw_backtest_main.py
(outputs/tw_backtest_curves.csv) instead of re-running ~23 exact QUBO solves
and the TWSE/TPEx/yfinance data pipeline on every page load. Regenerate with:

    .venv/bin/python3 tw_backtest_main.py --end 2026-08-14
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plotting import backtest_curves_figure_plotly, backtest_metrics_figure_plotly  # noqa: E402
from tw_backtest import curve_metrics  # noqa: E402

CURVES_CSV = ROOT / "outputs" / "tw_backtest_curves.csv"
PERIODS_PER_YEAR = 4  # quarterly rebalance, matching the CLI default used for the CSV

st.set_page_config(page_title="Taiwan Backtest: QUBO vs Equal-Weight", page_icon="📊", layout="wide")
st.title("📊 Taiwan top-20 backtest: QUBO selection vs. equal-weight control")

if not CURVES_CSV.exists():
    st.error(f"Missing `{CURVES_CSV.relative_to(ROOT)}` — run `tw_backtest_main.py` to generate it.")
    st.stop()

curves = pd.read_csv(CURVES_CSV, index_col="date", parse_dates=True)
st.caption(
    f"{curves.index[0]:%Y-%m-%d} → {curves.index[-1]:%Y-%m-%d} · quarterly rebalance · "
    "choose 10 of 20 candidates · 2-year trailing μ/Σ at each rebalance · "
    "selection solved exactly (NumPyMinimumEigensolver), not with QAOA"
)

st.markdown(
    "**Why this comparison:** the *equal-weight control* holds all 20 candidates with no selection at all, "
    "on the same rebalance grid. Any gap between it and the QUBO strategy is the contribution of the "
    "QUBO's mean-variance selection step; anything both share comes from how the candidate pool was built."
)

metrics = {col: curve_metrics(curves[col], PERIODS_PER_YEAR) for col in curves.columns}
q, c = metrics["qubo_strategy"], metrics["equal_weight_control"]

m1, m2, m3 = st.columns(3)
m1.metric("QUBO-selected, annualized", f"{q['annualized_return']:.1%}",
          delta=f"{q['annualized_return'] - c['annualized_return']:+.1%} vs. control")
m2.metric("Equal-weight control, annualized", f"{c['annualized_return']:.1%}")
m3.metric("Sharpe: QUBO vs. control", f"{q['sharpe']:.2f} vs. {c['sharpe']:.2f}")

st.subheader("1. Cumulative growth")
show_bm = st.checkbox("Also show 0050.TW and S&P 500 (TWD-adjusted)", value=False)
st.plotly_chart(backtest_curves_figure_plotly(curves, show_bm), width="stretch")

st.subheader("2. Return, risk and Sharpe")
st.plotly_chart(backtest_metrics_figure_plotly(metrics), width="stretch")

st.subheader("How to read this")
st.markdown(
    f"- **The QUBO selection underperformed the naive control** on both return "
    f"({q['annualized_return']:.1%} vs. {c['annualized_return']:.1%} annualized) and Sharpe "
    f"({q['sharpe']:.2f} vs. {c['sharpe']:.2f}). Its objective `q·xᵀΣx − μᵀx` penalizes variance, so it "
    f"trims the most volatile names — lower risk ({q['annualized_vol']:.1%} vs. {c['annualized_vol']:.1%} vol), "
    "which costs return in a near-uninterrupted bull market like 2021–2026.\n"
    "- **Most of the outperformance over 0050 / S&P 500 comes from the candidate pool, not the optimizer.** "
    "The pool is today's top-20 by market cap applied back to 2021 (look-ahead and survivorship bias); "
    "the control, which has no selection logic, still beats every benchmark.\n"
    "- No transaction costs or taxes; see the README's *Limitations specific to this backtest*."
)

with st.expander("Table view"):
    table = pd.DataFrame(metrics).T.rename(index={
        "qubo_strategy": "QUBO-selected (10 of 20)",
        "equal_weight_control": "Equal-weight ALL 20 (control)",
        "0050_TW": "0050.TW",
        "sp500": "S&P 500 (raw USD)",
        "sp500_twd": "S&P 500 (TWD-adjusted)",
    })
    st.dataframe(table.style.format({
        "total_return": "{:+.1%}", "annualized_return": "{:+.1%}", "annualized_vol": "{:.1%}", "sharpe": "{:.2f}",
    }), width="stretch")
