"""Taiwan QUBO Portfolio Tool.

A tool, not a demo: on open it shows the current recommended portfolio
(computed once via cardinality-constrained QUBO/exact solve over a
liquidity-screened TWSE/TPEx large-cap pool, cached), how it has
actually performed in TWD since the day it was created, and how that
compares to 0050 / 0056 / 00878 / the S&P 500 (currency-adjusted) over
the same window. You can edit the holdings, add tickers not in the
default pool, force specific names into the next recompute, size
positions by board lot (1000 shares), and export the result.

Run with: .venv/bin/streamlit run tw_portfolio_tool.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tw_data import CACHE_DIR, cached_series, fetch_yf_history
from tw_portfolio import (
    PortfolioState, allocate_shares, candidate_stats, compute_default_portfolio, ex_ante_volatility, load_state,
)
from qubo_portfolio import build_cardinality_qubo
from quantum_solver import check_versions, solve_qaoa
from tw_backtest import periodic_win_loss
from plotting import plot_live_tracking, plot_periodic_win_loss

BENCHMARKS = {
    "0050.TW": "元大台灣50",
    "0056.TW": "元大高股息",
    "00878.TW": "國泰永續高股息",
}

st.set_page_config(page_title="Taiwan QUBO Portfolio Tool", page_icon="🎯", layout="wide")

_version_warning = check_versions()
if _version_warning:
    st.error(_version_warning, icon="🚨")


def _init_state() -> None:
    st.session_state.setdefault("portfolio", load_state())
    st.session_state.setdefault("held_override", None)
    st.session_state.setdefault("qaoa_job", None)


_init_state()

st.title("🎯 Taiwan QUBO Portfolio Tool")
st.caption(
    "Cardinality-constrained mean-variance selection over TWSE/TPEx large caps, "
    "tracked live in TWD against 0050 / 0056 / 00878 / S&P 500."
)

with st.sidebar:
    st.header("Portfolio settings")
    pool_size = st.slider("Candidate pool size", 10, 25, 20, help="= qubit count for the QUBO/QAOA solve")
    budget = st.slider("Stocks to hold (K)", 3, pool_size - 1, min(10, pool_size - 1))
    risk_factor = st.slider("Risk aversion (q)", 0.0, 2.0, 0.5, step=0.05)
    lookback_years = st.slider("Lookback window (years)", 0.5, 3.0, 2.0, step=0.5)

    st.divider()
    st.subheader("Add a specific ticker")
    extra_ticker = st.text_input("Yahoo Finance ticker (e.g. 3008.TW)", value="")
    force_include = st.checkbox("Force this ticker into the portfolio", value=False)

    st.divider()
    recompute = st.button("🔄 Recompute best portfolio now", type="primary", width="stretch")
    st.caption("Recomputing resets the live-tracking start date to today.")

if recompute:
    with st.spinner("Fetching TWSE/TPEx universe and solving the QUBO..."):
        try:
            extras = [extra_ticker.strip()] if extra_ticker.strip() else None
            forced = extras if (extras and force_include) else None
            state = compute_default_portfolio(
                pool_size=pool_size, budget=budget, risk_factor=risk_factor,
                lookback_years=lookback_years, refresh=True,
                extra_tickers=extras, force_include=forced,
            )
            st.session_state["portfolio"] = state
            st.session_state["held_override"] = None
            st.success(f"Recomputed. Selected {len(state.tickers)} of {pool_size}.")
        except Exception as e:
            st.error(f"Recompute failed: {e}")

state: PortfolioState | None = st.session_state["portfolio"]

if state is None:
    st.info("No portfolio computed yet. Click **Recompute best portfolio now** in the sidebar to build one "
            "(fetches live TWSE/TPEx data — takes ~1-2 minutes the first time).")
    st.stop()

pool_df = pd.DataFrame(state.candidate_pool)
all_pool_tickers = pool_df["yf_ticker"].tolist() if not pool_df.empty else state.tickers
pool_names = dict(zip(pool_df["yf_ticker"], pool_df["name"])) if not pool_df.empty else {}
pool_names.update(state.names)

st.subheader("1. Current recommended portfolio")
inception_dt = pd.Timestamp(state.inception_date)
days_held = (pd.Timestamp.today().normalize() - inception_dt).days
c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdings", f"{len(state.tickers)} of {state.pool_size}")
c2.metric("Risk aversion (q)", f"{state.risk_factor}")
c3.metric("Set on", state.inception_date)
c4.metric("Days tracked", days_held)

st.markdown("**Edit holdings** (unchecking removes a stock; the rest is re-weighted equally — no re-solve needed):")
options = sorted(set(all_pool_tickers) | set(state.tickers))
default_held = st.session_state["held_override"] or state.tickers
held = st.multiselect(
    "Held tickers", options=options, default=default_held,
    format_func=lambda t: f"{t} · {pool_names.get(t, t)}",
)
st.session_state["held_override"] = held

with st.expander("Per-candidate return / volatility / Sharpe (to inform edits above)", expanded=True):
    stats_df = candidate_stats(state, held)
    if stats_df.empty:
        st.caption("Not available — this portfolio was computed before this feature was added. "
                   "Recompute to populate it.")
    else:
        st.caption(
            "Sorted by Sharpe descending. **mu** and **volatility** are each stock's own annualized "
            "expected return and std. dev. from the exact μ/Σ that drove the QUBO's selection (not a "
            "fresh estimate) — volatility here is each stock alone, ignoring covariance with the rest "
            "of the portfolio; that's a property of the stock, not of any specific combination. For "
            "portfolio-level (covariance-aware) volatility of what's actually held, see the Risk "
            "section below."
        )
        display_df = stats_df.copy()
        display_df["held"] = display_df["held"].map({True: "✓", False: ""})
        st.dataframe(
            display_df.style.format({"mu": "{:+.2%}", "volatility": "{:.2%}", "sharpe": "{:.2f}"})
            .background_gradient(subset=["sharpe"], cmap="RdYlGn"),
            width="stretch", hide_index=True,
        )

if not held:
    st.warning("Select at least one ticker to see performance.")
    st.stop()

capital = st.number_input("Capital to allocate (NT\\$)", min_value=10_000, value=1_000_000, step=10_000)


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_live_history(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        s = fetch_yf_history(t, start)
        if s is not None:
            frames[t] = s
    return pd.DataFrame(frames).ffill()


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_benchmark_history(start: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    bm = _fetch_live_history(tuple(BENCHMARKS.keys()), start)
    sp500 = cached_series("live_sp500", "^GSPC", start, refresh=False)
    fx = cached_series("live_fx_twdusd", "TWD=X", start, refresh=False)
    return bm, sp500, fx


with st.spinner("Fetching live prices..."):
    held_prices_hist = _fetch_live_history(tuple(held), state.inception_date)
    bm_hist, sp500_hist, fx_hist = _fetch_benchmark_history(state.inception_date)

# yfinance occasionally fails a same-day fetch outright (seen repeatedly
# in testing -- transient, not structural: a moment later the same call
# succeeds). Don't hard-fail the whole page for it: any held ticker
# missing from the fresh fetch falls back to its stored inception price
# (a stale-but-correct single point) rather than blocking the tool.
current_prices: dict[str, float] = {}
stale_tickers = []
for t in held:
    if t in held_prices_hist.columns and not held_prices_hist[t].dropna().empty:
        current_prices[t] = float(held_prices_hist[t].dropna().iloc[-1])
    elif t in state.inception_prices:
        current_prices[t] = state.inception_prices[t]
        stale_tickers.append(t)
    else:
        st.error(f"No price data available for {t} (not fetchable, and not in the stored portfolio). "
                 "Remove it from Held tickers or try again in a moment.")
        st.stop()

if stale_tickers:
    st.caption(f"⚠️ Live price fetch failed for {stale_tickers} just now (transient — Yahoo Finance "
               "sometimes drops a same-day request); showing the last known price instead. Reload to retry.")

weights = {t: 1.0 / len(held) for t in held}

st.subheader("2. Live performance since portfolio was set")

# Equal-weighted buy-and-hold index for the current holding set, no
# rebalancing in between (this is "what you'd actually have" from
# inception to now, not a re-optimized backtest). Tickers missing from
# the fresh fetch (see stale_tickers above) get a flat one-point series
# at their stored inception price so they don't break the average.
held_hist_complete = held_prices_hist.reindex(columns=held)
for t in stale_tickers:
    held_hist_complete[t] = current_prices[t]
norm = held_hist_complete / held_hist_complete.iloc[0]
portfolio_curve = norm.mean(axis=1)

bm_curves = {}
for tkr in BENCHMARKS:
    if tkr in bm_hist.columns and not bm_hist[tkr].dropna().empty:
        bm_curves[tkr] = bm_hist[tkr] / bm_hist[tkr].iloc[0]
sp500_curve = sp500_hist / sp500_hist.iloc[0]
fx_curve = fx_hist / fx_hist.iloc[0]
sp500_twd_curve = sp500_curve * fx_curve.reindex(sp500_curve.index).ffill()

total_return = portfolio_curve.iloc[-1] - 1
bm_0050_return = bm_curves["0050.TW"].iloc[-1] - 1 if "0050.TW" in bm_curves else float("nan")
sp500_twd_return = sp500_twd_curve.iloc[-1] - 1
m1, m2, m3 = st.columns(3)
m1.metric("Portfolio return (TWD)", f"{total_return:+.2%}")
m2.metric("vs 0050.TW", f"{total_return - bm_0050_return:+.2%}",
           help="Portfolio return minus 0050.TW return over the same window")
m3.metric("vs S&P 500 (TWD)", f"{total_return - sp500_twd_return:+.2%}",
           help="Portfolio return minus TWD-adjusted S&P 500 return over the same window")

if days_held < 5:
    st.caption("⚠️ This portfolio was set very recently — a few days of return is noise, not signal. "
               "The comparison becomes meaningful after weeks/months of live tracking.")

# --- Risk (variance/volatility) -------------------------------------
# Return alone makes a mean-variance-optimized portfolio look strictly
# worse than a naive equal-weight one whenever the naive one happened to
# win on raw return (as it did in this project's own historical
# backtest) -- that's the point of mean-variance optimization, and it's
# invisible without also showing risk. Two different, clearly-labeled
# numbers, not one:
MIN_REALIZED_VOL_DAYS = 10


def _realized_vol(curve: pd.Series) -> float | None:
    """Annualized std. dev. of the COMBINED portfolio curve's daily
    returns -- not an average of each holding's individual volatility.
    Since Var(sum_i w_i r_i) = w^T Sigma w by definition, taking std()
    of the already-combined equal-weighted series folds in every
    covariance term automatically; averaging individual vols would
    silently discard them (and be wrong -- it ignores diversification
    or under-counts correlated risk depending on the sign of the
    covariances). Same reasoning applies to ex_ante_volatility()'s
    explicit w @ Sigma @ w in tw_portfolio.py.
    """
    rets = curve.pct_change().dropna()
    if len(rets) < MIN_REALIZED_VOL_DAYS:
        return None
    return float(rets.std() * np.sqrt(252))


ex_ante_vol, ex_ante_gap = ex_ante_volatility(state, held)
ex_ante_vol_equal, _ = ex_ante_volatility(state, state.sigma_tickers)  # naive control: ALL pool tickers, equal-weight
realized_vol = _realized_vol(portfolio_curve)

st.markdown("**Risk**")
r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Ex-ante volatility (from Σ)",
    f"{ex_ante_vol:.2%}" if ex_ante_vol is not None else "N/A",
    help="sqrt(wᵀΣw), equal weights, using the SAME annualized variance-covariance matrix the QUBO "
         "optimized against at selection time -- a model prediction, not an observation. Covariance "
         "terms (off-diagonal Σ) are included by construction, not just each stock's own variance.",
)
r2.metric(
    "Ex-ante, equal-weight ALL pool",
    f"{ex_ante_vol_equal:.2%}" if ex_ante_vol_equal is not None else "N/A",
    delta=f"{ex_ante_vol - ex_ante_vol_equal:+.2%}" if (ex_ante_vol is not None and ex_ante_vol_equal is not None) else None,
    delta_color="inverse",  # lower QUBO vol vs. the naive control shows as "good" (green)
    help="Same sqrt(wᵀΣw) formula, but w = equal weight across ALL candidates in the pool -- the "
         "'naive control,' computed immediately from the same Σ, no waiting on live data. This is "
         "how to compare QUBO-selected vs. equal-weight risk RIGHT NOW without needing realized "
         "history: both numbers exist the moment the portfolio is computed.",
)
r3.metric(
    "Realized volatility (since incep.)",
    f"{realized_vol:.2%}" if realized_vol is not None else f"N/A (<{MIN_REALIZED_VOL_DAYS}d)",
    help="Annualized std. dev. of the portfolio's actual daily returns since inception -- what has "
         "actually happened, not a model prediction. N/A on a fresh portfolio is the honest answer, "
         "not a bug -- see the caption below for why this isn't backfilled with a faster substitute.",
)
sharpe = (total_return / (days_held / 365.25) / realized_vol) if (realized_vol and days_held >= 20) else None
r4.metric(
    "Realized Sharpe (ann.)",
    f"{sharpe:.2f}" if sharpe is not None else "N/A (<20d)",
    help="Annualized return over the tracked period, divided by realized annualized volatility "
         "(rf=0). Not meaningful with only a few weeks of history.",
)
st.caption(
    "**Comparing risk against 0050/S&P 500 specifically**: those aren't part of the pool Σ was "
    "estimated over, so there's no ex-ante number for them here — but unlike a fresh portfolio, "
    "they have decades of real trading history, so genuine realized volatility for them is "
    "available *right now*, just not through this live tracker (too new to be informative yet). "
    "Use section 5 below with a window of a year or more: it computes real, comparable annualized "
    "volatility for the QUBO strategy, the equal-weight control, 0050, and S&P 500 together, using "
    "proper out-of-sample rolling re-selection at each rebalance — not today's fixed basket "
    "backfilled, which is the mistake this tool used to make (see 'On not backfilling realized "
    "volatility' in the README)."
)
if ex_ante_gap:
    st.caption(
        f"⚠️ Ex-ante volatility excludes {ex_ante_gap} — added manually and not part of the original "
        "candidate pool the covariance matrix (Σ) was estimated over, so their covariance with the "
        "rest of the portfolio isn't available. Realized volatility above still includes them "
        "(it's computed from actual price history, not Σ)."
    )
st.caption(
    "Where Σ is used: `src/tw_portfolio.py`'s `compute_default_portfolio()` estimates Σ (annualized "
    "sample covariance of daily log returns over the lookback window) and hands it straight to the "
    "QUBO (`q · wᵀΣw − μᵀw`) that selects this portfolio — the same matrix, not a re-estimate, "
    "drives both ex-ante numbers above via `ex_ante_volatility()`. Realized volatility and the Sharpe "
    "ratio use actual observed prices instead, since Σ only reflects what was true as of the "
    "lookback window ending at selection time."
)
st.caption(
    "Why realized volatility isn't backfilled to avoid the N/A: the only way to get an 'immediate' "
    "realized number would be to apply *today's* selected basket and weights to its constituents' "
    "historical prices — that's a pro-forma backtest of the current selection, not something anyone "
    "actually held, and it's circular besides (this basket was partly chosen *for* its historical "
    "correlation structure, so 'realized' vol of it over the same window mostly re-derives the "
    "ex-ante number under a misleading label). It would also sit in the same column as benchmarks' "
    "genuinely realized volatility, implying a fact-vs-fact comparison that isn't there. If you want "
    "a properly-labeled backtest of this methodology over a past window, use section 5 below — it's "
    "already framed as a backtest, not live tracking."
)

if len(portfolio_curve) < 2:
    st.info("Chart needs at least 2 trading days of history — check back after the next close. "
            "This portfolio was set today, so there's only one price point so far.", icon="📈")
else:
    live_curves = {"portfolio": portfolio_curve, **bm_curves, "sp500_twd": sp500_twd_curve}
    st.plotly_chart(plot_live_tracking(live_curves), width="stretch")

comparison_rows = [{"name": "Your portfolio", "return": total_return, "volatility": realized_vol}]
for tkr, name in BENCHMARKS.items():
    if tkr in bm_curves:
        comparison_rows.append({"name": f"{tkr} ({name})", "return": bm_curves[tkr].iloc[-1] - 1,
                                 "volatility": _realized_vol(bm_curves[tkr])})
comparison_rows.append({"name": "S&P 500 (TWD-adjusted)", "return": sp500_twd_return,
                         "volatility": _realized_vol(sp500_twd_curve)})
comparison_rows.append({"name": "S&P 500 (raw USD)", "return": sp500_curve.iloc[-1] - 1,
                         "volatility": _realized_vol(sp500_curve)})
comp_df = pd.DataFrame(comparison_rows).sort_values("return", ascending=False)
# st.dataframe's Arrow-based rendering doesn't reliably honor Styler's
# na_rep for NaN cells (verified: works fine in a plain .style.to_html(),
# renders literal "None" through st.dataframe) -- format volatility as a
# display string manually instead of leaving it to the Styler.
comp_df["volatility"] = comp_df["volatility"].apply(
    lambda v: f"{v:.2%}" if pd.notna(v) else f"N/A (<{MIN_REALIZED_VOL_DAYS}d)"
)
st.dataframe(
    comp_df.style.format({"return": "{:+.2%}"})
    .background_gradient(subset=["return"], cmap="RdYlGn"),
    width="stretch", hide_index=True,
)
st.caption(
    "Both columns are since-inception for every row, including benchmarks — an apples-to-apples "
    "window throughout, not a mix of realized and pro-forma numbers. Volatility reads N/A until "
    f"≥{MIN_REALIZED_VOL_DAYS} trading days have accumulated; that's honest, not a bug (see the "
    "caption in the Risk section above for why it isn't backfilled with a faster substitute)."
)

st.subheader("3. Position sizing")
odd_lot = st.checkbox(
    "Allow odd-lot (零股) trading — single shares, not just full board lots",
    value=True,
    help="Off = round to whole board lots (1000 shares/一張) only. At smaller capital split across "
         "many names, board-lot-only rounding can zero out expensive stocks entirely (e.g. a single "
         "lot of a NT\\$2,000 stock needs NT\\$2M) — most brokers now support odd-lot orders, so this "
         "defaults on.",
)
alloc = allocate_shares(held, weights, current_prices, capital, odd_lot=odd_lot)
alloc_display = alloc.copy()
alloc_display["name"] = alloc_display["ticker"].map(lambda t: pool_names.get(t, t))
alloc_display = alloc_display[["ticker", "name", "weight", "price", "lots", "shares", "cost", "leftover_cash"]]
st.dataframe(
    alloc_display.style.format({
        "weight": "{:.1%}", "price": "NT${:,.2f}", "lots": "{:.3f}", "cost": "NT${:,.0f}", "leftover_cash": "NT${:,.0f}",
    }),
    width="stretch", hide_index=True,
)
total_cost = alloc["cost"].sum()
total_leftover = capital - total_cost
unit_desc = "single-share (零股)" if odd_lot else "board-lot (1000-share/一張)"
st.caption(f"Total deployed: NT\\${total_cost:,.0f}  ·  Cash leftover from {unit_desc} rounding: "
           f"NT\\${total_leftover:,.0f} ({total_leftover/capital:.1%} of capital).")
if not odd_lot and total_leftover / capital > 0.15:
    st.warning(
        f"{total_leftover/capital:.0%} of capital is sitting unallocated because of board-lot rounding. "
        "Either increase capital or enable odd-lot trading above.",
        icon="⚠️",
    )

st.download_button(
    "⬇️ Export portfolio (CSV)",
    alloc_display.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"tw_portfolio_{state.inception_date}.csv",
    mime="text/csv",
)

st.subheader("4. Quantum verification (optional)")
st.caption(
    "Re-solves the same QUBO with QAOA instead of exact diagonalization, on the pool/budget/risk "
    "settings the current portfolio was computed with — exact is what actually drives the "
    "recommendation above; this is a cross-check, not a better answer."
)
qaoa_col1, qaoa_col2 = st.columns([1, 2])
with qaoa_col1:
    qaoa_reps = st.slider("QAOA reps", 1, 3, 1, key="qaoa_reps")
    qaoa_maxiter = st.slider("COBYLA maxiter", 10, 100, 30, key="qaoa_maxiter")

    # Exponential, not quadratic -- statevector simulation cost is
    # fundamentally O(2^n). Calibrated on two real measurements from this
    # project: 10 qubits/reps=1/maxiter=50 -> 200s (4s/iter), and 20
    # qubits/reps=1/maxiter=20 -> still running after 59 min of CPU time
    # (>=180s/iter, lower bound only). That's a ~45x per-iteration jump
    # for +10 qubits -> per-iter(n) = 4s * 1.51^(n-10). A quadratic model
    # badly undersold this in an earlier version of this estimate.
    n_qubits = state.pool_size
    per_iter_s = 4 * (1.51 ** (n_qubits - 10))
    est_s = per_iter_s * qaoa_maxiter * qaoa_reps
    est_low_min, est_high_min = max(1, round(est_s / 60 * 0.5)), round(est_s / 60 * 2.5)
    st.warning(
        f"At {n_qubits} qubits, this configuration is a **rough order-of-magnitude estimate of "
        f"{est_low_min}-{est_high_min} minutes** — wide range because it's extrapolated from just two "
        "measurements. Grounded in this project's own benchmarks, not a guess: 20 qubits at "
        "reps=1/maxiter=20 was still running after nearly an hour of CPU time in testing. Runs in the "
        "background so the rest of the tool stays usable. Lower the pool size, reps, or maxiter for a "
        "faster (rougher) check.",
        icon="⏱️",
    )

    qaoa_job = st.session_state["qaoa_job"]
    qaoa_running = qaoa_job is not None and qaoa_job["status"] == "running"
    if st.button("Run QAOA verification", disabled=qaoa_running):
        try:
            tickers_all = pool_df["yf_ticker"].tolist()
            prices_hist = _fetch_live_history(
                tuple(tickers_all),
                (pd.Timestamp(state.inception_date) - pd.Timedelta(days=int(state.lookback_years * 365.25))).date().isoformat(),
            )
            window = prices_hist.dropna()
            log_ret = np.log(window / window.shift(1)).dropna()
            mu = (log_ret.mean() * 252).values
            sigma = (log_ret.cov() * 252).values
            qp, _ = build_cardinality_qubo(mu, sigma, budget=state.budget, risk_factor=state.risk_factor)

            job = {"status": "running", "run": None, "error": None, "start_time": time.time(), "tickers": tickers_all}

            def _worker(qp=qp, reps=qaoa_reps, maxiter=qaoa_maxiter, job=job):
                try:
                    job["run"] = solve_qaoa(qp, reps=reps, maxiter=maxiter)
                    job["status"] = "done"
                except Exception as e:  # noqa: BLE001
                    job["error"] = str(e)
                    job["status"] = "error"

            threading.Thread(target=_worker, daemon=True).start()
            st.session_state["qaoa_job"] = job
            st.rerun()
        except Exception as e:
            st.error(f"Could not start QAOA verification: {e}")

with qaoa_col2:
    @st.fragment(run_every=2)
    def _qaoa_status():
        job = st.session_state["qaoa_job"]
        if job is None:
            st.caption("Not run yet.")
            return
        if job["status"] == "running":
            st.info(f"Running... elapsed {time.time()-job['start_time']:.0f}s", icon="⏳")
        elif job["status"] == "error":
            st.error(f"Failed: {job['error']}")
        elif job["status"] == "done":
            run = job["run"]
            selected = [t for t, bit in zip(job["tickers"], run.result.x) if round(bit) == 1]
            gap = run.result.fval - state.objective
            st.success(f"QAOA finished in {run.wall_time_s:.0f}s. objective={run.result.fval:.4f} "
                       f"(gap vs exact = {gap:+.4f})")
            agree = set(selected) == set(state.tickers)
            st.write(("✅ Same selection as the exact solver." if agree else
                      f"⚠️ Different selection: {[t for t in selected if t not in state.tickers]} in, "
                      f"{[t for t in state.tickers if t not in selected]} out."))

    _qaoa_status()

with st.expander("5. Historical backtest reference (choose your own window)"):
    st.caption(
        "Supporting context, not the live number above: how the same pool-size/budget/risk-factor QUBO "
        "selection would have performed holding from a date you pick through another date you pick "
        "(today by default). **Deliberately not fixed to a long window** — a strategy's edge can be "
        "short-lived (see 'On alpha decay' in the README); testing a week or a month, not just multi-year "
        "windows, is the point. See README for the full methodology and its look-ahead caveats (the "
        "candidate pool is ranked by *today's* market caps, applied retroactively)."
    )
    bt_col1, bt_col2, bt_col3 = st.columns(3)
    today_ts = pd.Timestamp.today().normalize()
    with bt_col1:
        hist_start_date = st.date_input(
            "Start holding",
            value=(today_ts - pd.Timedelta(days=365)).date(),
            min_value=pd.Timestamp("2015-01-01").date(),
            max_value=(today_ts - pd.Timedelta(days=1)).date(),
        )
    with bt_col2:
        hist_end_date = st.date_input(
            "Test until",
            value=today_ts.date(),
            min_value=(pd.Timestamp(hist_start_date) + pd.Timedelta(days=1)).date(),
            max_value=today_ts.date(),
        )
    with bt_col3:
        rebalance_label = st.selectbox(
            "Rebalance frequency", ["Buy-and-hold (no rebalancing)", "Monthly", "Quarterly", "Yearly"], index=2,
        )
    rebalance_map = {"Buy-and-hold (no rebalancing)": None, "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    hist_rebalance_freq = rebalance_map[rebalance_label]
    window_days = (hist_end_date - hist_start_date).days
    if window_days < 14:
        st.caption(f"⚠️ {window_days}-day window — return/volatility here will be very noisy at this length; "
                   "useful for spotting whether an edge is even directionally present short-term, not for "
                   "precise numbers.")

    if st.button("Run historical backtest"):
        with st.spinner("Running..."):
            from tw_data import build_candidate_prices
            from tw_backtest import run_backtest, run_equal_weight_baseline, equity_curve, benchmark_curve, fx_adjusted_benchmark_curve
            from plotting import plot_backtest_equity_curves

            hist_start = hist_start_date.isoformat()
            hist_end = hist_end_date.isoformat()
            fetch_start = (pd.Timestamp(hist_start) - pd.Timedelta(days=int(state.lookback_years * 365.25) + 30)).date().isoformat()
            hist_prices, _ = build_candidate_prices(state.pool_size, fetch_start, refresh=False)
            hist_0050 = cached_series("bm_0050", "0050.TW", fetch_start, refresh=False)
            hist_sp500 = cached_series("bm_sp500", "^GSPC", fetch_start, refresh=False)
            hist_fx = cached_series("fx_twdusd", "TWD=X", fetch_start, refresh=False)

            periods = run_backtest(hist_prices, lookback_years=state.lookback_years, rebalance_freq=hist_rebalance_freq,
                                    budget=state.budget, risk_factor=state.risk_factor, start=hist_start, end=hist_end)
            strategy_curve = equity_curve(periods)
            dates = list(strategy_curve.index)
            control_curve = equity_curve(run_equal_weight_baseline(hist_prices, hist_rebalance_freq, hist_start, dates[-1].date().isoformat()))
            curve_0050 = benchmark_curve(hist_0050, dates)
            curve_sp500_twd = fx_adjusted_benchmark_curve(hist_sp500, hist_fx, dates)

            out = pd.DataFrame({
                "qubo_strategy": strategy_curve, "equal_weight_control": control_curve,
                "0050_TW": curve_0050, "sp500_twd": curve_sp500_twd,
            })
            Path("outputs").mkdir(exist_ok=True)
            plot_backtest_equity_curves(out, "outputs/tw_portfolio_tool_backtest.png")
            st.image("outputs/tw_portfolio_tool_backtest.png")

            n_years_hist = max(window_days / 365.25, 1 / 252)  # floor so a 1-day window doesn't divide-by-~0
            summary_rows = []
            for label, curve in [("QUBO-selected", strategy_curve), ("Equal-weight control", control_curve),
                                  ("0050.TW", curve_0050), ("S&P 500 (TWD-adj)", curve_sp500_twd)]:
                tot = curve.iloc[-1] / curve.iloc[0] - 1
                ann = (curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years_hist) - 1
                vol = _realized_vol(curve)
                summary_rows.append({"strategy": label, "total_return": tot, "annualized": ann,
                                      "volatility": vol if vol is not None else float("nan")})
            summary_df = pd.DataFrame(summary_rows)
            # st.dataframe's Arrow rendering doesn't honor Styler na_rep for
            # NaN (see section 2's comparison table for the same fix) --
            # format volatility as a display string manually instead.
            summary_df["volatility"] = summary_df["volatility"].apply(
                lambda v: f"{v:.2%}" if pd.notna(v) else f"N/A (<{MIN_REALIZED_VOL_DAYS}d)"
            )
            st.dataframe(
                summary_df.style.format({"total_return": "{:+.2%}", "annualized": "{:+.2%}"}),
                width="stretch", hide_index=True,
            )

st.subheader("6. Weekly win/loss: your selection vs. equal-weight-all")
st.caption(
    "Two FIXED baskets, same as the Risk section above — your currently held tickers vs. equal-weight "
    "across the whole candidate pool. Neither is re-optimized within a period; this isn't a rolling "
    "rebalance backtest, it's the SAME two portfolios sliced into weekly chunks so you can see WHICH "
    "weeks your selection actually won, not just one blended average over the whole window. Directly "
    "answers the alpha-decay question from section 5: is any edge spread evenly across weeks, "
    "concentrated in a few, or reversing over time?"
)
wl_col1, wl_col2 = st.columns(2)
_wl_default_start = min(
    pd.Timestamp(state.inception_date), pd.Timestamp.today().normalize() - pd.Timedelta(days=90)
).date()
with wl_col1:
    wl_start_date = st.date_input(
        "Since", value=_wl_default_start,
        min_value=pd.Timestamp("2015-01-01").date(),
        max_value=(pd.Timestamp.today().normalize() - pd.Timedelta(days=7)).date(),
        key="wl_start",
        help="Defaults to your portfolio's inception date, or 90 days ago if that's more recent "
             "(so there's enough data to show weekly patterns on a freshly-computed portfolio).",
    )
with wl_col2:
    wl_end_date = st.date_input(
        "Until", value=pd.Timestamp.today().normalize().date(),
        min_value=(pd.Timestamp(wl_start_date) + pd.Timedelta(days=7)).date(),
        max_value=pd.Timestamp.today().normalize().date(),
        key="wl_end",
    )

if st.button("Compare weekly"):
    with st.spinner("Fetching price history..."):
        wl_start = wl_start_date.isoformat()
        wl_held_hist = _fetch_live_history(tuple(held), wl_start)
        wl_all_hist = _fetch_live_history(tuple(state.sigma_tickers), wl_start)

    wl_held_ok = [t for t in held if t in wl_held_hist.columns and not wl_held_hist[t].dropna().empty]
    wl_all_ok = [t for t in state.sigma_tickers if t in wl_all_hist.columns and not wl_all_hist[t].dropna().empty]
    missing_wl = sorted(set(held) - set(wl_held_ok))
    missing_all = sorted(set(state.sigma_tickers) - set(wl_all_ok))
    if missing_wl or missing_all:
        st.caption(f"⚠️ Excluded from this comparison (no price data over the chosen window): "
                   f"held={missing_wl or 'none'}, pool={missing_all or 'none'}.")

    wl_held_complete = wl_held_hist[wl_held_ok].dropna() if wl_held_ok else pd.DataFrame()
    wl_all_complete = wl_all_hist[wl_all_ok].dropna() if wl_all_ok else pd.DataFrame()

    if wl_held_complete.empty or wl_all_complete.empty:
        st.error("Not enough overlapping price history in this window to compare.")
    else:
        qubo_wl_curve = (wl_held_complete / wl_held_complete.iloc[0]).mean(axis=1)
        equal_wl_curve = (wl_all_complete / wl_all_complete.iloc[0]).mean(axis=1)
        wl_df = periodic_win_loss(qubo_wl_curve, equal_wl_curve, "QUBO", "Equal-weight", freq="W-MON")

        if wl_df.empty:
            st.info("Window too short for even one full week — pick an earlier start date.")
        else:
            n_weeks = len(wl_df)
            n_qubo_wins = int((wl_df["winner"] == "QUBO").sum())
            n_ties = int((wl_df["winner"] == "Tie").sum())
            win_rate = n_qubo_wins / n_weeks
            avg_diff = float(wl_df["diff"].mean())

            m1, m2, m3 = st.columns(3)
            m1.metric("Weeks compared", n_weeks)
            m2.metric("QUBO won", f"{n_qubo_wins}/{n_weeks} ({win_rate:.0%})",
                      help=f"{n_ties} tie(s) counted in neither side's total.")
            m3.metric(
                "Avg weekly diff", f"{avg_diff:+.2%}",
                help="Mean(QUBO week return − Equal-weight week return), an unweighted average across "
                     "weeks — not the same number as comparing the two curves' total return over the "
                     "whole window (a few large weeks can dominate the aggregate but not this average).",
            )

            st.plotly_chart(plot_periodic_win_loss(wl_df, "QUBO"), width="stretch")

            display_wl = wl_df.copy()
            display_wl["period_start"] = display_wl["period_start"].dt.date
            display_wl["period_end"] = display_wl["period_end"].dt.date
            st.dataframe(
                display_wl.style.format({"QUBO": "{:+.2%}", "Equal-weight": "{:+.2%}", "diff": "{:+.2%}"})
                .background_gradient(subset=["diff"], cmap="RdYlGn"),
                width="stretch", hide_index=True,
            )
