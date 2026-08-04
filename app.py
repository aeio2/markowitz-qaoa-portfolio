"""Streamlit web app for the Markowitz + QUBO/QAOA portfolio project.

Two stages, matching the underlying pipeline:
  1. Classical continuous Markowitz (Monte Carlo + SLSQP efficient frontier) --
     fast, runs synchronously.
  2. Cardinality-constrained QUBO -> Ising -> exact ground state (fast) vs
     QAOA (can take minutes; runs in a background thread so the rest of the
     UI stays responsive while it grinds through the optimizer loop).

Run with: streamlit run app.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import annualized_moments, fetch_prices
from classical_mpt import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    monte_carlo_simulation,
    portfolio_performance,
)
from qubo_portfolio import build_cardinality_qubo, to_ising
from quantum_solver import check_versions, solve_exact, solve_qaoa
from plotting import frontier_figure_plotly, weights_figure_plotly

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "PG"]

st.set_page_config(page_title="Markowitz + QAOA Portfolio Optimizer", page_icon="📈", layout="wide")

_version_warning = check_versions()
if _version_warning:
    st.error(_version_warning, icon="🚨")


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_moments(tickers: tuple[str, ...], period: str):
    prices = fetch_prices(list(tickers), period=period)
    mu, sigma = annualized_moments(prices)
    return mu, sigma, len(prices)


@st.cache_data(show_spinner=False)
def _cached_classical(tickers: tuple[str, ...], period: str, n_mc: int, rf: float, seed: int = 42):
    mu, sigma, n_days = _cached_moments(tickers, period)
    mu_arr, sigma_arr = mu.values, sigma.values
    mc_df = monte_carlo_simulation(mu_arr, sigma_arr, n_portfolios=n_mc, risk_free_rate=rf, seed=seed)
    frontier_df = efficient_frontier(mu_arr, sigma_arr)
    max_sharpe_w = max_sharpe_portfolio(mu_arr, sigma_arr, rf)
    min_vol_w = min_volatility_portfolio(mu_arr, sigma_arr)
    return mu_arr, sigma_arr, n_days, mc_df, frontier_df, max_sharpe_w, min_vol_w


def _init_state() -> None:
    defaults = {
        "classical": None,
        "qubo_qp": None,
        "qubo_qubits": None,
        "exact_run": None,
        "qaoa_job": None,  # dict: status, thread, run, error, start_time, elapsed
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()

st.title("📈 Markowitz Efficient Frontier + QUBO/QAOA Cardinality-Constrained Portfolio")
st.caption(
    "Classical Modern Portfolio Theory next to its NP-hard combinatorial extension: "
    "\"hold exactly K of these N assets,\" solved both exactly and with QAOA."
)

with st.sidebar:
    st.header("Data")
    tickers_text = st.text_area("Tickers (comma-separated)", value=", ".join(DEFAULT_TICKERS), height=70)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    period = st.selectbox("History window", ["1y", "2y", "3y", "5y"], index=2)
    risk_free_rate = st.number_input("Risk-free rate (annualized)", value=0.0, step=0.005, format="%.3f")
    n_mc = st.slider("Monte Carlo portfolios", 1000, 50000, 20000, step=1000)

    st.divider()
    st.header("Cardinality constraint")
    max_budget = max(1, len(tickers) - 1)
    budget = st.slider("Assets to hold (K)", 1, max_budget, min(5, max_budget))
    risk_factor = st.slider("Risk aversion (q)", 0.0, 2.0, 0.5, step=0.05)

    st.divider()
    st.header("QAOA settings")
    qaoa_reps = st.slider("Circuit depth (reps)", 1, 5, 1)
    qaoa_maxiter = st.slider("COBYLA max iterations", 10, 300, 50, step=10)
    st.warning(
        "Benchmark on this machine: 10 qubits, reps=3, maxiter=200 took **~23 minutes** "
        "and still landed ~0.2 above the exact optimum. Start small (reps=1, maxiter=50) "
        "and increase only if you're willing to wait.",
        icon="⏱️",
    )

    st.divider()
    run_classical = st.button("1. Fetch data & run classical optimization", type="primary", width="stretch")

if len(tickers) < 3:
    st.info("Enter at least 3 tickers in the sidebar to begin.")
    st.stop()

if run_classical:
    with st.spinner(f"Fetching {period} of history for {len(tickers)} tickers and solving the continuous frontier..."):
        try:
            result = _cached_classical(tuple(tickers), period, n_mc, risk_free_rate)
            st.session_state["classical"] = {"tickers": tickers, "result": result}
            st.session_state["exact_run"] = None
            st.session_state["qaoa_job"] = None
        except Exception as e:
            st.error(f"Failed to fetch/optimize: {e}")

classical = st.session_state["classical"]

if classical is None:
    st.info("Set your tickers and click **Fetch data & run classical optimization** in the sidebar to start.")
    st.stop()

if classical["tickers"] != tickers:
    st.warning("Tickers changed since the last run — click **Fetch data & run classical optimization** again to refresh.")

mu_arr, sigma_arr, n_days, mc_df, frontier_df, max_sharpe_w, min_vol_w = classical["result"]
run_tickers = classical["tickers"]
max_sharpe_perf = portfolio_performance(max_sharpe_w, mu_arr, sigma_arr)
min_vol_perf = portfolio_performance(min_vol_w, mu_arr, sigma_arr)

st.subheader("1. Classical continuous Markowitz")
st.caption(f"{n_days} trading days · {len(run_tickers)} tickers · {n_mc:,} simulated portfolios")

c1, c2 = st.columns(2)
with c1:
    st.metric("Max-Sharpe return / vol", f"{max_sharpe_perf[0]:.2%} / {max_sharpe_perf[1]:.2%}")
with c2:
    st.metric("Min-volatility return / vol", f"{min_vol_perf[0]:.2%} / {min_vol_perf[1]:.2%}")

discrete_points: list[tuple[str, float, float]] = []
weights_by_method = {
    "Max Sharpe (continuous)": max_sharpe_w,
    "Min volatility (continuous)": min_vol_w,
}

st.subheader("2. Cardinality-constrained selection (QUBO / Ising)")
st.write(
    f"Adding *\"hold exactly {budget} of these {len(run_tickers)} assets, equal-weighted\"* turns the "
    "continuous QP above into a combinatorial search over "
    f"C({len(run_tickers)}, {budget}) subsets — encoded as a QUBO, mapped to an Ising Hamiltonian."
)

col_build, col_exact, col_qaoa = st.columns(3)

with col_build:
    if st.button("Build QUBO", width="stretch"):
        qp, _ = build_cardinality_qubo(mu_arr, sigma_arr, budget=budget, risk_factor=risk_factor)
        hamiltonian, offset = to_ising(qp)
        st.session_state["qubo_qp"] = qp
        st.session_state["qubo_qubits"] = hamiltonian.num_qubits
        st.session_state["exact_run"] = None
        st.session_state["qaoa_job"] = None

if st.session_state["qubo_qp"] is not None:
    st.caption(f"Ising Hamiltonian ready: **{st.session_state['qubo_qubits']} qubits**.")

with col_exact:
    exact_disabled = st.session_state["qubo_qp"] is None
    if st.button("Solve exact (NumPyMinimumEigensolver)", disabled=exact_disabled, width="stretch"):
        st.session_state["exact_run"] = solve_exact(st.session_state["qubo_qp"])

with col_qaoa:
    qaoa_job = st.session_state["qaoa_job"]
    qaoa_running = qaoa_job is not None and qaoa_job["status"] == "running"
    qaoa_disabled = st.session_state["qubo_qp"] is None or qaoa_running
    if st.button("Run QAOA (background)", disabled=qaoa_disabled, width="stretch"):
        job = {"status": "running", "run": None, "error": None, "start_time": time.time()}

        def _worker(qp=st.session_state["qubo_qp"], reps=qaoa_reps, maxiter=qaoa_maxiter, job=job):
            try:
                job["run"] = solve_qaoa(qp, reps=reps, maxiter=maxiter)
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["error"] = str(e)
                job["status"] = "error"

        thread = threading.Thread(target=_worker, daemon=True)
        job["thread"] = thread
        st.session_state["qaoa_job"] = job
        thread.start()
        st.rerun()


@st.fragment(run_every=2)
def _qaoa_status_fragment():
    job = st.session_state["qaoa_job"]
    if job is None:
        return
    if job["status"] == "running":
        elapsed = time.time() - job["start_time"]
        st.info(f"QAOA running in the background... elapsed {elapsed:.0f}s", icon="⏳")
    elif job["status"] == "error":
        st.error(f"QAOA failed: {job['error']}")
    elif job["status"] == "done":
        elapsed = time.time() - job["start_time"]
        st.success(f"QAOA finished in {elapsed:.0f}s.")


_qaoa_status_fragment()

exact_run = st.session_state["exact_run"]
qaoa_job = st.session_state["qaoa_job"]
qaoa_run = qaoa_job["run"] if (qaoa_job and qaoa_job["status"] == "done") else None

equal_weight = 1.0 / budget
result_cols = st.columns(2)

if exact_run is not None:
    selected = [t for t, bit in zip(run_tickers, exact_run.result.x) if round(bit) == 1]
    w = equal_weight * exact_run.result.x.round()
    ret, vol = portfolio_performance(w, mu_arr, sigma_arr)
    discrete_points.append(("Exact", ret, vol))
    weights_by_method["Exact"] = w
    with result_cols[0]:
        st.markdown("**Exact (NumPyMinimumEigensolver)**")
        st.write(f"objective = `{exact_run.result.fval:.6f}`  ·  {exact_run.wall_time_s:.3f}s")
        st.write(f"selected: {', '.join(selected)}")
        st.write(f"return = {ret:.2%}  ·  vol = {vol:.2%}")

if qaoa_run is not None:
    selected = [t for t, bit in zip(run_tickers, qaoa_run.result.x) if round(bit) == 1]
    w = equal_weight * qaoa_run.result.x.round()
    ret, vol = portfolio_performance(w, mu_arr, sigma_arr)
    discrete_points.append(("QAOA", ret, vol))
    weights_by_method["QAOA"] = w
    with result_cols[1]:
        st.markdown(f"**QAOA (reps={qaoa_reps})**")
        st.write(f"objective = `{qaoa_run.result.fval:.6f}`  ·  {qaoa_run.wall_time_s:.1f}s")
        st.write(f"selected: {', '.join(selected)}")
        st.write(f"return = {ret:.2%}  ·  vol = {vol:.2%}")
        if exact_run is not None:
            gap = qaoa_run.result.fval - exact_run.result.fval
            st.write(f"gap vs. exact: `{gap:+.6f}`")

st.subheader("3. Risk/return comparison")
fig = frontier_figure_plotly(mc_df, frontier_df, max_sharpe_perf, min_vol_perf, discrete_points)
st.plotly_chart(fig, width="stretch")

st.subheader("4. Portfolio weights by method")
wfig = weights_figure_plotly(run_tickers, weights_by_method)
st.plotly_chart(wfig, width="stretch")

with st.expander("Method notes / limitations"):
    st.markdown(
        "- Expected returns and covariance are historical-mean estimates "
        "(annualized sample mean/covariance of log returns) — Markowitz's known "
        "estimation-error sensitivity applies here too.\n"
        "- QAOA runs on Qiskit's `StatevectorSampler` (simulator), not real quantum hardware.\n"
        "- Qubit count equals the number of tickers — stay in the low teens for "
        "interactive use.\n"
        "- The exact solver is brute-force diagonalization; it's only tractable "
        "*because* qubit counts stay small, not a claim it scales."
    )
