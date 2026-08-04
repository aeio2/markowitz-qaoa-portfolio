"""End-to-end pipeline:

  1. Fetch prices for N tickers (Yahoo Finance) and estimate annualized
     mu / Sigma.
  2. Classical continuous Markowitz: Monte Carlo cloud + SLSQP efficient
     frontier + max-Sharpe / min-vol portfolios.
  3. Add a cardinality constraint ("hold exactly K of N assets") --
     this turns the continuous QP into an NP-hard combinatorial
     selection problem. Encode it as a QUBO via Qiskit Finance's
     PortfolioOptimization application and map it to an Ising
     Hamiltonian.
  4. Solve that Hamiltonian's ground state two ways and compare:
     NumPyMinimumEigensolver (exact) vs QAOA (statevector simulator).
  5. Plot everything on one risk/return chart.

Usage:
    python3 main.py --tickers AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V PG \
        --period 3y --budget 5 --risk-factor 0.5 --qaoa-reps 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from quantum_solver import check_versions, solve_exact, solve_qaoa, summarize
from plotting import plot_frontier

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "PG",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols (10-12 recommended -> 10-12 qubits)")
    p.add_argument("--period", default="3y", help="yfinance history window, e.g. 2y, 3y, 5y")
    p.add_argument("--budget", type=int, default=5, help="Number of assets to hold (cardinality K)")
    p.add_argument("--risk-factor", type=float, default=0.5, help="Risk-aversion coefficient q in q*x^T Sigma x - mu^T x")
    p.add_argument("--risk-free-rate", type=float, default=0.0, help="Annualized risk-free rate for Sharpe ratio")
    p.add_argument("--n-mc", type=int, default=20_000, help="Number of Monte Carlo portfolios to simulate")
    p.add_argument("--qaoa-reps", type=int, default=3, help="QAOA circuit depth (p)")
    p.add_argument("--qaoa-maxiter", type=int, default=200, help="COBYLA max iterations for QAOA")
    p.add_argument("--output", default="outputs/efficient_frontier.png", help="Path to save the comparison plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    version_warning = check_versions()
    if version_warning:
        print(f"WARNING: {version_warning}")
    tickers = args.tickers
    n = len(tickers)
    if args.budget >= n:
        print(f"Warning: budget ({args.budget}) >= number of tickers ({n}); "
              "cardinality constraint will be trivial (selects everyone).")

    print(f"[1/4] Fetching {args.period} of price history for {n} tickers: {tickers}")
    prices = fetch_prices(tickers, period=args.period)
    mu, sigma = annualized_moments(prices)
    mu_arr, sigma_arr = mu.values, sigma.values
    print(f"      -> {len(prices)} trading days after alignment.")

    print("[2/4] Classical continuous Markowitz optimization (scipy.optimize / SLSQP)")
    mc_df = monte_carlo_simulation(mu_arr, sigma_arr, n_portfolios=args.n_mc, risk_free_rate=args.risk_free_rate)
    frontier_df = efficient_frontier(mu_arr, sigma_arr)
    max_sharpe_w = max_sharpe_portfolio(mu_arr, sigma_arr, args.risk_free_rate)
    min_vol_w = min_volatility_portfolio(mu_arr, sigma_arr)
    max_sharpe_perf = portfolio_performance(max_sharpe_w, mu_arr, sigma_arr)
    min_vol_perf = portfolio_performance(min_vol_w, mu_arr, sigma_arr)
    print(f"      Max-Sharpe portfolio:   return={max_sharpe_perf[0]:.4f}  vol={max_sharpe_perf[1]:.4f}")
    for t, w in zip(tickers, max_sharpe_w):
        if w > 1e-3:
            print(f"        {t}: {w:.3f}")
    print(f"      Min-volatility portfolio: return={min_vol_perf[0]:.4f}  vol={min_vol_perf[1]:.4f}")

    print(f"[3/4] Building cardinality-constrained QUBO (choose {args.budget} of {n} assets, equal-weighted)")
    qp, _ = build_cardinality_qubo(mu_arr, sigma_arr, budget=args.budget, risk_factor=args.risk_factor)
    hamiltonian, offset = to_ising(qp)
    print(f"      Ising Hamiltonian: {hamiltonian.num_qubits} qubits, offset={offset:.6f}")

    print("[4/4] Solving ground state: NumPyMinimumEigensolver (exact) vs QAOA (statevector simulator)")
    exact_run = solve_exact(qp)
    qaoa_run = solve_qaoa(qp, reps=args.qaoa_reps, maxiter=args.qaoa_maxiter)
    print(summarize([exact_run, qaoa_run], tickers))

    discrete_points = []
    equal_weight = 1.0 / args.budget
    for run in (exact_run, qaoa_run):
        w = equal_weight * (run.result.x.round().astype(float))
        ret, vol = portfolio_performance(w, mu_arr, sigma_arr)
        discrete_points.append((run.label.split(" (")[0], ret, vol))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_frontier(
        mc_df, frontier_df,
        max_sharpe_point=max_sharpe_perf,
        min_vol_point=min_vol_perf,
        discrete_points=discrete_points,
        output_path=args.output,
    )
    print(f"\nSaved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
