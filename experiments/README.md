# experiments/ — the QUBO/QAOA route, measured and rejected

Nothing in here produces any number in the main README. It is kept
because *why* it isn't used is itself a result.

The selection problem in this project is:

```
minimize   q * xᵀΣx − μᵀx
subject to Σxᵢ = B,  xᵢ ∈ {0,1}
```

There are two ways to solve it. This directory holds the one that lost.

## The two routes

| | QUBO → Ising → QAOA (here) | Enumerate feasible subsets (`src/portfolio_selection.py`) |
|---|---|---|
| Cardinality constraint | folded into the objective as a quadratic penalty; feasibility must be checked afterwards | hard constraint, cannot return an infeasible portfolio |
| Search space (N=20, B=10) | 2²⁰ = 1,048,576 binary strings | C(20,10) = 184,756 feasible subsets |
| Solve time at that size | exact eigensolver ~0.25s; **QAOA >15 min and still running** | ~0.2s |
| Optimality | QAOA is a heuristic with no guarantee | exact |
| Dependencies | qiskit, qiskit-aer, qiskit-algorithms, qiskit-finance, qiskit-optimization | numpy |

## The measurements that settled it

**At 10 tickers (budget=5, risk_factor=0.5):**

```
Exact (NumPyMinimumEigensolver): objective=-0.940630  time=0.01s   selected=['GOOGL','AMZN','NVDA','JPM','V']
QAOA (reps=3, maxiter=200):      objective=-0.736344  time=1382s   selected=['AAPL','MSFT','NVDA','JPM','V']  gap=+0.204286
QAOA (reps=1, maxiter=50):       objective=-0.934755  time=198s    gap=+0.005875
```

Two things worth internalizing. First, the exact solver is both faster
(milliseconds vs. minutes) *and* guaranteed optimal at this size.
Second, QAOA's quality is not simply "more depth is better" — the
shallower `reps=1` run landed **closer** to optimal than the deeper
`reps=3` run, because COBYLA got stuck in a worse local optimum at
higher dimensionality.

**At 20 tickers (the size the Taiwan backtest actually uses):**

A single QAOA run at `reps=1, maxiter=20` — a *smaller* budget than the
10-ticker default above — ran for over 15 minutes without finishing, and
in a separate attempt was still running after ~59 minutes of CPU time.
Exact solve at the same size: 0.25s.

That is 2ⁿ statevector-simulation cost made concrete: doubling the
variable count did not double the runtime, it changed it by orders of
magnitude. Calibrated on the two real measurements above, per-iteration
cost scales roughly as `4s × 1.51^(n−10)`.

The Taiwan backtest has ~23 rebalance dates. At exact-solve speed that is
about 6 seconds of solving. At QAOA speed it is, conservatively, **four
hours or more** — for an answer that is not guaranteed to be optimal and
that measurably was not, at the one size where both could be run.

## Conclusion

QAOA was implemented correctly, benchmarked honestly, and then not used.
The main pipeline solves the constrained problem directly. Since it never
constructs an unconstrained reformulation, it is no longer accurate to
call it a QUBO at all — hence `portfolio_selection.py` rather than
`qubo_portfolio.py`.

The value of the exercise was in building the QUBO → Ising → QAOA
pipeline correctly and being able to *measure* the gap, not in QAOA
winning. It did not win.

## Files

```
qubo_portfolio.py     Cardinality-constrained problem -> QuadraticProgram -> QUBO -> Ising Hamiltonian
quantum_solver.py     NumPyMinimumEigensolver (exact) and QAOA solvers + qiskit version guard
qaoa_comparison.py    CLI: efficient frontier + exact vs QAOA on the same Hamiltonian (10-12 tickers)
app_qaoa_demo.py      Streamlit demo of the above, with background QAOA runs
```

## Running these

Needs the quantum stack, which the main project does not:

```bash
.venv/bin/pip install -r ../requirements.txt -r ../requirements-experiments.txt
.venv/bin/python3 qaoa_comparison.py --tickers AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V PG --budget 5
.venv/bin/streamlit run app_qaoa_demo.py
```

**Version note:** `qiskit-finance` 0.4.x (now under the `qiskit-community`
org) removed several deprecated import paths — `PortfolioOptimization`
lives in `qiskit_finance.applications`, **not**
`qiskit_optimization.applications` where older tutorials import it from.
`quantum_solver.check_versions()` warns on drift instead of failing with
an opaque `TypeError` deep inside COBYLA's callback. Keep these in a
`.venv`; during development something on this machine twice silently
downgraded the *global* qiskit install mid-session.
