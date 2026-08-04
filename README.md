# Markowitz Efficient Frontier + QUBO/QAOA Cardinality-Constrained Portfolio

Two portfolio optimization problems solved on the same real market data,
side by side:

1. **Classical continuous Markowitz (Modern Portfolio Theory).**
   Monte Carlo simulation of random long-only portfolios plus an exact
   efficient frontier traced with `scipy.optimize` (SLSQP). Standard
   quadratic-programming MPT — no restriction on how many assets you hold.

2. **Cardinality-constrained selection (NP-hard combinatorial extension).**
   Add the real-world constraint "hold exactly K of these N assets" and
   the problem stops being a continuous QP and becomes a combinatorial
   search over `C(N, K)` subsets. It's encoded as a QUBO (Quadratic
   Unconstrained Binary Optimization), mapped to an Ising Hamiltonian,
   and its ground state is found two ways for comparison:
   - `NumPyMinimumEigensolver` — exact diagonalization (ground truth,
     only tractable because N stays around 10-12 qubits).
   - `QAOA` — the variational quantum heuristic, run on Qiskit's
     statevector simulator.

The two are plotted together: the constrained (K-of-N) solutions sit on
or inside the unconstrained efficient frontier, showing the price of
the cardinality constraint in risk/return terms.

## Why this problem, and not a synthetic one

Cardinality-constrained mean-variance is the most-cited "quantum
finance" example precisely because it's the smallest natural step from
classical MPT into NP-hard territory: swap "how much of each asset" for
"which K assets," and the feasible set changes from a simplex to a
combinatorial one. At 10-12 assets it also fits on a laptop simulator,
so the QAOA result can be checked directly against the exact
ground state rather than taken on faith.

## Setup

Use an isolated virtual environment — see the version-drift note below for
why this matters more than usual here:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then run everything through `.venv/bin/python3` / `.venv/bin/streamlit`
(or `source .venv/bin/activate` first).

**Version note:** `qiskit-finance` is now community-maintained under the
`qiskit-community` org, and 0.4.x removed several previously-deprecated
import paths. In this environment, `PortfolioOptimization` lives in
`qiskit_finance.applications` — **not**
`qiskit_optimization.applications`, which is where a lot of older
tutorials import it from. If you upgrade any of the `qiskit-*` packages
independently, re-check that import before assuming old code/tutorials
still apply.

This isn't hypothetical: mid-development, something on this machine
repeatedly (twice) silently downgraded the global `qiskit` 2.5.1 → 1.2.4
and `qiskit-algorithms` 0.4.0 → 0.3.1 install between my commands, which
changes QAOA's expected sampler type (V2 → V1) and breaks it with an
opaque `TypeError` deep inside COBYLA's callback. Reinstalling the pinned
versions globally didn't stick — something else on this system keeps
resetting global site-packages. **The `.venv` above exists specifically
to be immune to that**; don't skip it in favor of a bare `pip install`
into the system Python. `src/quantum_solver.py`'s `check_versions()`
still checks installed-vs-pinned versions and surfaces a clear warning
(in both `main.py` and `app.py`) instead of that opaque error, as a
belt-and-suspenders check.

## Usage

### Web app (Streamlit)

```bash
.venv/bin/streamlit run app.py
```

Opens an interactive dashboard: set tickers/period/budget/risk settings
in the sidebar, run the classical optimization, then build the QUBO and
solve it exactly and/or with QAOA. The exact solver is instant; QAOA runs
in a background thread (with a live elapsed-time indicator) so the rest
of the UI stays usable while it grinds through the optimizer loop — see
the runtime warning in the sidebar before cranking up `reps`/`maxiter`.

### CLI

```bash
.venv/bin/python3 main.py \
  --tickers AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V PG \
  --period 3y \
  --budget 5 \
  --risk-factor 0.5 \
  --qaoa-reps 3
```

- `--tickers`: 10-12 recommended (maps 1:1 to qubits in the QAOA circuit).
- `--budget`: K, the number of assets to hold (cardinality constraint).
- `--risk-factor`: risk-aversion coefficient `q` in `q * x^T Sigma x - mu^T x`.
- `--qaoa-reps`: QAOA circuit depth (p). Higher = better approximation,
  slower classical optimizer loop.

Output: console summary of the continuous max-Sharpe / min-vol
portfolios, the exact vs. QAOA cardinality-constrained selections and
their objective gap, plus a saved PNG (`outputs/efficient_frontier.png`)
overlaying all of them on one risk/return chart.

## Project layout

```
src/
  data.py             fetch prices (yfinance), compute annualized mu / Sigma
  classical_mpt.py    Monte Carlo simulation + SLSQP efficient frontier
  qubo_portfolio.py   cardinality-constrained QUBO (qiskit-finance PortfolioOptimization)
  quantum_solver.py   exact (NumPyMinimumEigensolver) vs QAOA solvers, version guard
  plotting.py         matplotlib chart (CLI) + Plotly charts (web app)
main.py               CLI orchestrating the full pipeline
app.py                Streamlit web app (interactive, background QAOA)
```

## Math reference

Continuous MPT (long-only, fully invested):

```
minimize   w^T Sigma w
subject to sum(w) = 1,  w_i >= 0
```

swept across target returns to trace the frontier, plus the closed-form
Sharpe-maximizing and variance-minimizing special cases.

Cardinality-constrained selection (equal-weighted, binary):

```
minimize   q * x^T Sigma x  -  mu^T x
subject to sum(x_i) = K,  x_i in {0, 1}
```

which is exactly `qiskit_finance.applications.PortfolioOptimization`'s
`to_quadratic_program()` output, converted to an Ising Hamiltonian via
`QuadraticProgram.to_ising()` for QAOA/VQE.

## Observed results (10 tickers, budget=5, risk_factor=0.5)

```
Exact (NumPyMinimumEigensolver): objective=-0.940630  time=0.01s   selected=['GOOGL', 'AMZN', 'NVDA', 'JPM', 'V']
QAOA (reps=3, maxiter=200):      objective=-0.736344  time=1382s   selected=['AAPL', 'MSFT', 'NVDA', 'JPM', 'V']  gap=+0.204286
QAOA (reps=1, maxiter=50):       objective=-0.934755  time=198s    gap=+0.005875
```

Worth internalizing: at this problem size the exact solver is both faster
(milliseconds vs. minutes on a laptop) and guaranteed optimal, while QAOA
is a heuristic whose quality depends on depth/iteration budget in a way
that isn't simply "more is better" -- the shallower reps=1 run actually
landed *closer* to optimal than the deeper reps=3 run here, because
COBYLA got stuck in a worse local optimum at higher dimensionality. This
is the honest current state of NISQ-era QAOA on classically-simulable
problem sizes: it is not yet reliably competitive with brute-force
diagonalization, and the value of the exercise is in building the QUBO ->
Ising -> QAOA pipeline correctly and being able to *measure* the gap and
its variance, not in QAOA winning. The web app's sidebar defaults to
reps=1/maxiter=50 for this reason -- a reasonable few-minutes-not-tens-
of-minutes starting point, tunable from there.

## Known limitations

- Simulator only — QAOA runs on `StatevectorSampler`, not real quantum
  hardware. That's a deliberate scope choice (this is meant to run
  end-to-end on a laptop), not a claim about NISQ hardware performance.
- Qubit count = number of candidate tickers. State-vector simulation is
  fine through the low teens; don't push `--tickers` much past 12-14
  without switching to a shot-based sampler or fewer reps.
- Expected returns and covariance are historical-mean estimates
  (annualized sample mean/covariance of log returns) — the well-known
  estimation-error sensitivity of Markowitz applies here too.
