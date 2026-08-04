"""Solve the cardinality-constrained portfolio QUBO two ways and compare:

1. NumPyMinimumEigensolver -- exact diagonalization of the Ising
   Hamiltonian (brute-force ground state). Only tractable because our
   qubit counts (= number of candidate assets) stay in the 10-12 range;
   this is the ground-truth reference, not something that scales.
2. QAOA -- variational quantum eigensolver run on a statevector
   simulator, used as the NISQ-era heuristic for the same Hamiltonian.

Both are wrapped as MinimumEigenOptimizers so they consume/produce the
same QuadraticProgram interface as the classical solvers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import qiskit
import qiskit_algorithms
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    OptimizationResult,
)

# QAOA's sampler contract changed between qiskit-algorithms releases (V1 vs
# V2 primitives) in a way that fails deep inside COBYLA's callback with an
# opaque TypeError if the installed stack drifts from what this project was
# built against -- this happened once already in development (some other
# pip operation silently downgraded qiskit 2.5.1 -> 1.2.4 and
# qiskit-algorithms 0.4.0 -> 0.3.1 mid-session). Fail fast with a clear
# message instead.
_EXPECTED_QISKIT = "2.5.1"
_EXPECTED_QISKIT_ALGORITHMS = "0.4.0"


def check_versions() -> str | None:
    """Return a warning string if the installed qiskit stack doesn't match
    what this project pins in requirements.txt, else None.
    """
    if qiskit.__version__ != _EXPECTED_QISKIT or qiskit_algorithms.__version__ != _EXPECTED_QISKIT_ALGORITHMS:
        return (
            f"Installed qiskit={qiskit.__version__}, qiskit-algorithms={qiskit_algorithms.__version__}, "
            f"but this project was built against qiskit=={_EXPECTED_QISKIT}, "
            f"qiskit-algorithms=={_EXPECTED_QISKIT_ALGORITHMS}. QAOA may fail with an opaque error "
            f"if these have drifted apart (V1/V2 sampler incompatibility). "
            f"Run: pip install -r requirements.txt"
        )
    return None


@dataclass
class SolverRun:
    label: str
    result: OptimizationResult
    wall_time_s: float


def solve_exact(qp: QuadraticProgram) -> SolverRun:
    """Ground-state solution via exact diagonalization (reference)."""
    t0 = time.perf_counter()
    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = solver.solve(qp)
    return SolverRun("Exact (NumPyMinimumEigensolver)", result, time.perf_counter() - t0)


def solve_qaoa(
    qp: QuadraticProgram,
    reps: int = 3,
    maxiter: int = 200,
    seed: int | None = 42,
) -> SolverRun:
    """Ground-state solution via QAOA on a statevector simulator."""
    t0 = time.perf_counter()
    sampler = StatevectorSampler(seed=seed) if seed is not None else StatevectorSampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps)
    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)
    return SolverRun(f"QAOA (reps={reps})", result, time.perf_counter() - t0)


def summarize(runs: list[SolverRun], tickers: list[str]) -> str:
    lines = []
    exact_fval = next((r.result.fval for r in runs if "Exact" in r.label), None)
    for run in runs:
        selected = [t for t, bit in zip(tickers, run.result.x) if round(bit) == 1]
        gap = ""
        if exact_fval is not None and "Exact" not in run.label:
            gap = f"  |  gap vs exact = {run.result.fval - exact_fval:+.6f}"
        lines.append(
            f"[{run.label}] objective={run.result.fval:.6f}  "
            f"time={run.wall_time_s:.2f}s  selected={selected}{gap}"
        )
    return "\n".join(lines)
