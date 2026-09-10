"""Cardinality-constrained mean-variance selection, solved exactly by
enumeration over the feasible set.

    minimize   q * x^T Sigma x  -  mu^T x
    subject to sum(x) == B,  x_i in {0, 1}

This is the same objective the QUBO/Ising formulation encodes -- see
``experiments/qaoa_comparison.py`` for that formulation and for the QAOA
benchmark -- but solved directly over the C(n, B) feasible subsets rather
than over all 2^n binary strings.

Why enumerate the feasible set instead of routing through a QUBO:

  - The cardinality constraint stays a *hard* constraint. The QUBO route
    folds it into the objective as a quadratic penalty, which means
    choosing a penalty coefficient and then checking afterwards that the
    returned solution is actually feasible. Enumeration cannot return an
    infeasible portfolio.
  - It is less work, not more: C(20, 10) = 184,756 feasible subsets vs.
    2^20 = 1,048,576 binary strings for the same problem.
  - Forcing specific names into the portfolio ("I want 2330.TW in there
    regardless") is just a smaller enumeration, instead of a variable
    substitution on a QuadraticProgram followed by reconstructing the
    full solution vector.
  - It removes qiskit from the code path that produces every number in
    the backtest and the portfolio tool.

This does not scale, and is not meant to. The problem is NP-hard; this is
tractable only because the candidate pool is deliberately pre-filtered to
~20 names first (see ``tw_universe.py``). ``max_evaluations`` guards
against silently starting a run that will not finish.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations, islice

import numpy as np

# Refuse to start an enumeration bigger than this. C(20,10) = 184,756 and
# C(24,12) = 2,704,156, so this permits a somewhat larger pool than the
# default 20 while still failing fast on anything genuinely intractable.
DEFAULT_MAX_EVALUATIONS = 5_000_000


@dataclass(frozen=True)
class CardinalityProblem:
    """A cardinality-constrained mean-variance problem instance."""

    mu: np.ndarray          # annualized expected returns, shape (n,)
    sigma: np.ndarray       # annualized covariance matrix, shape (n, n)
    budget: int             # the "K" in "choose K of N"
    risk_factor: float      # q in  q * x^T Sigma x - mu^T x

    @property
    def n_assets(self) -> int:
        return len(self.mu)


@dataclass
class SolverRun:
    """Result of one solve, plus how long it took."""

    label: str
    x: np.ndarray           # binary selection vector, shape (n,)
    fval: float             # objective value at x
    wall_time_s: float
    n_evaluated: int = 0    # feasible subsets actually scored


def build_cardinality_problem(
    mu: np.ndarray,
    sigma: np.ndarray,
    budget: int,
    risk_factor: float = 0.5,
) -> CardinalityProblem:
    """Validate inputs and package them as a CardinalityProblem.

    Args:
        mu: annualized expected returns, shape (n,).
        sigma: annualized covariance matrix, shape (n, n).
        budget: number of assets to select (the "K" in "choose K of N").
        risk_factor: risk-aversion coefficient q in q*x^T Sigma x - mu^T x.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)

    if sigma.shape != (n, n):
        raise ValueError(f"sigma must be ({n}, {n}) to match mu, got {sigma.shape}")
    if not (1 <= budget <= n):
        raise ValueError(f"budget must be between 1 and {n}, got {budget}")

    return CardinalityProblem(mu=mu, sigma=sigma, budget=int(budget), risk_factor=float(risk_factor))


def objective(x: np.ndarray, problem: CardinalityProblem) -> float:
    """Evaluate q * x^T Sigma x - mu^T x for a single selection vector."""
    x = np.asarray(x, dtype=float).ravel()
    return float(problem.risk_factor * x @ problem.sigma @ x - problem.mu @ x)


def solve_exact(
    problem: CardinalityProblem,
    force_include: list[int] | None = None,
    chunk_size: int = 100_000,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
) -> SolverRun:
    """Exact solution by enumerating every feasible subset.

    Scores subsets in vectorized chunks so peak memory stays flat
    (chunk_size x n floats) regardless of how many subsets there are.

    Args:
        problem: the instance to solve.
        force_include: asset indices that must appear in the portfolio.
            Reduces the enumeration to C(n - len(forced), B - len(forced)).
        chunk_size: subsets scored per vectorized block.
        max_evaluations: refuse to start if the feasible set is larger.
    """
    t0 = time.perf_counter()

    mu, sigma = problem.mu, problem.sigma
    n, budget, q = problem.n_assets, problem.budget, problem.risk_factor

    forced = sorted(set(force_include or []))
    for i in forced:
        if not (0 <= i < n):
            raise ValueError(f"force_include index {i} out of range for {n} assets")
    if len(forced) > budget:
        raise ValueError(f"force_include has {len(forced)} assets but budget is {budget}")

    free = [i for i in range(n) if i not in forced]
    k = budget - len(forced)

    total = math.comb(len(free), k)
    if total > max_evaluations:
        raise ValueError(
            f"Enumerating C({len(free)}, {k}) = {total:,} subsets exceeds "
            f"max_evaluations={max_evaluations:,}. Shrink the candidate pool "
            f"(see tw_universe.py) or raise the limit deliberately."
        )

    best_x: np.ndarray | None = None
    best_val = np.inf
    subsets = combinations(free, k)

    while True:
        block = list(islice(subsets, chunk_size))
        if not block:
            break

        rows = len(block)
        selection = np.zeros((rows, n), dtype=float)
        if k:
            cols = np.asarray(block, dtype=int)
            selection[np.repeat(np.arange(rows), k), cols.ravel()] = 1.0
        if forced:
            selection[:, forced] = 1.0

        # Per-row  q * s Sigma s^T - mu s  for every subset in the block.
        risk = np.einsum("ij,jk,ik->i", selection, sigma, selection)
        values = q * risk - selection @ mu

        j = int(np.argmin(values))
        if values[j] < best_val:
            best_val = float(values[j])
            best_x = selection[j].copy()

    if best_x is None:  # pragma: no cover - guarded by build_cardinality_problem
        raise RuntimeError("No feasible portfolio found")

    return SolverRun(
        label=f"Exact (enumerated C({len(free)}, {k}) = {total:,})",
        x=best_x,
        fval=best_val,
        wall_time_s=time.perf_counter() - t0,
        n_evaluated=total,
    )


def selection_from_result(x: np.ndarray, tickers: list[str]) -> list[str]:
    """Map a binary solution vector back to the selected tickers."""
    return [ticker for ticker, bit in zip(tickers, x) if round(bit) == 1]
