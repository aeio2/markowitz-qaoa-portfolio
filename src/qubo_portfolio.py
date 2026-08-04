"""Cardinality-constrained Markowitz portfolio -> QUBO -> Ising Hamiltonian.

Adding a "choose exactly K of N assets" cardinality constraint turns the
continuous mean-variance problem (solved in classical_mpt.py with SLSQP)
into an NP-hard combinatorial selection problem:

    minimize   q * x^T Sigma x  -  mu^T x
    subject to sum(x_i) == B          (budget / cardinality: hold exactly B assets)
               x_i in {0, 1}          (equal-weighted: hold or don't hold asset i)

This is the standard formulation used by Qiskit Finance's
``PortfolioOptimization`` application, which builds the QuadraticProgram
for us and (via ``to_ising``) maps it onto an Ising Hamiltonian suitable
for QAOA / VQE.

Version note: PortfolioOptimization lives in
``qiskit_finance.applications`` in the installed version (qiskit-finance
0.4.1, now maintained under the qiskit-community org). Older tutorials
import it from ``qiskit_optimization.applications`` -- that path does not
exist in this environment, so don't assume tutorial code matches the
installed version without checking.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_finance.applications import PortfolioOptimization
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_cardinality_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    budget: int,
    risk_factor: float = 0.5,
) -> tuple[QuadraticProgram, PortfolioOptimization]:
    """Build the cardinality-constrained mean-variance QuadraticProgram.

    Args:
        mu: annualized expected returns, shape (n,).
        sigma: annualized covariance matrix, shape (n, n).
        budget: number of assets to select (the "K" in "choose K of N").
        risk_factor: risk-aversion coefficient q in q*x^T Sigma x - mu^T x.
    """
    n = len(mu)
    if not (1 <= budget <= n):
        raise ValueError(f"budget must be between 1 and {n}, got {budget}")

    portfolio = PortfolioOptimization(
        expected_returns=mu,
        covariances=sigma,
        risk_factor=risk_factor,
        budget=budget,
    )
    qp = portfolio.to_quadratic_program()
    return qp, portfolio


def to_qubo(qp: QuadraticProgram) -> QuadraticProgram:
    """Fold the budget/cardinality equality constraint into a quadratic
    penalty term, producing the unconstrained QUBO the Ising mapping
    requires (``to_ising`` refuses any QuadraticProgram that still has
    explicit constraints).
    """
    return QuadraticProgramToQubo().convert(qp)


def to_ising(qp: QuadraticProgram) -> tuple[SparsePauliOp, float]:
    """Convert a constraint-free QUBO QuadraticProgram (see ``to_qubo``)
    to an Ising Hamiltonian.

    Returns the Pauli-Z Hamiltonian and the constant offset such that
    H = sum_{ij} J_ij Z_i Z_j + sum_i h_i Z_i + offset reproduces the
    original QUBO objective on {0,1} variables mapped to {+1,-1} spins.
    """
    qubo = to_qubo(qp)
    hamiltonian, offset = qubo.to_ising()
    return hamiltonian, offset


def selection_from_result(x: np.ndarray, tickers: list[str]) -> list[str]:
    """Map a binary solution vector back to the selected tickers."""
    return [ticker for ticker, bit in zip(tickers, x) if round(bit) == 1]
