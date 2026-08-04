"""Classical Modern Portfolio Theory: Monte Carlo simulation + Markowitz
efficient frontier via constrained quadratic optimization (scipy.optimize).

This is the continuous-weight formulation: weights are real numbers in
[0, 1] that sum to 1 (long-only, fully invested). It has no cardinality
constraint on the number of assets held -- that extension is handled
separately in qubo_portfolio.py / quantum_solver.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(
    weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> tuple[float, float]:
    """Annualized expected return and volatility for a weight vector."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ sigma @ weights))
    return ret, vol


def monte_carlo_simulation(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_portfolios: int = 20_000,
    risk_free_rate: float = 0.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Simulate random long-only, fully-invested portfolios (Dirichlet
    sampling over the simplex) and score each by return, volatility, and
    Sharpe ratio. Used to visualize the feasible region and approximate
    the efficient frontier before refining it with exact optimization.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights))
    sharpe = (rets - risk_free_rate) / vols

    df = pd.DataFrame(weights, columns=[f"w_{i}" for i in range(n_assets)])
    df["return"] = rets
    df["volatility"] = vols
    df["sharpe"] = sharpe
    return df


def _long_only_constraints(n_assets: int) -> tuple[tuple, tuple]:
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    return bounds, constraints


def min_volatility_portfolio(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Weights of the global minimum-variance portfolio (SLSQP)."""
    n_assets = len(mu)
    bounds, constraints = _long_only_constraints(n_assets)
    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        lambda w: w @ sigma @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Min-volatility optimization failed: {result.message}")
    return result.x


def max_sharpe_portfolio(
    mu: np.ndarray, sigma: np.ndarray, risk_free_rate: float = 0.0
) -> np.ndarray:
    """Weights of the portfolio maximizing the Sharpe ratio (SLSQP)."""
    n_assets = len(mu)
    bounds, constraints = _long_only_constraints(n_assets)
    x0 = np.ones(n_assets) / n_assets

    def neg_sharpe(w: np.ndarray) -> float:
        ret, vol = portfolio_performance(w, mu, sigma)
        return -(ret - risk_free_rate) / vol

    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")
    return result.x


def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, n_points: int = 50
) -> pd.DataFrame:
    """Trace the efficient frontier by minimizing volatility for a sweep
    of target returns spanning the min-vol portfolio's return up to the
    highest single-asset expected return.
    """
    n_assets = len(mu)
    bounds, base_constraints = _long_only_constraints(n_assets)

    min_vol_w = min_volatility_portfolio(mu, sigma)
    min_ret, _ = portfolio_performance(min_vol_w, mu, sigma)
    max_ret = float(np.max(mu))

    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_vols = []
    frontier_weights = []

    x0 = np.ones(n_assets) / n_assets
    for target in target_returns:
        constraints = base_constraints + (
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        )
        result = minimize(
            lambda w: w @ sigma @ w,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            frontier_vols.append(np.sqrt(result.fun))
            frontier_weights.append(result.x)
        else:
            frontier_vols.append(np.nan)
            frontier_weights.append(np.full(n_assets, np.nan))

    df = pd.DataFrame(frontier_weights, columns=[f"w_{i}" for i in range(n_assets)])
    df["return"] = target_returns
    df["volatility"] = frontier_vols
    return df.dropna()
