"""The 'default portfolio' state for the live portfolio tool.

Unlike tw_backtest.py (a fixed historical window, 2021-2026), this
module answers "what should I hold right now, and how has holding it
done since I decided that" -- a portfolio is computed once, persisted
with its inception date and inception prices, and every subsequent tool
open computes live performance from inception to *now* (whenever now
is), not a hardcoded window. Recomputing replaces the stored portfolio
and resets the inception point -- that's a deliberate action the user
takes (a button), not something that happens silently on every load.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qubo_portfolio import build_cardinality_qubo
from quantum_solver import solve_exact
from tw_data import CACHE_DIR, add_custom_ticker, build_candidate_prices

STATE_PATH = CACHE_DIR / "default_portfolio.json"
LOT_SIZE = 1000  # Taiwan board lot ("一張") = 1000 shares


@dataclass
class PortfolioState:
    computed_at: str
    inception_date: str
    pool_size: int
    budget: int
    risk_factor: float
    lookback_years: float
    tickers: list[str]
    names: dict[str, str]
    inception_prices: dict[str, float]
    objective: float
    candidate_pool: list[dict] = field(default_factory=list)
    # The exact variance-covariance matrix (annualized) the QUBO optimized
    # against, plus the ticker order it's indexed by -- kept so ex-ante
    # portfolio variance can be recomputed for whatever subset of the pool
    # the user is currently holding (see ex_ante_volatility below), without
    # re-fetching data or re-estimating Sigma from scratch every rerun.
    sigma_tickers: list[str] = field(default_factory=list)
    sigma_annualized: list[list[float]] = field(default_factory=list)
    # mu (annualized expected return per candidate, same order as
    # sigma_tickers) -- the other half of what the QUBO optimized
    # (q*wSw - mu*w). Stored so per-candidate return/vol/Sharpe can be
    # shown while editing holdings, from the exact numbers that drove
    # selection, not a fresh estimate.
    mu_annualized: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "PortfolioState":
        return cls(**json.loads(s))


def load_state() -> PortfolioState | None:
    if not STATE_PATH.exists():
        return None
    return PortfolioState.from_json(STATE_PATH.read_text())


def save_state(state: PortfolioState) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(state.to_json())


def compute_default_portfolio(
    pool_size: int = 20,
    budget: int = 10,
    risk_factor: float = 0.5,
    lookback_years: float = 2.0,
    refresh: bool = False,
    extra_tickers: list[str] | None = None,
    force_include: list[str] | None = None,
) -> PortfolioState:
    """Fetch the current TWSE/TPEx top-`pool_size` candidate pool, solve
    the cardinality-constrained QUBO (choose `budget` of `pool_size`)
    using trailing `lookback_years` of returns ending today, and return
    the resulting portfolio as a fresh PortfolioState (inception = today).

    extra_tickers: user-supplied tickers to fold into the candidate pool
    before optimizing (e.g. a stock they specifically want considered).
    force_include: tickers that MUST be in the final selection --
    implemented by fixing those QUBO variables to 1 before solving.
    """
    today = date.today()
    fetch_start = (pd.Timestamp(today) - pd.Timedelta(days=int(lookback_years * 365.25) + 30)).date().isoformat()

    prices, pool = build_candidate_prices(pool_size, fetch_start, refresh)
    for t in (extra_tickers or []):
        if t not in prices.columns:
            prices, pool = add_custom_ticker(prices, pool, t, fetch_start)

    window = prices.loc[prices.index > pd.Timestamp(today) - pd.Timedelta(days=int(lookback_years * 365.25))]
    log_ret = np.log(window / window.shift(1)).dropna()
    mu = (log_ret.mean() * 252).values
    sigma = (log_ret.cov() * 252).values
    tickers_all = list(prices.columns)

    qp, _ = build_cardinality_qubo(mu, sigma, budget=budget, risk_factor=risk_factor)
    if force_include:
        fixed = {t: 1 for t in force_include if t in tickers_all}
        if fixed:
            qp = qp.substitute_variables({f"x_{tickers_all.index(t)}": 1 for t in fixed})

    run = solve_exact(qp)
    x = run.result.x if not force_include else _reconstruct_full_solution(run, tickers_all, force_include)
    selected = [t for t, bit in zip(tickers_all, x) if round(bit) == 1]

    inception_prices = {t: float(prices[t].iloc[-1]) for t in selected}
    names = {row["yf_ticker"]: row["name"] for _, row in pool.iterrows() if row["yf_ticker"] in selected}
    for t in selected:
        names.setdefault(t, t)

    state = PortfolioState(
        computed_at=datetime.now().isoformat(timespec="seconds"),
        inception_date=today.isoformat(),
        pool_size=pool_size, budget=budget, risk_factor=risk_factor, lookback_years=lookback_years,
        tickers=selected, names=names, inception_prices=inception_prices,
        objective=float(run.result.fval),
        candidate_pool=pool.to_dict(orient="records"),
        sigma_tickers=tickers_all,
        sigma_annualized=sigma.tolist(),
        mu_annualized=mu.tolist(),
    )
    save_state(state)
    return state


def ex_ante_volatility(state: PortfolioState, held: list[str]) -> tuple[float | None, list[str]]:
    """Equal-weighted portfolio volatility from the SAME annualized
    variance-covariance matrix (Sigma) the QUBO optimized against at
    selection time: vol = sqrt(w^T Sigma w), w = 1/len(held) for each
    held ticker. This is "ex-ante" -- what the optimizer expected, not
    what actually happened (see realized volatility in the tool for that).

    Returns (volatility_or_None, tickers_not_covered). If any ticker in
    `held` isn't in state.sigma_tickers (e.g. added manually after the
    fact, not part of the pool Sigma was estimated over), it's excluded
    from the calculation and returned in tickers_not_covered so the
    caller can disclose the gap explicitly rather than silently omitting
    it or crashing.
    """
    if not state.sigma_tickers or not state.sigma_annualized:
        return None, list(held)  # older cached state predates this field

    covered = [t for t in held if t in state.sigma_tickers]
    not_covered = [t for t in held if t not in state.sigma_tickers]
    if not covered:
        return None, not_covered

    sigma = np.array(state.sigma_annualized)
    idx = [state.sigma_tickers.index(t) for t in covered]
    sub_sigma = sigma[np.ix_(idx, idx)]
    w = np.full(len(covered), 1.0 / len(covered))
    variance = float(w @ sub_sigma @ w)
    return (float(np.sqrt(max(variance, 0.0))), not_covered)


def candidate_stats(state: PortfolioState, held: list[str], risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Per-candidate expected return / volatility / Sharpe, straight from
    the exact mu and diag(Sigma) that drove the QUBO's selection -- for
    deciding how to edit holdings, not a fresh estimate. Individual
    volatility here is each stock's own std. dev. (sqrt(Sigma_ii)),
    which deliberately ignores covariance with the rest of the portfolio
    -- that's a property of the STOCK, not of any particular combination;
    portfolio-level (covariance-aware) volatility is what
    ex_ante_volatility() computes for whatever's actually held.

    Returns one row per ticker in state.sigma_tickers (the full pool at
    computation time), with a `held` column -- empty DataFrame if the
    state predates these fields (older cached portfolio).
    """
    if not state.sigma_tickers or not state.sigma_annualized or not state.mu_annualized:
        return pd.DataFrame(columns=["ticker", "name", "mu", "volatility", "sharpe", "held"])

    sigma = np.array(state.sigma_annualized)
    mu = np.array(state.mu_annualized)
    # state.names only covers the SELECTED tickers; candidate_pool has
    # every candidate that existed at computation time, selected or not.
    pool_names = {row["yf_ticker"]: row["name"] for row in state.candidate_pool}
    rows = []
    for i, t in enumerate(state.sigma_tickers):
        vol = float(np.sqrt(max(sigma[i, i], 0.0)))
        sharpe = (mu[i] - risk_free_rate) / vol if vol > 0 else float("nan")
        rows.append({
            "ticker": t, "name": pool_names.get(t, state.names.get(t, t)),
            "mu": float(mu[i]), "volatility": vol, "sharpe": sharpe,
            "held": t in held,
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def _reconstruct_full_solution(run, tickers_all: list[str], force_include: list[str]) -> np.ndarray:
    """qp.substitute_variables renumbers remaining free variables, so
    run.result.x is shorter than len(tickers_all) when variables were
    fixed. Re-expand to a full-length 0/1 vector aligned with
    tickers_all: fixed tickers -> 1, solved free variables -> in order.
    """
    fixed_set = set(t for t in force_include if t in tickers_all)
    full = np.zeros(len(tickers_all))
    free_idx = [i for i, t in enumerate(tickers_all) if t not in fixed_set]
    for i, bit in zip(free_idx, run.result.x):
        full[i] = bit
    for i, t in enumerate(tickers_all):
        if t in fixed_set:
            full[i] = 1
    return full


def allocate_shares(
    tickers: list[str], weights: dict[str, float], prices: dict[str, float], capital: float,
    odd_lot: bool = True,
) -> pd.DataFrame:
    """Share allocation for a target capital.

    Taiwan trades in board lots of 1000 shares ('一張' / 整股) by default,
    but most brokers now also support odd-lot trading ('零股') in single
    shares. Board-lot-only rounding is coarse: at typical retail capital
    (NT$1-2M split across 10 names), a single lot of an expensive stock
    (e.g. a NT$2,000+ name needs NT$2M+ for just one lot) can exceed the
    entire per-position budget, silently producing an all-zero
    allocation that looks broken even though the math is right. Odd-lot
    mode (the default here) avoids that; pass odd_lot=False to force
    whole-board-lot sizing for brokers/accounts that require it.
    """
    unit = 1 if odd_lot else LOT_SIZE
    rows = []
    for t in tickers:
        target_value = weights[t] * capital
        price = prices[t]
        units = int(target_value // (price * unit))
        shares = units * unit
        cost = shares * price
        rows.append({
            "ticker": t, "weight": weights[t], "price": price,
            "lots": shares / LOT_SIZE, "shares": shares, "cost": cost,
            "target_value": target_value,
        })
    df = pd.DataFrame(rows)
    df["leftover_cash"] = df["target_value"] - df["cost"]
    return df


def live_performance(state: PortfolioState, current_prices: dict[str, float]) -> pd.DataFrame:
    """Equal-weighted return of each holding from inception to now."""
    rows = []
    for t in state.tickers:
        p0, p1 = state.inception_prices[t], current_prices[t]
        rows.append({"ticker": t, "name": state.names.get(t, t), "inception_price": p0, "current_price": p1, "return": p1 / p0 - 1})
    return pd.DataFrame(rows)
