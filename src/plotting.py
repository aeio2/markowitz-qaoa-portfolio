"""Plot the Monte Carlo cloud, the continuous efficient frontier, and the
discrete cardinality-constrained portfolios (exact + QAOA) on the same
risk/return plane so the classical-vs-combinatorial trade-off is visible
directly: the constrained portfolios are necessarily on or inside the
unconstrained frontier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Validated categorical/sequential palette (dataviz skill reference palette).
# Sequential blue ramp for the Monte Carlo cloud (continuous magnitude = Sharpe).
_SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]
# Chart chrome.
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_SURFACE = "#fcfcfb"
# Discrete portfolio markers: distinct hues + distinct shapes + direct labels
# (never color alone), chosen to avoid the documented yellow/orange collision.
_METHOD_STYLE = {
    "Max Sharpe (continuous)": {"color": "#eda100", "symbol": "star"},
    "Min volatility (continuous)": {"color": "#1baf7a", "symbol": "star"},
    "Exact": {"color": "#e34948", "symbol": "diamond"},
    "QAOA": {"color": "#e87ba4", "symbol": "square"},
}


def plot_frontier(
    mc_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    max_sharpe_point: tuple[float, float],
    min_vol_point: tuple[float, float],
    discrete_points: list[tuple[str, float, float]],
    output_path: str,
) -> None:
    """Save the combined efficient-frontier figure to ``output_path``.

    discrete_points: list of (label, return, volatility) for each
    cardinality-constrained solution (e.g. exact vs QAOA) to overlay.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        mc_df["volatility"],
        mc_df["return"],
        c=mc_df["sharpe"],
        cmap="viridis",
        s=6,
        alpha=0.35,
        label="Monte Carlo portfolios",
    )
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")

    ax.plot(
        frontier_df["volatility"],
        frontier_df["return"],
        color="red",
        linewidth=2,
        label="Efficient frontier (SLSQP)",
    )

    ax.scatter(
        [max_sharpe_point[1]],
        [max_sharpe_point[0]],
        marker="*",
        color="gold",
        edgecolor="black",
        s=400,
        zorder=5,
        label="Max Sharpe (continuous)",
    )
    ax.scatter(
        [min_vol_point[1]],
        [min_vol_point[0]],
        marker="*",
        color="cyan",
        edgecolor="black",
        s=400,
        zorder=5,
        label="Min volatility (continuous)",
    )

    markers = ["D", "s", "P", "X"]
    for i, (label, ret, vol) in enumerate(discrete_points):
        ax.scatter(
            [vol],
            [ret],
            marker=markers[i % len(markers)],
            s=220,
            edgecolor="black",
            zorder=6,
            label=label,
        )

    ax.set_xlabel("Annualized volatility (risk)")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Markowitz Efficient Frontier vs. Cardinality-Constrained (QUBO/QAOA) Portfolios")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def frontier_figure_plotly(
    mc_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    max_sharpe_point: tuple[float, float],
    min_vol_point: tuple[float, float],
    discrete_points: list[tuple[str, float, float]],
):
    """Interactive Plotly version of the frontier chart for the web app.

    discrete_points: list of (label, return, volatility) where label is
    one of the keys in _METHOD_STYLE (e.g. "Exact", "QAOA").
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=mc_df["volatility"], y=mc_df["return"],
        mode="markers",
        marker=dict(
            size=4,
            color=mc_df["sharpe"],
            colorscale=[[i / (len(_SEQ_BLUE) - 1), c] for i, c in enumerate(_SEQ_BLUE)],
            colorbar=dict(
                title="Sharpe<br>ratio", tickfont=dict(color=_INK_SECONDARY),
                x=1.02, len=0.9, thickness=14,
            ),
            opacity=0.45,
            line=dict(width=0),
        ),
        name="Monte Carlo portfolios",
        hovertemplate="Volatility %{x:.3f}<br>Return %{y:.3f}<br>Sharpe %{marker.color:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"], y=frontier_df["return"],
        mode="lines",
        line=dict(color=_INK_PRIMARY, width=2.5),
        name="Efficient frontier (SLSQP)",
        hovertemplate="Volatility %{x:.3f}<br>Return %{y:.3f}<extra>Frontier</extra>",
    ))

    # Points near the left edge of the cloud get right-anchored labels (and
    # vice versa) so text renders inside the plot instead of clipping at
    # the axis boundary.
    vol_mid = (mc_df["volatility"].min() + mc_df["volatility"].max()) / 2

    def add_point(label: str, ret: float, vol: float) -> None:
        style = _METHOD_STYLE.get(label, {"color": _INK_SECONDARY, "symbol": "circle"})
        textposition = "middle right" if vol <= vol_mid else "middle left"
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret],
            mode="markers+text",
            marker=dict(
                size=16, color=style["color"], symbol=style["symbol"],
                line=dict(width=1.5, color=_INK_PRIMARY),
            ),
            text=[label],
            textposition=textposition,
            textfont=dict(color=_INK_PRIMARY, size=11),
            name=label,
            hovertemplate=f"{label}<br>Volatility %{{x:.3f}}<br>Return %{{y:.3f}}<extra></extra>",
        ))

    add_point("Max Sharpe (continuous)", max_sharpe_point[0], max_sharpe_point[1])
    add_point("Min volatility (continuous)", min_vol_point[0], min_vol_point[1])
    for label, ret, vol in discrete_points:
        add_point(label, ret, vol)

    fig.update_layout(
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(color=_INK_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(title="Annualized volatility (risk)", gridcolor=_GRIDLINE, zerolinecolor=_GRIDLINE, tickfont=dict(color=_INK_MUTED)),
        yaxis=dict(title="Annualized expected return", gridcolor=_GRIDLINE, zerolinecolor=_GRIDLINE, tickfont=dict(color=_INK_MUTED)),
        legend=dict(
            bgcolor=_SURFACE, bordercolor=_GRIDLINE, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        margin=dict(l=60, r=90, t=90, b=50),
        height=620,
    )
    return fig


def weights_figure_plotly(tickers: list[str], weights_by_method: dict[str, np.ndarray]):
    """Grouped bar chart of portfolio weights per ticker, one series per
    method, using the same colors/legend as frontier_figure_plotly so the
    two charts read as one system.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for label, weights in weights_by_method.items():
        style = _METHOD_STYLE.get(label, {"color": _INK_SECONDARY, "symbol": "circle"})
        fig.add_trace(go.Bar(
            x=tickers, y=weights,
            name=label,
            marker=dict(color=style["color"]),
            hovertemplate=f"{label}<br>%{{x}}: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(color=_INK_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(title=None, gridcolor=_GRIDLINE, tickfont=dict(color=_INK_MUTED)),
        yaxis=dict(title="Portfolio weight", tickformat=".0%", gridcolor=_GRIDLINE, tickfont=dict(color=_INK_MUTED)),
        legend=dict(bgcolor=_SURFACE, bordercolor=_GRIDLINE, borderwidth=1),
        margin=dict(l=60, r=20, t=20, b=50),
        height=380,
    )
    return fig
