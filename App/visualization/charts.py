from __future__ import annotations

"""
charts.py

Definición de clases de chart reutilizables para el dashboard.
Cada chart sabe cómo añadir sus trazas al subplot correspondiente.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .data_prep import PreparedData


# Paleta por defecto
COLORS = {
    "equity": "#00cc96",
    "balance": "#EF553B",
    "drawdown": "#FFA15A",
    "floating": "#636EFA",
    "deposit_load": "#19D3F3",
    "open_trades": "#FF851B",
    "open_lots": "#AB63FA",
    "ot_long": "#2ECC71",
    "ot_short": "#E74C3C",
    "ol_long": "#1ABC9C",
    "ol_short": "#C0392B",
}


def _step_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una serie (x,y) en forma 'hv' duplicando puntos para usar Scattergl."""
    n = len(x)
    if n <= 1:
        return x, y
    x2 = np.empty(2 * n - 1, dtype=x.dtype)
    y2 = np.empty(2 * n - 1, dtype=y.dtype)
    x2[0] = x[0]
    y2[0] = y[0]
    x2[1::2] = x[1:]
    y2[1::2] = y[:-1]
    x2[2::2] = x[1:]
    y2[2::2] = y[1:]
    return x2, y2


def _line(
    fig: go.Figure,
    x,
    y,
    name: str,
    color: str,
    row: int,
    *,
    step: bool = False,
    dash: Optional[str] = None,
    visible: Optional[bool] = True,
    hoverfmt: Optional[str] = None,
    suffix: str = "",
) -> None:
    if step:
        x, y = _step_xy(np.asarray(x), np.asarray(y))
    trace_cls = go.Scattergl
    if hoverfmt:
        hovertemplate = f"%{{y{hoverfmt}}}{suffix}<extra>" + name + "</extra>"
    else:
        hovertemplate = "%{y}" + suffix + "<extra>" + name + "</extra>"
    fig.add_trace(
        trace_cls(
            x=x,
            y=y,
            mode="lines",
            name=name,
            hovertemplate=hovertemplate,
            line=dict(width=1, color=color, dash=dash),
            visible=("legendonly" if visible is False else True),
        ),
        row=row,
        col=1,
    )


class BaseChart:
    def title(self) -> str:
        raise NotImplementedError

    def yaxis_title(self) -> str:
        return "Value"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        raise NotImplementedError


@dataclass
class EquityBalanceChart(BaseChart):
    show_balance_step: bool = True

    def title(self) -> str:
        return "Equity (MTM) y Balance (Step)" if self.show_balance_step else "Equity (MTM)"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        _line(fig, data.dates, data.equity, "Equity", COLORS["equity"], row=row, hoverfmt=":.2f")
        if self.show_balance_step:
            _line(
                fig,
                data.dates,
                data.balance,
                "Balance",
                COLORS["balance"],
                row=row,
                step=True,
                hoverfmt=":.2f",
            )


class DrawdownChart(BaseChart):
    def title(self) -> str:
        return "Floating vs Initial (%)"

    def yaxis_title(self) -> str:
        return "Floating [%]"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        _line(
            fig,
            data.dates,
            data.floating_init_pct,
            "Floating vs Initial (%)",
            COLORS["drawdown"],
            row=row,
            hoverfmt=":.2f",
            suffix="%",
        )


class FloatingPLChart(BaseChart):
    def title(self) -> str:
        return "Floating P/L"

    def yaxis_title(self) -> str:
        return "Floating P/L"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        _line(fig, data.dates, data.floating, "Floating P/L", COLORS["floating"], row=row, hoverfmt=":.2f")


@dataclass
class OpenTradesChart(BaseChart):
    show_sides: bool = True

    def title(self) -> str:
        return "Open Trades"

    def yaxis_title(self) -> str:
        return "Open Trades"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        _line(
            fig,
            data.dates,
            data.open_trades_total,
            "Open Trades (Total)",
            COLORS["open_trades"],
            row=row,
            hoverfmt=":.0f",
        )
        if self.show_sides and data.ot_long is not None:
            _line(
                fig,
                data.dates,
                data.ot_long,
                "Open Trades (Long)",
                COLORS["ot_long"],
                row=row,
                dash="dot",
                visible=False,
                hoverfmt=":.0f",
            )
            _line(
                fig,
                data.dates,
                data.ot_short,
                "Open Trades (Short)",
                COLORS["ot_short"],
                row=row,
                dash="dot",
                visible=False,
                hoverfmt=":.0f",
            )


@dataclass
class OpenLotsChart(BaseChart):
    show_sides: bool = True

    def title(self) -> str:
        return "Open Lots"

    def yaxis_title(self) -> str:
        return "Open Lots"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        _line(
            fig,
            data.dates,
            data.open_lots_total,
            "Open Lots (Total)",
            COLORS["open_lots"],
            row=row,
            hoverfmt=":.2f",
        )
        if self.show_sides and data.ol_long is not None:
            _line(
                fig,
                data.dates,
                data.ol_long,
                "Open Lots (Long)",
                COLORS["ol_long"],
                row=row,
                dash="dot",
                visible=False,
                hoverfmt=":.2f",
            )
            _line(
                fig,
                data.dates,
                data.ol_short,
                "Open Lots (Short)",
                COLORS["ol_short"],
                row=row,
                dash="dot",
                visible=False,
                hoverfmt=":.2f",
            )


@dataclass
class DepositLoadChart(BaseChart):
    def title(self) -> str:
        return "Deposit Load (%)"

    def yaxis_title(self) -> str:
        return "Load [%]"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        _line(fig, data.dates, data.deposit_load_pct, "Deposit Load (%)", COLORS["deposit_load"], row=row, hoverfmt=":.2f", suffix="%")


@dataclass
class CommissionsChart(BaseChart):
    def title(self) -> str:
        return "Commissions"

    def yaxis_title(self) -> str:
        return "$"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        # Barras apiladas: open y close; línea: total acumulado
        fig.add_trace(
            go.Bar(
                x=data.dates,
                y=data.comm_open,
                name="Comm Open",
                marker_color="#FFA15A",
                hovertemplate="%{y:.2f}<extra>Comm Open</extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=data.dates,
                y=data.comm_close,
                name="Comm Close",
                marker_color="#EF553B",
                hovertemplate="%{y:.2f}<extra>Comm Close</extra>",
            ),
            row=row,
            col=1,
        )
        _line(
            fig,
            data.dates,
            data.comm_total_cum,
            "Comm Total (Cum)",
            "#19D3F3",
            row=row,
            hoverfmt=":.2f",
        )


@dataclass
class ReturnBalanceChart(BaseChart):
    def title(self) -> str:
        return "Return vs Initial (Balance %)"

    def yaxis_title(self) -> str:
        return "Return [%]"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        init = float(data.initial_balance or (data.balance[0] if len(data.balance) else 1.0))
        init = init if init > 0 else 1.0
        series = (np.asarray(data.balance, dtype=float) / init - 1.0) * 100.0
        _line(fig, data.dates, series, "Return vs Initial (Balance)", COLORS["equity"], row=row, hoverfmt=":.2f%")


@dataclass
class ReturnEquityChart(BaseChart):
    def title(self) -> str:
        return "Return vs Initial (Equity %)"

    def yaxis_title(self) -> str:
        return "Return [%]"

    def add_to(self, fig: go.Figure, data: PreparedData, row: int) -> None:
        init = float(data.initial_balance or (data.equity[0] if len(data.equity) else 1.0))
        init = init if init > 0 else 1.0
        series = (np.asarray(data.equity, dtype=float) / init - 1.0) * 100.0
        _line(fig, data.dates, series, "Return vs Initial (Equity)", COLORS["balance"], row=row, hoverfmt=":.2f%")
