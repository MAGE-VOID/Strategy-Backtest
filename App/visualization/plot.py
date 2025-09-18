from __future__ import annotations

"""
plot.py

Compatibilidad y punto de entrada público. Reexporta un plotter más modular y
robusto basado en un dashboard con N charts.
"""

from typing import Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio

from .charts import (
    BaseChart,
    DrawdownChart,
    EquityBalanceChart,
    FloatingPLChart,
    DepositLoadChart,
    CommissionsChart,
    OpenLotsChart,
    OpenTradesChart,
)
from .dashboard import BacktestDashboard


class BacktestPlotter:
    """
    Plotter de backtest (compatibilidad) que delega en BacktestDashboard.

    Ejemplos:
        BacktestPlotter().show(result)
        BacktestPlotter(max_points=200_000).build(result)
        BacktestPlotter(downsample=False).show(result)
        BacktestPlotter(use_gpu=False).show(result)
        # Con charts personalizados:
        BacktestPlotter(charts=[EquityBalanceChart(), DrawdownChart()])
    """

    def __init__(
        self,
        *,
        charts: Optional[Sequence[BaseChart]] = None,
        use_gpu: bool = True,
        max_points: Optional[int] = None,
        downsample: bool = True,
        template: str = "plotly_dark",
        renderer: str = "browser",
        title: Optional[str] = None,
    ) -> None:
        default_charts: Sequence[BaseChart] = (
            EquityBalanceChart(),
            DrawdownChart(),
            FloatingPLChart(),
            OpenTradesChart(),
            OpenLotsChart(),
            DepositLoadChart(),
            CommissionsChart(),
        )
        self._dashboard = BacktestDashboard(
            charts=(list(charts) if charts is not None else list(default_charts)),
            use_gpu=use_gpu,
            max_points=max_points,
            downsample=downsample,
            template=template,
            renderer=renderer,
            title=title,
        )
        # Mantener mismo comportamiento de render por defecto
        pio.renderers.default = renderer

    def build(self, result_backtest: dict) -> go.Figure:
        return self._dashboard.build(result_backtest)

    def show(self, result_backtest: dict) -> go.Figure:
        return self._dashboard.show(result_backtest)
