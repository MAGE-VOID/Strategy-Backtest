from __future__ import annotations

"""
dashboard.py

Orquestador del dashboard de backtest. Soporta N charts mediante clases.
Separa preparación de datos (data_prep) de la construcción de gráficos.
"""

from typing import Iterable, List, Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .charts import BaseChart
from .data_prep import DataPreparer, PreparedData


class BacktestDashboard:
    """Dashboard configurable para backtests con N charts.

    Parámetros principales:
      - charts: lista de objetos que implementan BaseChart (orden define las filas)
      - use_gpu, max_points, downsample: control de rendimiento
      - template: tema Plotly (por defecto dark)
      - renderer: salida (por defecto navegador)
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
        height_per_row: int = 260,
    ) -> None:
        self.charts: List[BaseChart] = list(charts) if charts is not None else []
        self.prep = DataPreparer(use_gpu=use_gpu, max_points=max_points, downsample=downsample)
        self.template = template
        self.renderer = renderer
        self.title = title or "Cuenta única con múltiples estrategias (aisladas por symbol+magic)"
        self.height_per_row = int(height_per_row)

    def add_chart(self, chart: BaseChart) -> None:
        self.charts.append(chart)

    def build(self, result_backtest: dict) -> go.Figure:
        if not self.charts:
            raise ValueError("No hay charts definidos. Usa add_chart(...) o pasa una lista en el constructor.")

        pio.templates.default = self.template
        pio.renderers.default = self.renderer

        data: PreparedData = self.prep.prepare(result_backtest)

        # Layout vertical con ejes X compartidos
        rows = len(self.charts)
        titles = [c.title() for c in self.charts]
        # Distribución de alturas: más peso a la primera fila
        if rows == 1:
            row_heights = [1.0]
        else:
            # 46% primera, resto proporcional
            rest = 1.0 - 0.46
            per = rest / (rows - 1)
            row_heights = [0.46] + [per] * (rows - 1)

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=row_heights,
            subplot_titles=titles,
        )

        # Añadir trazas por chart
        for i, chart in enumerate(self.charts, start=1):
            chart.add_to(fig, data, row=i)

        # Ejes y layout
        for i, chart in enumerate(self.charts, start=1):
            fig.update_yaxes(title_text=chart.yaxis_title(), row=i, col=1, fixedrange=True)

        # Altura basada en número de filas para hacer la página scrolleable cuando sea necesario
        target_height = max(300, int(self.height_per_row * rows))

        fig.update_layout(
            hovermode="x unified",
            legend_title="Metric",
            title=self.title,
            xaxis_title="Date",
            dragmode="zoom",
            autosize=True,
            height=target_height,
            barmode="relative",
        )

        return fig

    def show(self, result_backtest: dict) -> go.Figure:
        fig = self.build(result_backtest)
        fig.show(config={"responsive": True, "scrollZoom": True})
        return fig
