import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots

# Configurar Plotly para que use el renderizador 'browser'
pio.renderers.default = "browser"


def add_trace_gl(fig, df, x_col, y_col, name, color, row, col):
    """Función auxiliar para agregar una traza usando Scattergl."""
    fig.add_trace(
        go.Scattergl(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=name,
            line=dict(color=color, width=1),
        ),
        row=row,
        col=col,
    )


def plot_equity_balance(
    result_backtest: dict, max_points: int = 10000, theme: str = "dark"
) -> go.Figure:
    """
    Crea y muestra un gráfico interactivo responsive con cuatro subgráficos utilizando Scattergl:
      - Subplot 1 (grande): Evolución de la Equity y el Balance.
      - Subplot 2 (pequeño): Evolución del Drawdown Exposure (equity - balance).
      - Subplot 3 (pequeño): Rotación de Operaciones Abiertas (cantidad de operaciones abiertas).
      - Subplot 4 (pequeño): Cantidad total de lotes abiertos.

    El gráfico está configurado para que el pan vertical esté bloqueado (los ejes Y son fijos), de modo que
    al hacer scroll se realice zoom horizontal, manteniendo los rangos verticales.

    Se aplica downsampling si el número de puntos supera max_points para conservar rendimiento.

    NOTA: Para lograr scroll vertical interno, es recomendable incrustar la figura en un contenedor HTML
    con CSS (por ejemplo, `overflow-y: auto; max-height: 90vh;`).

    Parameters:
      result_backtest (dict): Diccionario con resultados del backtest (debe incluir "equity_over_time").
      max_points (int): Número máximo de puntos a trazar.
      theme (str): "dark" o "white". Se usa "plotly_dark" para dark mode.

    Returns:
      go.Figure: La figura interactiva generada.
    """
    # Seleccionar template y colores según el tema
    if theme.lower() == "dark":
        template = "plotly_dark"
        color_map = {
            "equity": "#00cc96",
            "balance": "#EF553B",
            "drawdown_exposure": "#636EFA",
            "open_trades": "#FF851B",
            "open_lots": "#AB63FA",
        }
    elif theme.lower() == "white":
        template = "plotly_white"
        color_map = {
            "equity": "green",
            "balance": "blue",
            "drawdown_exposure": "orange",
            "open_trades": "purple",
            "open_lots": "magenta",
        }
    else:
        template = "plotly"
        color_map = {
            "equity": "green",
            "balance": "blue",
            "drawdown_exposure": "orange",
            "open_trades": "purple",
            "open_lots": "magenta",
        }
    pio.templates.default = template

    # Preparar los datos
    df = pd.DataFrame(result_backtest["equity_over_time"])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    n_points = len(df)
    if n_points > max_points:
        step = int(n_points / max_points)
        df = df.iloc[::step].reset_index(drop=True)

    if "drawdown_exposure" not in df.columns:
        df["drawdown_exposure"] = df["equity"] - df["balance"]
    if "open_trades" not in df.columns:
        df["open_trades"] = np.nan
    if "open_lots" not in df.columns:
        df["open_lots"] = np.nan

    # Crear la figura con 4 subplots con alturas relativas:
    # Subplot 1: 60% de la altura; Subplot 2 y 3: 15% cada uno; Subplot 4: 10%.
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.15, 0.15, 0.10],
        subplot_titles=(
            "Equity and Balance",
            "Drawdown Exposure",
            "Open Trades (Rotation)",
            "Open Lots",
        ),
    )

    # Subplot 1: Equity y Balance
    add_trace_gl(fig, df, "date", "equity", "Equity", color_map["equity"], row=1, col=1)
    add_trace_gl(
        fig, df, "date", "balance", "Balance", color_map["balance"], row=1, col=1
    )

    # Subplot 2: Drawdown Exposure
    add_trace_gl(
        fig,
        df,
        "date",
        "drawdown_exposure",
        "Drawdown Exposure",
        color_map["drawdown_exposure"],
        row=2,
        col=1,
    )

    # Subplot 3: Open Trades
    add_trace_gl(
        fig,
        df,
        "date",
        "open_trades",
        "Open Trades",
        color_map["open_trades"],
        row=3,
        col=1,
    )

    # Subplot 4: Open Lots
    add_trace_gl(
        fig, df, "date", "open_lots", "Open Lots", color_map["open_lots"], row=4, col=1
    )

    # Actualizar layout para que sea responsive y bloquear pan vertical (fijando los ejes Y)
    fig.update_layout(
        template=template,
        title="Equity, Balance, Drawdown Exposure, Open Trades and Open Lots Over Time",
        hovermode="x unified",
        legend_title="Metric",
        xaxis_title="Date",
        dragmode="zoom",  # Permite zoom en lugar de pan
        autosize=True,
    )
    # Fijar los ejes Y para bloquear pan vertical (permitiendo zoom horizontal)
    fig.update_yaxes(title_text="Value", row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Drawdown Exposure", row=2, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Open Trades", row=3, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Open Lots", row=4, col=1, fixedrange=True)

    # Mostrar figura en modo responsive.
    fig.show(config={"responsive": True, "scrollZoom": True})
    return fig
