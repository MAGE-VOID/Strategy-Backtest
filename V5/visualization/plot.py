# visualization/plot.py
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# Configurar Plotly para que use el renderizador 'browser'
pio.renderers.default = "browser"


def plot_equity_balance(
    result_backtest: dict, max_points: int = 10000, theme: str = "dark"
) -> go.Figure:
    """
    Crea y muestra un gráfico interactivo que muestra la evolución de la equidad y el balance a lo largo del tiempo.
    Se aplica downsampling si el número de puntos es muy alto, manteniendo la precisión de los datos.
    Las líneas se muestran delgadas y nítidas.

    Parameters:
        result_backtest (dict): Diccionario con resultados del backtest, que debe incluir la clave "equity_over_time".
        max_points (int): Número máximo de puntos a trazar; si se supera, se reduce la muestra para mejorar el rendimiento.
        theme (str): "dark" o "white". En dark mode se utiliza el template "plotly_dark" y colores optimizados.
    """
    # Seleccionar template y mapa de colores según el tema
    if theme.lower() == "dark":
        template = "plotly_dark"
        color_map = {"equity": "#00cc96", "balance": "#EF553B"}
    elif theme.lower() == "white":
        template = "plotly_white"
        color_map = {"equity": "green", "balance": "blue"}
    else:
        template = "plotly"
        color_map = {"equity": "green", "balance": "blue"}

    pio.templates.default = template

    # Preparar los datos
    df = pd.DataFrame(result_backtest["equity_over_time"])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    n_points = len(df)
    if n_points > max_points:
        step = int(n_points / max_points)
        df = df.iloc[::step].reset_index(drop=True)

    fig = px.line(
        df,
        x="date",
        y=["equity", "balance"],
        labels={"value": "Value", "variable": "Metric", "date": "Date"},
        title="Equity and Balance Over Time",
        color_discrete_map=color_map,
        template=template,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Metric",
        hovermode="x unified",
    )
    fig.update_traces(line=dict(width=1, shape="linear"))

    fig.show()
    return fig
