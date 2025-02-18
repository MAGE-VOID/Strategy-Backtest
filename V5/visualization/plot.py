# visualization/plot.py
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Configurar Plotly para que use el renderizador 'browser'
pio.renderers.default = "browser"


def plot_equity_balance(result_backtest: dict, max_points: int = 10000) -> None:
    """
    Crea y muestra un gráfico interactivo que muestra la evolución de la equidad y el balance a lo largo del tiempo.
    Se aplica downsampling si el número de puntos es muy alto, pero se mantiene la precisión de los datos.
    Las líneas se muestran delgadas y nítidas (equity en verde y balance en azul).

    Parameters:
        result_backtest (dict): Diccionario con resultados del backtest, que debe incluir la clave "equity_over_time".
        max_points (int): Número máximo de puntos a trazar. Si se supera, se reduce la muestra para mejorar el rendimiento.
    """
    # Convertir los datos a DataFrame y asegurar que 'date' es datetime
    equity_data = pd.DataFrame(result_backtest["equity_over_time"])
    equity_data["date"] = pd.to_datetime(equity_data["date"])
    equity_data.sort_values("date", inplace=True)

    # Downsampling: si hay demasiados puntos, se reduce la muestra sin perder precisión general
    n_points = len(equity_data)
    if n_points > max_points:
        step = int(n_points / max_points)
        equity_data = equity_data.iloc[::step].reset_index(drop=True)

    # Crear el gráfico interactivo con Plotly Express, definiendo los colores fijos
    fig = px.line(
        equity_data,
        x="date",
        y=["equity", "balance"],
        labels={"value": "Value", "variable": "Metric", "date": "Date"},
        title="Equity and Balance Over Time",
        color_discrete_map={"equity": "green", "balance": "blue"},
    )

    # Actualizar el layout para mejorar la interactividad y presentación
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Metric",
        hovermode="x unified",
    )

    # Ajustar los trazados para que las líneas sean más delgadas y precisas
    fig.update_traces(line=dict(width=1, shape="linear"))

    # Mostrar el gráfico en una nueva ventana del navegador
    fig.show()
