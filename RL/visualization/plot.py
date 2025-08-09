import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots

# Configurar Plotly para que use el renderizador 'browser'
pio.renderers.default = "browser"


def _downsample_evenly(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df.reset_index(drop=True)
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def add_trace_gl(fig, df, x_col, y_col, name, color, row, col):
    fig.add_trace(
        go.Scattergl(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=name,
            line=dict(width=1, color=color),
        ),
        row=row,
        col=col,
    )


def _dd_series(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    cum_max = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(cum_max > 0, (cum_max - values) / cum_max * 100.0, 0.0)
    return -dd_pct  # negativo (0 en máximos, <0 en DD)


def plot_equity_balance(
    result_backtest: dict, max_points: int = 10000, theme: str = "dark"
) -> go.Figure:
    """
    1) Equity (línea) y Balance (escalera tipo MT5)
    2) Drawdown de Equity (%)
    3) Floating P/L (Equity - Balance)
    4) # Operaciones abiertas
    5) Lotes abiertos
    """
    # Template y colores
    if theme.lower() == "dark":
        template = "plotly_dark"
        color_map = {
            "equity": "#00cc96",
            "balance": "#EF553B",
            "drawdown": "#FFA15A",
            "floating": "#636EFA",
            "open_trades": "#FF851B",
            "open_lots": "#AB63FA",
        }
    elif theme.lower() == "white":
        template = "plotly_white"
        color_map = {
            "equity": "green",
            "balance": "blue",
            "drawdown": "orange",
            "floating": "orange",
            "open_trades": "purple",
            "open_lots": "magenta",
        }
    else:
        template = "plotly"
        color_map = {
            "equity": "green",
            "balance": "blue",
            "drawdown": "orange",
            "floating": "orange",
            "open_trades": "purple",
            "open_lots": "magenta",
        }
    pio.templates.default = template

    # -------------------- Preparar datos -------------------- #
    if "equity_over_time" not in result_backtest:
        raise ValueError("result_backtest debe incluir la clave 'equity_over_time'.")

    df = pd.DataFrame(result_backtest["equity_over_time"])
    if df.empty:
        raise ValueError("No hay datos en 'equity_over_time' para graficar.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ("equity", "balance"):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en equity_over_time.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["equity", "balance"])
    df["floating_pl"] = df["equity"] - df["balance"]

    # Drawdown de Equity (%)
    dd_equity = _dd_series(df["equity"].to_numpy())
    df["dd_equity_pct"] = dd_equity

    if "open_trades" not in df.columns:
        df["open_trades"] = np.nan
    if "open_lots" not in df.columns:
        df["open_lots"] = np.nan

    # Downsample (manteniendo coherencia)
    df = _downsample_evenly(df, max_points)

    # -------------------- Figura y trazas -------------------- #
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.46, 0.14, 0.14, 0.13, 0.13],
        subplot_titles=(
            "Equity (MTM) y Balance (Step)",
            "Drawdown de Equity (%)",
            "Floating P/L (Equity - Balance)",
            "Open Trades",
            "Open Lots",
        ),
    )

    # 1) Equity + Balance
    add_trace_gl(fig, df, "date", "equity", "Equity", color_map["equity"], row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["balance"],
            mode="lines",
            name="Balance",
            line=dict(width=1, color=color_map["balance"]),
            line_shape="hv",  # escalera tipo MT5
        ),
        row=1,
        col=1,
    )

    # 2) Drawdown de Equity (%)
    add_trace_gl(
        fig,
        df,
        "date",
        "dd_equity_pct",
        "DD Equity (%)",
        color_map["drawdown"],
        row=2,
        col=1,
    )

    # 3) Floating P/L
    add_trace_gl(
        fig,
        df,
        "date",
        "floating_pl",
        "Floating P/L",
        color_map["floating"],
        row=3,
        col=1,
    )

    # 4) Open Trades
    add_trace_gl(
        fig,
        df,
        "date",
        "open_trades",
        "Open Trades",
        color_map["open_trades"],
        row=4,
        col=1,
    )

    # 5) Open Lots
    add_trace_gl(
        fig, df, "date", "open_lots", "Open Lots", color_map["open_lots"], row=5, col=1
    )

    # -------------------- Layout -------------------- #
    fig.update_layout(
        template=template,
        title="Cuenta única con múltiples estrategias (aisladas por symbol+magic)",
        hovermode="x unified",
        legend_title="Metric",
        xaxis_title="Date",
        dragmode="zoom",
        autosize=True,
    )

    fig.update_yaxes(title_text="Value", row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="DD [%]", row=2, col=1, fixedrange=True)
    fig.update_yaxes(
        title_text="Floating P/L", row=3, col=1, zeroline=True, fixedrange=True
    )
    fig.update_yaxes(
        title_text="Open Trades", row=4, col=1, rangemode="tozero", fixedrange=True
    )
    fig.update_yaxes(
        title_text="Open Lots", row=5, col=1, rangemode="tozero", fixedrange=True
    )

    fig.show(config={"responsive": True, "scrollZoom": True})
    return fig
