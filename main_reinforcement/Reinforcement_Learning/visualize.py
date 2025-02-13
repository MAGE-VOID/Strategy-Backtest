import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_trades(df, trades_log, symbol="EURUSD"):
    plt.figure(figsize=(12, 6))

    # Extraer las fechas y precios de cierre
    prices = df["Close"].values
    dates = df.index

    # Graficar la serie temporal de los precios
    plt.plot(dates, prices, label=f"Precio de {symbol}", color="blue")

    # Variables para las señales de compra y cierre
    buy_signals = []
    close_signals = []

    # Procesar las operaciones registradas en trades_log
    for trade in trades_log:
        date = trade["date"]
        price = trade["price"]
        trade_id = trade["id"]

        if trade["action"] == "buy":
            buy_signals.append((date, price, trade_id))
        elif trade["action"] == "close":
            close_signals.append((date, price, trade_id))

    # Añadir puntos verdes para las compras e incluir el ID como anotación
    if buy_signals:
        buy_dates, buy_prices, buy_ids = zip(*buy_signals)
        plt.scatter(
            buy_dates, buy_prices, color="green", label="Compra", marker="^", s=100
        )
        for i, buy_id in enumerate(buy_ids):
            plt.text(
                buy_dates[i], buy_prices[i], f"ID: {buy_id}", color="green", fontsize=9
            )

    # Añadir puntos rojos para los cierres de posiciones e incluir el ID como anotación
    if close_signals:
        close_dates, close_prices, close_ids = zip(*close_signals)
        plt.scatter(
            close_dates, close_prices, color="red", label="Cierre", marker="v", s=100
        )
        for i, close_id in enumerate(close_ids):
            plt.text(
                close_dates[i],
                close_prices[i],
                f"ID: {close_id}",
                color="red",
                fontsize=9,
            )

    # Configurar el gráfico
    plt.title(f"Operaciones de compra y cierre para {symbol}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre")
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()



def plot_training_progress(total_rewards, total_losses):
    """
    Función para visualizar las recompensas y pérdidas durante el entrenamiento en subgráficos.

    Parámetros:
    - total_rewards: Lista de las recompensas totales por cada episodio.
    - total_losses: Lista de las pérdidas totales por cada episodio.
    """
    epochs = range(1, len(total_rewards) + 1)

    # Crear una figura con dos subplots: uno para recompensas y otro para pérdidas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Recompensas
    ax1.plot(epochs, total_rewards, label="Recompensa Total", color="blue", marker="o")
    ax1.set_xlabel("Episodios")
    ax1.set_ylabel("Recompensa Total")
    ax1.set_title("Recompensa Total por Episodio")
    ax1.grid(True)

    # Pérdidas
    ax2.plot(epochs, total_losses, label="Pérdida Total", color="red", marker="x")
    ax2.set_xlabel("Episodios")
    ax2.set_ylabel("Pérdida Total")
    ax2.set_title("Pérdida Total por Episodio")
    ax2.grid(True)

    # Ajustar el layout para evitar la superposición
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
