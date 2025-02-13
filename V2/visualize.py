import pandas as pd
import matplotlib.pyplot as plt


# Función para graficar el balance y la equidad
def plot_equity_balance(resultBacktest):
    equity_data = pd.DataFrame(resultBacktest["equity_over_time"])

    # Convertir las fechas a formato datetime
    equity_data["date"] = pd.to_datetime(equity_data["date"])

    # Crear la figura y los ejes
    plt.figure(figsize=(10, 6))

    # Graficar equity
    plt.plot(equity_data["date"], equity_data["equity"], label="Equity", color="green")

    # Graficar balance
    plt.plot(equity_data["date"], equity_data["balance"], label="Balance", color="blue")

    # Agregar título y etiquetas
    plt.title("Equity and Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")

    # Mostrar leyenda
    plt.legend()

    # Mostrar el gráfico
    plt.grid(True)
    plt.show()


# Función para graficar el precio con señales de compra, venta y cierres
def plot_price_with_signals(resultBacktest, df, symbol):
    # Extraer las operaciones (trades) y los datos de precio para el símbolo específico
    trades = pd.DataFrame(resultBacktest["trades"])
    price_data = df[df["Symbol"] == symbol].copy()

    # Asegurar que el índice 'date' es datetime para sincronizar con las operaciones
    price_data.index = pd.to_datetime(price_data.index)
    trades["open_time"] = pd.to_datetime(trades["open_time"])
    trades["close_time"] = pd.to_datetime(trades["close_time"])

    # Crear la figura y el gráfico
    plt.figure(figsize=(12, 8))

    # Graficar la serie de precios (usando el precio de cierre)
    plt.plot(
        price_data.index, price_data["Close"], label=f"{symbol} Price", color="blue"
    )

    # Recorrer todas las operaciones y graficar las señales de compra y venta
    for _, trade in trades.iterrows():
        if trade["symbol"] == symbol:
            if trade["status"] == "open" and trade["type"] == "long":
                # Señal de apertura de compra (Buy Open)
                plt.scatter(
                    trade["open_time"],
                    trade["entry"],
                    color="green",
                    marker="^",
                    s=100,
                    label=(
                        "Buy (Open)"
                        if "Buy (Open)" not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                )
            elif trade["status"] == "closed" and trade["type"] == "long":
                # Señal de cierre de compra (Buy Close)
                plt.scatter(
                    trade["close_time"],
                    trade["exit"],
                    color="red",
                    marker="v",
                    s=100,
                    label=(
                        "Buy (Closed)"
                        if "Buy (Closed)"
                        not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                )
            elif trade["status"] == "open" and trade["type"] == "short":
                # Señal de apertura de venta (Sell Open)
                plt.scatter(
                    trade["open_time"],
                    trade["entry"],
                    color="orange",
                    marker="v",
                    s=100,
                    label=(
                        "Sell (Open)"
                        if "Sell (Open)" not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                )
            elif trade["status"] == "closed" and trade["type"] == "short":
                # Señal de cierre de venta (Sell Close)
                plt.scatter(
                    trade["close_time"],
                    trade["exit"],
                    color="purple",
                    marker="^",
                    s=100,
                    label=(
                        "Sell (Closed)"
                        if "Sell (Closed)"
                        not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                )

    # Mejorar la visualización ajustando los límites del eje Y
    plt.ylim(price_data["Close"].min() * 0.95, price_data["Close"].max() * 1.05)

    # Añadir título, etiquetas, cuadrícula y leyenda
    plt.title(f"{symbol} Price with Buy and Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    # Mostrar el gráfico
    plt.show()
