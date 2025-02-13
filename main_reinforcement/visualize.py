import pandas as pd
import matplotlib.pyplot as plt

# Función para graficar el balance y la equidad y opcionalmente imprimir equity_over_time
def plot_equity_balance(resultBacktest, show_data=False):
    equity_data = pd.DataFrame(resultBacktest["equity_over_time"])

    # Convertir las fechas a formato datetime
    equity_data["date"] = pd.to_datetime(equity_data["date"])

    # Si show_data es True, imprimir equity_over_time
    if show_data:
        print("\nEquity Over Time Data:")
        print(equity_data)

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

def plot_rewards_and_losses(episodes, rewards_per_episode, losses_per_episode):
    """
    Función para graficar recompensas y pérdidas por episodio.
    
    :param episodes: Número total de episodios.
    :param rewards_per_episode: Lista de recompensas acumuladas por episodio.
    :param losses_per_episode: Lista de pérdidas acumuladas por episodio.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Gráfica de recompensas
    ax1.plot(range(1, episodes + 1), rewards_per_episode, label="Recompensas", color="blue")
    ax1.set_title("Recompensas por Episodio")
    ax1.set_xlabel("Episodios")
    ax1.set_ylabel("Recompensas")
    ax1.grid(True)

    # Gráfica de pérdidas
    ax2.plot(range(1, episodes + 1), losses_per_episode, label="Pérdidas", color="red")
    ax2.set_title("Pérdidas por Episodio")
    ax2.set_xlabel("Episodios")
    ax2.set_ylabel("Pérdidas")
    ax2.grid(True)

    # Ajustar layout y mostrar la gráfica
    plt.tight_layout()
    plt.show()