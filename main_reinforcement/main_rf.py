import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
from agentDQN import DQNAgent
import CustomLibrary as CL
import BacktestEngine as BT
import Entry_Strategy_Manager_RF as ES
import visualize as VZ

# Parámetros de MetaTrader5
account_number = 51344621
server_name = "ICMarketsSC-Demo"
password = "bCFNLB9k"

mt5.initialize()
CL.LoginAccount(account_number, server_name, password)

# Parámetros de entrada para los datos
inp_start_date = datetime(2020, 10, 1)
inp_end_date = datetime(2020, 10, 31)
timeframe = mt5.TIMEFRAME_M5

# Definir los símbolos
symbols = [
    "AUDUSD",
    "EURUSD",
    "GBPUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
]

# Procesar datos
df, df_standardized, df_manual_standardized = CL.Process_Data(
    inp_start_date, inp_end_date, symbols, timeframe
)

# Parámetros del agente
state_size = 50
action_size = 2
batch_size = 32
episodes = 1000
num_agents = 3

# Crear múltiples agentes
agents = [DQNAgent(state_size=state_size, action_size=action_size) for _ in range(num_agents)]

# Inicializar el backtest engine para múltiples agentes
bt_engine = BT.BacktestEngineRL(initial_balance=1000, num_agents=num_agents)

# Inicializar el Entry Strategy Manager
entry_manager = ES.EntryManagerRL(bt_engine.position_managers)

# Variables para almacenar el mejor resultado por agente
best_rewards = [-np.inf] * num_agents  # Usar recompensas para seleccionar el mejor agente
best_results = [None] * num_agents
best_episodes = [None] * num_agents

# Listas para almacenar recompensas, pérdidas y drawdowns por episodio para cada agente
rewards_per_episode = [[] for _ in range(num_agents)]
losses_per_episode = [[] for _ in range(num_agents)]
drawdowns_per_episode = [[] for _ in range(num_agents)]  # Guardar drawdowns

# Ciclo de entrenamiento para múltiples agentes
for episode in range(episodes):
    print(f"--- Episodio {episode + 1}/{episodes} ---")

    # Reiniciar balance y posiciones para cada agente al inicio de cada episodio
    for pm in bt_engine.position_managers:
        pm.balance = 1000
        pm.positions = {}

    # Ejecutar el backtest para este episodio con múltiples agentes
    resultBacktest = bt_engine.run_backtest(
        df, entry_manager, agents, window_size=100, DebugPositions=False
    )

    # Obtener y almacenar estadísticas para cada agente
    for agent_idx, agent in enumerate(agents):
        statistics = resultBacktest["statistics"][agent_idx]
        reward = agent.calculate_reward(statistics)

        # Obtener drawdown y guardarlo
        drawdown_max = statistics["equity_maximal_drawdown"]  # Guardar drawdown máximo
        print(f"Agente {agent_idx + 1}: Drawdown final después del episodio {episode + 1}: {drawdown_max}")
        drawdowns_per_episode[agent_idx].append(drawdown_max)

        rewards_per_episode[agent_idx].append(reward)
        losses_per_episode[agent_idx].append(statistics["total_loss"])

        # Verificar si es el mejor modelo para este agente basado en la recompensa (no balance)
        if reward > best_rewards[agent_idx]:  # Usar recompensas en lugar del balance
            best_rewards[agent_idx] = reward
            best_results[agent_idx] = {
                "trades": resultBacktest["trades"][agent_idx],
                "equity_over_time": resultBacktest["equity_over_time"][agent_idx]
            }
            best_episodes[agent_idx] = episode + 1

        # Entrenar al agente después de cada episodio
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Al final de cada episodio, seleccionar el mejor agente basado en la recompensa (calculate_reward)
    best_agent_idx = np.argmax(best_rewards)  # Seleccionar el mejor agente en función de la recompensa
    print(f"El mejor agente en el episodio {episode + 1} fue el agente {best_agent_idx + 1}")

    # Copiar los pesos y la memoria del mejor agente a los demás agentes (herencia)
    best_agent = agents[best_agent_idx]
    for agent_idx, agent in enumerate(agents):
        if agent_idx != best_agent_idx:  # No copiar en el mejor agente
            agent.model.set_weights(best_agent.model.get_weights())  # Copiar pesos del modelo
            agent.memory = best_agent.memory.copy()  # Copiar memoria del mejor agente

# Identificar el mejor agente en general al final de todos los episodios
best_agent_idx = np.argmax(best_rewards)  # Basado en la recompensa calculada
print(f"El mejor agente fue el agente {best_agent_idx + 1} con una recompensa final de {best_rewards[best_agent_idx]}")

# Graficar la equidad y balance del mejor agente usando el resultado corregido que incluye `equity_over_time`
VZ.plot_equity_balance(best_results[best_agent_idx])

# Graficar las recompensas y pérdidas del mejor agente
VZ.plot_rewards_and_losses(episodes, rewards_per_episode[best_agent_idx], losses_per_episode[best_agent_idx])

# Graficar la evolución del drawdown del mejor agente
drawdowns_best_agent = drawdowns_per_episode[best_agent_idx]
plt.figure(figsize=(10, 6))
plt.plot(range(1, episodes + 1), drawdowns_best_agent, label="Drawdown", color="purple")
plt.title("Drawdown Máximo por Episodio - Mejor Agente")
plt.xlabel("Episodios")
plt.ylabel("Drawdown Máximo")
plt.grid(True)
plt.show()

print(f"Entrenamiento completado después de {episodes} episodios para {num_agents} agentes.")
