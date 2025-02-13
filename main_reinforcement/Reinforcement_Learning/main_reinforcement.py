import numpy as np
import tensorflow as tf
import gym
import random
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime
from collections import deque
import pandas as pd
import time
from Library import Process_Data
from visualize import plot_trades, plot_training_progress
from train_agent import create_model, train_agent
import tf2onnx
import os

# Configurar TensorFlow para usar solo el 50% de la VRAM disponible
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)],
            )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(
            f"{len(gpus)} GPU física(s), {len(logical_gpus)} GPU lógica(s) disponibles"
        )
    except RuntimeError as e:
        print(e)

# Inicialización de MetaTrader5 y obtención de datos históricos
inp_start_date = datetime(2019, 1, 1)
inp_end_date = datetime(2019, 5, 31)
symbols = ["EURUSD"]  # Puedes agregar los símbolos que quieras

df, df_standardized, df_manual_standardized = Process_Data(
    inp_start_date, inp_end_date, symbols, mt5.TIMEFRAME_H4
)

# Definir el entorno de Gym personalizado
class TradingEnv(gym.Env):
    def __init__(self, df, symbol="EURUSD"):
        super(TradingEnv, self).__init__()
        self.window_size = 50
        self.df = df[df["Symbol"] == symbol].copy()
        self.dates = self.df.index
        self.prices = self.df["Close"].values
        self.current_step = self.window_size
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 1), dtype=np.float32
        )
        self.position = None
        self.position_action = None
        self.position_step = 0
        self.max_steps = len(self.prices) - 1

    def reset(self):
        self.current_step = self.window_size
        self.position = None
        self.position_action = None
        self.position_step = 0
        return self._get_observation()

    def _get_observation(self):
        prices = self.prices[self.current_step - self.window_size : self.current_step]
        mean = prices.mean()
        std = prices.std()
        if std == 0:
            std = 1  # Para evitar división por cero
        observation = (prices - mean) / std
        observation = observation.reshape(-1, 1)  # Agrega dimensión de características
        return observation

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        done = False

        # Acción 0: Abrir una nueva posición (comprar)
        if action == 0:
            if self.position is None:
                self.position = current_price
                self.position_step = 1
            else:
                # Penalizar si intenta abrir una posición cuando ya hay una abierta
                reward -= 1  # Penalización menor

        # Acción 1: Cerrar la posición
        elif action == 1:
            if self.position is not None:
                profit = current_price - self.position
                reward += profit * 100
                self.position = None
                self.position_step = 0
            else:
                # Penalizar si intenta cerrar una posición cuando no hay ninguna abierta
                reward -= 1  # Penalización menor

        # Penalización por mantener una posición abierta (opcional)
        if self.position is not None and action != 1:
            reward -= 0.01  # Pequeña penalización por cada paso con posición abierta

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Actualizar la observación
        observation = self._get_observation()

        return observation, reward, done, {}

# Crear el entorno
env = TradingEnv(df, symbol="EURUSD")

# Crear los modelos
input_dim = env.observation_space.shape[0]
model = create_model(input_dim=input_dim, output_dim=2, learning_rate=0.001)
target_model = create_model(input_dim=input_dim, output_dim=2, learning_rate=0.001)
target_model.set_weights(model.get_weights())

# Ejecutar el entrenamiento y obtener las operaciones registradas
total_rewards, total_losses, best_trades_log = train_agent(env, model, target_model)

# Mostrar gráfico de las operaciones realizadas
plot_trades(
    df[df["Symbol"] == "EURUSD"],
    best_trades_log,
    symbol="EURUSD",
)
plot_training_progress(total_rewards, total_losses)

# **Guardar el modelo en formato ONNX**
# Definir la ruta de salida
output_folder = "D:/FOREX/MT5_1/MQL5/Files"
inp_model_name = "model.reinforcement.learning.onnx"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_path = os.path.join(output_folder, inp_model_name)

# Eliminar el archivo existente si existe
if os.path.exists(output_path):
    os.remove(output_path)

# **Definir la firma de entrada**
spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)

# **Convertir el modelo Keras a ONNX**
import tf2onnx
model_proto, _ = tf2onnx.convert.from_keras(
    model, input_signature=spec, opset=13, output_path=output_path
)

print(f"Modelo guardado en {output_path}")
