import numpy as np
import tensorflow as tf
from collections import deque
import random

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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = (
            state_size  # Tamaño de la ventana (50 velas con las features necesarias)
        )
        self.action_size = action_size  # 2 acciones: comprar o no hacer nada
        self.memory = deque(maxlen=2000)  # Memoria para replay
        self.gamma = 0.95  # Factor de descuento
        self.epsilon = 1.0  # Tasa de exploración inicial
        self.epsilon_min = (
            0.01  # Mínimo epsilon (para asegurarse de que siempre explora un poco)
        )
        self.epsilon_decay = (
            0.98  # Decaimiento del epsilon (para ir reduciendo la exploración)
        )
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.prev_drawdown_peak_intensity = None
        self.prev_drawdown_depth_std = None

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation="relu")
        )
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        # Almacenar las experiencias
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Seleccionar una acción basado en el estado (usando epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Selecciona la acción con mayor valor Q

    def replay(self, batch_size):
        # Entrenar la red neuronal usando replay memory
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def calculate_reward(self, statistics):
        """
        Calcula la recompensa basada en el balance y recompensa la reducción en drawdown_peak_intensity,
        drawdown_depth_std, y maximal equity drawdown. Penaliza si estos valores son altos.
        """
        # Recompensa base: la ganancia obtenida (balance final - balance inicial)
        reward = statistics["final_balance"] - statistics["initial_balance"]

        # Penalización basada en drawdown_peak_intensity, drawdown_depth_std, y maximal equity drawdown
        penalty_drawdown_peak_intensity = statistics["drawdown_peak_intensity"]
        penalty_drawdown_depth_std = statistics["drawdown_depth_std"]
        penalty_max_drawdown = statistics["equity_maximal_drawdown"]

        # Aumentar el peso de los factores de penalización y recompensa para drawdowns
        drawdown_penalty_factor = (
            0.005  # Factor de penalización incrementado para drawdowns
        )
        drawdown_reward_factor = (
            0.005  # Factor de recompensa por reducción de drawdowns
        )

        # Penalización total por drawdowns
        total_penalty = drawdown_penalty_factor * (
            penalty_drawdown_peak_intensity
            + penalty_drawdown_depth_std
            + penalty_max_drawdown
        )

        # Comparar con episodios anteriores para recompensar la reducción de drawdowns
        reward_for_drawdown_improvement = 0

        if (
            self.prev_drawdown_peak_intensity is not None
            and self.prev_drawdown_depth_std is not None
        ):
            # Si el drawdown_peak_intensity ha mejorado (disminuido)
            if penalty_drawdown_peak_intensity < self.prev_drawdown_peak_intensity:
                improvement = (
                    self.prev_drawdown_peak_intensity - penalty_drawdown_peak_intensity
                )
                reward_for_drawdown_improvement += drawdown_reward_factor * improvement

            # Si el drawdown_depth_std ha mejorado (disminuido)
            if penalty_drawdown_depth_std < self.prev_drawdown_depth_std:
                improvement = self.prev_drawdown_depth_std - penalty_drawdown_depth_std
                reward_for_drawdown_improvement += drawdown_reward_factor * improvement

            # Si el maximal equity drawdown ha mejorado (disminuido)
            if penalty_max_drawdown < self.prev_max_drawdown:
                improvement = self.prev_max_drawdown - penalty_max_drawdown
                reward_for_drawdown_improvement += drawdown_reward_factor * improvement

        # Actualizar las métricas para el próximo episodio
        self.prev_drawdown_peak_intensity = penalty_drawdown_peak_intensity
        self.prev_drawdown_depth_std = penalty_drawdown_depth_std
        self.prev_max_drawdown = penalty_max_drawdown

        # Recompensa ajustada
        adjusted_reward = reward - total_penalty + reward_for_drawdown_improvement

        return adjusted_reward
