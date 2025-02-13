import numpy as np
import random
import tensorflow as tf
from collections import deque

def create_model(input_dim, output_dim, learning_rate=0.001):
    model = tf.keras.Sequential()
    # Cambiar la forma de entrada a (input_dim,)
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    # Añadir una dimensión extra dentro del modelo
    model.add(tf.keras.layers.Reshape((input_dim, 1)))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(output_dim, activation="linear"))
    model.compile(
        loss="huber_loss", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


def replay(memory, model, target_model, gamma, batch_size):
    if len(memory) < batch_size:
        return 0

    minibatch = random.sample(memory, batch_size)

    # Extraer y preparar los estados y próximos estados
    states = np.array([transition[0] for transition in minibatch])  # (batch_size, window_size, 1)
    actions = np.array([transition[1] for transition in minibatch])
    rewards = np.array([transition[2] for transition in minibatch])
    next_states = np.array([transition[3] for transition in minibatch])  # (batch_size, window_size, 1)
    dones = np.array([transition[4] for transition in minibatch])

    q_values = model.predict(states)
    q_values_next = target_model.predict(next_states)

    targets = q_values.copy()

    for i in range(batch_size):
        if dones[i]:
            targets[i][actions[i]] = rewards[i]
        else:
            targets[i][actions[i]] = rewards[i] + gamma * np.amax(q_values_next[i])

    # Entrenar el modelo
    history = model.fit(states, targets, epochs=1, verbose=0)
    total_loss = history.history["loss"][0]

    return total_loss

def train_agent(
    env,
    model,
    target_model,
    memory=deque(maxlen=10000),
    num_trainings=10,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.999,
    batch_size=32,
    update_target_steps=1000,
):
    best_profit = -float("inf")
    best_model_weights = model.get_weights()
    best_trades_log = []
    all_profits = []

    total_rewards = []
    total_losses = []

    trade_id = 0
    total_steps = 0

    for training in range(num_trainings):
        print(f"\nInicio del entrenamiento {training + 1}/{num_trainings}")

        actions_taken = []
        profits = []
        trades_log = []
        total_profit = 0
        episode_loss = 0

        # Restablecer el entorno para el nuevo episodio
        state = env.reset()
        state = state.reshape(1, env.observation_space.shape[0], 1)

        position_open = False
        open_trade = None

        while True:
            # Selección de acción
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_space.n)
            else:
                q_values = model.predict(state)
                action = np.argmax(q_values[0])

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, env.observation_space.shape[0], 1)
            memory.append((state[0], action, reward, next_state[0], done))
            total_profit += reward

            # Entrenar el modelo
            loss = replay(memory, model, target_model, gamma, batch_size)
            episode_loss += loss

            # Actualizar la red objetivo periódicamente
            total_steps += 1
            if total_steps % update_target_steps == 0:
                target_model.set_weights(model.get_weights())

            # Reducir epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Registrar acción tomada
            actions_taken.append(
                (env.current_step, action, env.prices[env.current_step])
            )

            # Obtener la fecha correspondiente al paso actual
            date = env.dates[env.current_step]

            # Registrar operaciones de compra/cierre con fecha
            if action == 0 and not position_open:
                trade_id += 1
                open_trade = {
                    "id": trade_id,
                    "action": "buy",
                    "step": env.current_step,
                    "date": date,
                    "price": env.prices[env.current_step],
                }
                trades_log.append(open_trade)
                position_open = True
            elif action == 1 and position_open:
                trades_log.append(
                    {
                        "id": open_trade["id"],
                        "action": "close",
                        "step": env.current_step,
                        "date": date,
                        "price": env.prices[env.current_step],
                    }
                )
                position_open = False
                open_trade = None

            # Mover al siguiente estado
            state = next_state
            profits.append(total_profit)

            if done:
                break

        # Almacenar recompensas y pérdidas
        total_rewards.append(total_profit)
        total_losses.append(episode_loss)

        # Almacenar todas las ganancias para mostrarlas al final
        all_profits.append((profits, actions_taken))

        # Guardar el mejor modelo y su trades_log si el total_profit es el mayor alcanzado
        if total_profit > best_profit:
            best_profit = total_profit
            best_model_weights = model.get_weights()
            best_trades_log = trades_log
            print(f"Nuevo mejor modelo con ganancia total de: {total_profit}")

    # Establecer el mejor modelo
    model.set_weights(best_model_weights)
    print(f"Mejor ganancia obtenida: {best_profit}")

    # Devolver solo el trades_log del mejor modelo
    return total_rewards, total_losses, best_trades_log
