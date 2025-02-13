import MetaTrader5 as mt5
import numpy as np


class EntryManagerRL:
    def __init__(self, position_managers, strategies_params=None):
        """
        Modificado para aceptar una lista de `position_managers`, uno por agente.
        """
        self.position_managers = position_managers
        self.strategies_params = strategies_params or {}
        self.symbol_points = {}
        self.first_position_tp = [{} for _ in position_managers]  # Para cada agente
        self.last_position_price = [{} for _ in position_managers]  # Para cada agente
        self.grid_positions = [{} for _ in position_managers]  # Para cada agente
        self.lot_multiplier = 1.35
        self.initial_lot_size = 10.0

    def get_symbol_points(self, symbol):
        """Obtiene los puntos del símbolo de manera eficiente."""
        if symbol not in self.symbol_points:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Symbol {symbol} not found.")
            self.symbol_points[symbol] = symbol_info.point
        return self.symbol_points[symbol]

    def apply_rl_strategy(self, symbol, action, current_price, date, agent_idx):
        """
        Aplica la estrategia controlada por el agente de Reinforcement Learning (RL).
        El agente envía la acción (0 o 1), donde 1 indica comprar.
        Cada agente se identifica por `agent_idx`.
        """
        point = self.get_symbol_points(symbol)

        if action == 1:  # Acción de compra
            self.grid_buy(symbol, current_price, date, agent_idx)

    def grid_buy(self, symbol, current_price, date, agent_idx):
        """Estrategia de compra con grid para un agente específico."""
        long_positions = self.position_managers[agent_idx].get_positions(symbol, "long")

        if not long_positions:
            self.open_initial_grid_position(symbol, current_price, date, agent_idx)
        else:
            last_price = self.last_position_price[agent_idx].get(symbol, None)
            grid_distance = self.strategies_params.get(
                "grid_distance", 100
            ) * self.get_symbol_points(symbol)

            # Si el precio actual es lo suficientemente bajo en relación con la última posición
            if last_price is None or current_price <= last_price - grid_distance:
                self.add_grid_position(symbol, current_price, date, agent_idx)

    def open_initial_grid_position(self, symbol, current_price, date, agent_idx):
        """Abre la primera posición del grid para un agente específico."""
        tp_distance = self.strategies_params.get("tp_distance", 100)
        tp = current_price + tp_distance * self.get_symbol_points(symbol)
        self.first_position_tp[agent_idx][symbol] = tp
        self.last_position_price[agent_idx][symbol] = current_price
        self.grid_positions[agent_idx][symbol] = 1
        self.position_managers[agent_idx].open_position(
            symbol,
            "long",
            current_price,
            self.initial_lot_size,
            sl=None,
            tp=tp,
            open_date=date,
        )

    def add_grid_position(self, symbol, current_price, date, agent_idx):
        """Agrega una posición al grid para un agente específico."""
        # Lote incrementado en función del número de posiciones anteriores
        lot_size = self.initial_lot_size * (
            self.lot_multiplier ** self.grid_positions[agent_idx][symbol]
        )
        self.position_managers[agent_idx].open_position(
            symbol,
            "long",
            current_price,
            lot_size,
            sl=None,
            tp=self.first_position_tp[agent_idx][symbol],
            open_date=date,
        )

        # Actualizar el precio de la última posición abierta y el número de posiciones del grid
        self.last_position_price[agent_idx][symbol] = current_price
        self.grid_positions[agent_idx][symbol] += 1

        # Actualizar el TP en base al promedio de posiciones abiertas
        self.update_tp(symbol, agent_idx)

    def update_tp(self, symbol, agent_idx):
        """Actualiza el Take Profit basado en el promedio de las posiciones abiertas para un agente específico."""
        long_positions = self.position_managers[agent_idx].get_positions(symbol, "long")
        if not long_positions:
            return

        # Cálculo eficiente del precio promedio ponderado
        entry_prices = np.array([pos["entry_price"] for pos in long_positions.values()])
        lot_sizes = np.array([pos["lot_size"] for pos in long_positions.values()])
        total_lots = lot_sizes.sum()

        if total_lots == 0:
            return

        average_price = np.dot(entry_prices, lot_sizes) / total_lots

        # Calcular el nuevo TP basado en el promedio más la distancia
        tp_distance = self.strategies_params.get("tp_distance", 100)
        new_tp = average_price + tp_distance * self.get_symbol_points(symbol)
        self.first_position_tp[agent_idx][symbol] = new_tp

        # Actualizar el TP de todas las posiciones abiertas
        for position in long_positions.values():
            position["tp"] = new_tp

    def manage_tp_sl(self, symbol, current_price, date, agent_idx):
        """Gestiona el cierre de posiciones cuando el precio alcanza el TP para un agente específico."""
        if (
            symbol in self.first_position_tp[agent_idx]
            and current_price >= self.first_position_tp[agent_idx][symbol]
        ):
            positions = self.position_managers[agent_idx].get_positions(symbol)
            for ticket, position in positions.items():
                if position["position"] == "long":
                    self.position_managers[agent_idx].close_position(ticket, current_price, date)

            # Limpiar datos del grid para el símbolo
            self.clear_symbol_data(symbol, agent_idx)

    def clear_symbol_data(self, symbol, agent_idx):
        """Limpia los datos almacenados de un símbolo después de cerrar todas las posiciones para un agente específico."""
        self.first_position_tp[agent_idx].pop(symbol, None)
        self.last_position_price[agent_idx].pop(symbol, None)
        self.grid_positions[agent_idx].pop(symbol, None)
