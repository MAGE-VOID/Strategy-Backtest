# backtest/managers/entry_manager.py

import numpy as np
from backtest.managers.position_manager import PositionManager


class EntryManager:
    def __init__(
        self, initial_balance, strategies_params=None, symbol_points_mapping=None
    ):
        self.position_manager = PositionManager(initial_balance, symbol_points_mapping)
        self.strategies_params = strategies_params or {}
        self.symbol_points_mapping = symbol_points_mapping or {}

        # Estado para grid
        self.first_position_tp = {}
        self.first_position_sl = {}
        self.last_position_price = {}
        self.grid_positions = {}

        # Parámetros de estrategia
        self.tp_distance = self.strategies_params.get("tp_distance", 100)
        self.sl_distance = self.strategies_params.get("sl_distance", None)
        self.grid_distance = self.strategies_params.get("grid_distance", 100)
        self.lot_multiplier = self.strategies_params.get("lot_multiplier", 1.35)
        self.initial_lot_size = self.strategies_params.get("initial_lot_size", 0.01)

    def get_symbol_data(self, symbol):
        if symbol not in self.symbol_points_mapping:
            raise ValueError(f"Symbol {symbol} not found in provided mapping.")
        d = self.symbol_points_mapping[symbol]
        return d["point"], d["tick_value"]

    def get_symbol_points(self, symbol):
        point, _ = self.get_symbol_data(symbol)
        return point

    def calculate_tp_sl(self, current_price, symbol, is_buy=True):
        point, _ = self.get_symbol_data(symbol)
        tp_dist = self.tp_distance * point
        sl_dist = self.sl_distance * point if self.sl_distance is not None else None

        if is_buy:
            tp = current_price + tp_dist
            sl = current_price - sl_dist if sl_dist is not None else None
        else:
            tp = current_price - tp_dist
            sl = current_price + sl_dist if sl_dist is not None else None

        return tp, sl

    def calculate_lot_size(self, symbol):
        return self.initial_lot_size * (
            self.lot_multiplier ** self.grid_positions.get(symbol, 0)
        )

    def update_symbol_data(self, symbol, current_price, tp):
        """
        Inicializa los contadores de grid en la primera posición.
        """
        self.first_position_tp[symbol] = tp
        self.last_position_price[symbol] = current_price
        self.grid_positions[symbol] = 1

    def apply_strategy(
        self, strategy_name, symbol, signal_buy, signal_sell, current_price, index, date
    ):
        if strategy_name == "simple_buy":
            self._execute_trade(symbol, signal_buy, current_price, date, is_buy=True)
        elif strategy_name == "simple_sell":
            self._execute_trade(symbol, signal_sell, current_price, date, is_buy=False)
        elif strategy_name == "grid_buy" and signal_buy:
            self.grid_buy(symbol, current_price, date)

    def _execute_trade(self, symbol, signal, current_price, date, is_buy=True):
        if signal and not self.get_positions(symbol, "long" if is_buy else "short"):
            self._open_position(
                symbol, current_price, date, is_buy, self.initial_lot_size
            )

    def _open_position(self, symbol, current_price, date, is_buy=True, lot_size=None):
        tp, sl = self.calculate_tp_sl(current_price, symbol, is_buy)
        pos_type = "long" if is_buy else "short"
        self.position_manager.open_position(
            symbol, pos_type, current_price, lot_size, sl=sl, tp=tp, open_date=date
        )

    def grid_buy(self, symbol, current_price, date):
        point = self.get_symbol_points(symbol)
        long_positions = self.get_positions(symbol, "long")

        if not long_positions:
            tp = current_price + self.tp_distance * point
            self._open_position(
                symbol, current_price, date, is_buy=True, lot_size=self.initial_lot_size
            )
            self.update_symbol_data(symbol, current_price, tp)
        else:
            last_price = self.last_position_price[symbol]
            if current_price <= last_price - self.grid_distance * point:
                self._add_grid_position(symbol, current_price)

    def _add_grid_position(self, symbol, current_price):
        lot_size = self.calculate_lot_size(symbol)
        # Abrimos nueva posición sin fecha (se hereda None)
        self._open_position(symbol, current_price, None, is_buy=True, lot_size=lot_size)
        self.last_position_price[symbol] = current_price
        self.grid_positions[symbol] += 1
        self.update_tp_sl(symbol)

    def update_tp_sl(self, symbol):
        """
        Recalcula TP/SL de todas las posiciones abiertas de un símbolo tras un nuevo grid.
        """
        long_positions = self.get_positions(symbol, "long")
        if not long_positions:
            return

        prices = np.array([pos["entry_price"] for pos in long_positions.values()])
        lots = np.array([pos["lot_size"] for pos in long_positions.values()])
        total = lots.sum()
        if total == 0:
            return

        avg_price = np.dot(prices, lots) / total
        point = self.get_symbol_points(symbol)

        new_tp = avg_price + self.tp_distance * point
        new_sl = (
            avg_price - self.sl_distance * point
            if self.sl_distance is not None
            else None
        )

        self.first_position_tp[symbol] = new_tp
        if new_sl is not None:
            self.first_position_sl[symbol] = new_sl

        self.position_manager.update_symbol_tp_sl(symbol, tp=new_tp, sl=new_sl)

    def clear_symbol_data(self, symbol):
        """
        Resetea todos los estados de grid para un símbolo sin posiciones abiertas.
        """
        for d in (
            self.first_position_tp,
            self.first_position_sl,
            self.last_position_price,
            self.grid_positions,
        ):
            d.pop(symbol, None)

    def get_positions(self, symbol=None, position_type=None):
        all_pos = self.position_manager.positions
        # Si no filtramos por símbolo, devolvemos todo (incluidos distintos magics)
        if symbol is None:
            return all_pos

        # Magic actual
        current_magic = self.position_manager.default_magic

        return {
            ticket: pos
            for ticket, pos in all_pos.items()
            if pos["symbol"] == symbol
            and (current_magic is None or pos.get("magic") == current_magic)
        }
