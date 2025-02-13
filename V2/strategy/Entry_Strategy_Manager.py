import MetaTrader5 as mt5
from position_management.position_manager import PositionManager
import numpy as np

class EntryManager:
    def __init__(self, initial_balance, strategies_params=None):
        self.position_manager = PositionManager(initial_balance)
        self.strategies_params = strategies_params or {}
        self.symbol_points = {}

    def get_symbol_points(self, symbol):
        if symbol not in self.symbol_points:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Symbol {symbol} not found.")
            self.symbol_points[symbol] = symbol_info.point
        return self.symbol_points[symbol]

    def calculate_tp_sl(self, current_price, symbol, is_buy=True):
        point = self.get_symbol_points(symbol)
        tp_distance = self.strategies_params.get("tp_distance", 100) * point
        sl_distance = self.strategies_params.get("sl_distance", None)

        if is_buy:
            tp = current_price + tp_distance
            sl = current_price - sl_distance * point if sl_distance else None
        else:
            tp = current_price - tp_distance
            sl = current_price + sl_distance * point if sl_distance else None

        return tp, sl

    def apply_strategy(self, strategy_name, symbol, signal_buy, signal_sell, current_price, index, date):
        if strategy_name == "grid_buy":
            self.grid_buy(symbol, current_price, date)

    def grid_buy(self, symbol, current_price, date):
        long_positions = self.get_positions(symbol, "long")
        if not long_positions:
            tp = current_price + self.strategies_params["tp_distance"] * self.get_symbol_points(symbol)
            self._open_position(symbol, current_price, date, True, self.strategies_params["initial_lot_size"])

    def _open_position(self, symbol, current_price, date, is_buy, lot_size):
        tp, sl = self.calculate_tp_sl(current_price, symbol, is_buy)
        position_type = "long" if is_buy else "short"
        self.position_manager.open_position(symbol, position_type, current_price, lot_size, sl, tp, date)

    def get_positions(self, symbol=None, position_type=None):
        return self.position_manager.get_positions(symbol, position_type)
