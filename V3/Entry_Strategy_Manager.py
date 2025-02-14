import MetaTrader5 as mt5
import numpy as np
import PositionManager as PS


class EntryManager:
    def __init__(self, initial_balance, strategies_params=None):
        self.position_manager = PS.PositionManager(initial_balance)
        self.strategies_params = strategies_params or {}
        self.symbol_points = {}
        self.first_position_tp = {}
        self.last_position_price = {}
        self.grid_positions = {}
        self.tp_distance = self.strategies_params.get("tp_distance", 100)
        self.grid_distance = self.strategies_params.get("grid_distance", 100)
        self.lot_multiplier = self.strategies_params.get("lot_multiplier", 1.35)
        self.initial_lot_size = self.strategies_params.get("initial_lot_size", 10.0)

    def get_symbol_points(self, symbol):
        if symbol not in self.symbol_points:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Symbol {symbol} not found.")
            self.symbol_points[symbol] = symbol_info.point
        return self.symbol_points[symbol]

    def calculate_tp_sl(self, current_price, symbol, is_buy=True):
        point = self.get_symbol_points(symbol)
        tp_distance = self.tp_distance * point
        sl_distance = self.strategies_params.get("sl_distance", None)
        if is_buy:
            tp = current_price + tp_distance
            sl = current_price - sl_distance * point if sl_distance else None
        else:
            tp = current_price - tp_distance
            sl = current_price + sl_distance * point if sl_distance else None
        return tp, sl

    def calculate_lot_size(self, symbol):
        return self.initial_lot_size * (
            self.lot_multiplier ** self.grid_positions.get(symbol, 0)
        )

    def update_symbol_data(self, symbol, current_price, tp):
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
        elif strategy_name == "grid_buy":
            if signal_buy:  # Ya es un booleano, no se indexa
                self.grid_buy(symbol, current_price, date)

    def _execute_trade(self, symbol, signal, current_price, date, is_buy=True):
        if signal and not self.get_positions(symbol, "long" if is_buy else "short"):
            self._open_position(
                symbol, current_price, date, is_buy, self.initial_lot_size
            )

    def _open_position(self, symbol, current_price, date, is_buy=True, lot_size=None):
        tp, sl = self.calculate_tp_sl(current_price, symbol, is_buy)
        position_type = "long" if is_buy else "short"
        self.position_manager.open_position(
            symbol, position_type, current_price, lot_size, sl=sl, tp=tp, open_date=date
        )

    def grid_buy(self, symbol, current_price, date):
        long_positions = self.get_positions(symbol, "long")
        if not long_positions:
            tp = current_price + self.tp_distance * self.get_symbol_points(symbol)
            self._open_position(
                symbol, current_price, date, is_buy=True, lot_size=self.initial_lot_size
            )
            self.update_symbol_data(symbol, current_price, tp)
        else:
            last_price = self.last_position_price[symbol]
            grid_distance = self.grid_distance * self.get_symbol_points(symbol)
            if current_price <= last_price - grid_distance:
                self._add_grid_position(symbol, current_price)

    def _add_grid_position(self, symbol, current_price):
        lot_size = self.calculate_lot_size(symbol)
        self._open_position(symbol, current_price, None, is_buy=True, lot_size=lot_size)
        self.last_position_price[symbol] = current_price
        self.grid_positions[symbol] += 1
        self.update_tp(symbol)

    def update_tp(self, symbol):
        long_positions = self.get_positions(symbol, "long")
        if not long_positions:
            return
        positions = list(long_positions.values())
        entry_prices = np.array([pos["entry_price"] for pos in positions])
        lot_sizes = np.array([pos["lot_size"] for pos in positions])
        total_lots = lot_sizes.sum()
        if total_lots == 0:
            return
        average_price = np.dot(entry_prices, lot_sizes) / total_lots
        new_tp = average_price + self.tp_distance * self.get_symbol_points(symbol)
        self.first_position_tp[symbol] = new_tp
        for position in positions:
            position["tp"] = new_tp


    def manage_tp_sl(self, symbol, current_price, date):
        if (
            symbol in self.first_position_tp
            and current_price >= self.first_position_tp[symbol]
        ):
            self._close_positions(symbol, current_price, date)

    def _close_positions(self, symbol, current_price, date):
        positions = self.get_positions(symbol)
        for ticket, position in positions.items():
            if position["position"] == "long":
                self.position_manager.close_position(ticket, current_price, date)
        self.clear_symbol_data(symbol)

    def clear_symbol_data(self, symbol):
        self.first_position_tp.pop(symbol, None)
        self.last_position_price.pop(symbol, None)
        self.grid_positions.pop(symbol, None)

    def get_results(self):
        return self.position_manager.results

    def get_balance(self):
        return self.position_manager.balance

    def get_positions(self, symbol=None, position_type=None):
        if symbol is None:
            return self.position_manager.positions
        return {
            ticket: pos
            for ticket, pos in self.position_manager.positions.items()
            if pos["symbol"] == symbol
            and (position_type is None or pos["position"] == position_type)
        }
