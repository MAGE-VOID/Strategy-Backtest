# backtest/managers/position_manager.py

from datetime import datetime
import numpy as np


class PositionManager:
    def __init__(self, balance, symbol_points_mapping=None):
        self.positions = {}
        self.balance = balance
        self.ticket_counter = 0
        self.global_counter = 0
        self.results = []
        self.symbol_points_mapping = symbol_points_mapping or {}

    def open_position(
        self, symbol, position_type, price, lot_size, sl=None, tp=None, open_date=None
    ):
        open_time = (
            open_date.strftime("%Y-%m-%d %H:%M:%S")
            if open_date
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self.positions[self.ticket_counter] = {
            "symbol": symbol,
            "position": position_type,
            "entry_price": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "open_time": open_time,
            "status": "open",
        }

        self.results.append(
            {
                "count": self.global_counter + 1,
                "symbol": symbol,
                "ticket": self.ticket_counter,
                "type": position_type,
                "entry": price,
                "lot_size": lot_size,
                "sl": sl,
                "tp": tp,
                "open_time": open_time,
                "status": "open",
            }
        )
        self.ticket_counter += 1
        self.global_counter += 1

    def close_position(self, ticket, current_price, close_date):
        position = self.positions.pop(ticket, None)
        if position is None:
            return

        symbol = position["symbol"]
        point = self.symbol_points_mapping[symbol]["point"]
        tick_value = self.symbol_points_mapping[symbol]["tick_value"]

        if position["position"] == "long":
            price_diff = current_price - position["entry_price"]
        else:  # short
            price_diff = position["entry_price"] - current_price

        profit = (price_diff / point) * tick_value * position["lot_size"]
        self.balance += profit

        self.results.append(
            {
                "count": self.global_counter + 1,
                "symbol": symbol,
                "ticket": ticket,
                "type": position["position"],
                "entry": position["entry_price"],
                "close_time": close_date.strftime("%Y-%m-%d %H:%M:%S"),
                "exit": current_price,
                "profit": profit,
                "balance": self.balance,
                "status": "closed",
            }
        )
        self.global_counter += 1

    def update_position_tp_sl(self, ticket, tp=None, sl=None):
        """
        Actualiza TP y/o SL de una posición abierta (en memoria y en results).
        """
        pos = self.positions.get(ticket)
        if not pos:
            return
        if tp is not None:
            pos["tp"] = tp
        if sl is not None:
            pos["sl"] = sl

        # También ajustamos el entry record en results
        for res in reversed(self.results):
            if res.get("ticket") == ticket and res.get("status") == "open":
                if tp is not None:
                    res["tp"] = tp
                if sl is not None:
                    res["sl"] = sl
                break

    def update_symbol_tp_sl(self, symbol, tp=None, sl=None):
        """
        Actualiza TP/SL de todas las posiciones abiertas para un símbolo.
        """
        for ticket, pos in self.positions.items():
            if pos["symbol"] == symbol:
                self.update_position_tp_sl(ticket, tp, sl)

    def get_results(self):
        return self.results

    def get_balance(self):
        return self.balance
