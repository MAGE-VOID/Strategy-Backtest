from datetime import datetime
import numpy as np


class PositionManager:
    def __init__(self, balance):
        self.positions = {}
        self.balance = balance
        self.ticket_counter = 0
        self.global_counter = 0
        self.results = []

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

        profit = (
            (current_price - position["entry_price"]) * position["lot_size"]
            if position["position"] == "long"
            else (position["entry_price"] - current_price) * position["lot_size"]
        )
        self.balance += profit

        self.results.append(
            {
                "count": self.global_counter + 1,
                "symbol": position["symbol"],
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

    def get_results(self):
        return self.results

    def get_balance(self):
        return self.balance
