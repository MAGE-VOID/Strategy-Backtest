# backtest/managers/position_manager.py

from datetime import datetime
import pandas as pd


class PositionManager:
    def __init__(self, balance, symbol_points_mapping=None):
        self.positions = {}
        self.balance = balance
        self.ticket_counter = 0
        self.global_counter = 0
        self.results = []

        # metadata
        self.symbol_points_mapping = symbol_points_mapping or {}
        self.sym2idx = {}

    def open_position(
        self,
        symbol,
        position_type,
        price,
        lot_size,
        sl=None,
        tp=None,
        open_date=None,
        magic=None,
    ):
        # Normalizar fecha de apertura a pd.Timestamp para comparaciones consistentes
        if open_date is None:
            open_dt = pd.Timestamp.utcnow()
        else:
            open_dt = pd.Timestamp(open_date)  # acepta datetime, str, numpy, etc.
        open_time = open_dt.strftime("%Y-%m-%d %H:%M:%S")

        sym_idx = self.sym2idx.get(symbol)
        meta = self.symbol_points_mapping.get(symbol, {})
        point = meta.get("point")
        # point_value = dinero por 1 punto y 1 lote
        tick = meta.get("point_value")
        direction = 1 if position_type == "long" else -1

        pos = {
            "symbol": symbol,
            "position": position_type,
            "entry_price": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "open_time": open_time,  # para reporte
            "open_dt": open_dt,  # para lógica
            "status": "open",
            "sym_idx": sym_idx,
            "point": point,
            "tick": tick,
            "dir": direction,
            "magic": magic,
        }
        self.positions[self.ticket_counter] = pos

        # log de apertura
        result = {
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
            "sym_idx": sym_idx,
            "point": point,
            "tick": tick,
            "dir": direction,
            "magic": magic,
        }
        self.results.append(result)

        self.ticket_counter += 1
        self.global_counter += 1

    def close_position(self, ticket, current_price, close_date):
        position = self.positions.pop(ticket, None)
        if position is None:
            return

        point = position["point"]
        tick = position["tick"]

        # Profit realizado por posición (independiente)
        if position["position"] == "long":
            diff = current_price - position["entry_price"]
        else:
            diff = position["entry_price"] - current_price

        profit = (diff / point) * tick * position["lot_size"]
        self.balance += profit

        result = {
            "count": self.global_counter + 1,
            "symbol": position["symbol"],
            "ticket": ticket,
            "type": position["position"],
            "entry": position["entry_price"],
            "close_time": pd.Timestamp(close_date).strftime("%Y-%m-%d %H:%M:%S"),
            "exit": current_price,
            "profit": profit,
            "balance": self.balance,
            "status": "closed",
            "sym_idx": position.get("sym_idx"),
            "point": point,
            "tick": tick,
            "dir": position.get("dir"),
            "magic": position.get("magic"),
        }
        self.results.append(result)
        self.global_counter += 1

    def update_position_tp_sl(self, ticket, tp=None, sl=None):
        pos = self.positions.get(ticket)
        if not pos:
            return
        if tp is not None:
            pos["tp"] = tp
        if sl is not None:
            pos["sl"] = sl

        # reflejar en el log de apertura
        for rec in reversed(self.results):
            if rec.get("ticket") == ticket and rec.get("status") == "open":
                if tp is not None:
                    rec["tp"] = tp
                if sl is not None:
                    rec["sl"] = sl
                break

    def update_symbol_tp_sl(self, symbol, magic, tp=None, sl=None):
        """
        Actualiza TP/SL sólo para posiciones del (symbol, magic) especificado.
        """
        for ticket, pos in self.positions.items():
            if pos["symbol"] == symbol and pos.get("magic") == magic:
                self.update_position_tp_sl(ticket, tp, sl)

    def get_results(self):
        return self.results

    def get_balance(self):
        return self.balance
