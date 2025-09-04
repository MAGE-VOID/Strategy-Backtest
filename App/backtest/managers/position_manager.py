# backtest/managers/position_manager.py

from datetime import datetime
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from backtest.utils.normalization import (
    normalize_lot as util_normalize_lot,
    normalize_price as util_normalize_price,
)


class PositionManager:
    def __init__(self, balance, symbol_points_mapping=None):
        self.positions = {}
        self.balance = float(
            Decimal(balance).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
        self.ticket_counter = 0
        self.global_counter = 0
        self.results = []

        # metadata
        self.symbol_points_mapping = symbol_points_mapping or {}
        self.sym2idx = {}

    # -------- Normalizadores internos -------- #
    def _normalize_lot(self, lot: float) -> float:
        return util_normalize_lot(lot)

    def _normalize_price(self, symbol: str, price):
        return util_normalize_price(symbol, price, self.symbol_points_mapping)

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
        tick = meta.get("point_value")
        digits = meta.get("digits")
        direction = 1 if position_type == "long" else -1

        # Normalizaciones
        lot_size = self._normalize_lot(lot_size)
        entry_price = self._normalize_price(symbol, price)
        sl = self._normalize_price(symbol, sl)
        tp = self._normalize_price(symbol, tp)

        pos = {
            "symbol": symbol,
            "position": position_type,
            "entry_price": entry_price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "open_time": open_time,  # para reporte
            "open_dt": open_dt,  # para lógica
            "status": "open",
            "sym_idx": sym_idx,
            "point": point,
            "tick": tick,  # dinero por 1 punto y 1 lote
            "dir": direction,
            "magic": magic,
            "digits": digits,
        }
        self.positions[self.ticket_counter] = pos

        # log de apertura
        result = {
            "count": self.global_counter + 1,
            "symbol": symbol,
            "ticket": self.ticket_counter,
            "type": position_type,
            "entry": entry_price,
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

        symbol = position["symbol"]
        current_price = self._normalize_price(symbol, current_price)
        point = position["point"]
        tick = position["tick"]

        # Profit realizado por posición (independiente)
        if position["position"] == "long":
            diff = current_price - position["entry_price"]
        else:
            diff = position["entry_price"] - current_price

        # Seguridad: si falta tick/point, no afectar balance
        if not point or not tick:
            profit = 0.0
        else:
            raw_profit = (diff / point) * tick * position["lot_size"]
            profit = float(
                Decimal(raw_profit).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )

        # Balance con 2 decimales
        new_balance = float(
            Decimal(self.balance + profit).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )
        self.balance = new_balance

        result = {
            "count": self.global_counter + 1,
            "symbol": symbol,
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
        symbol = pos["symbol"]
        if tp is not None:
            pos["tp"] = self._normalize_price(symbol, tp)
        if sl is not None:
            pos["sl"] = self._normalize_price(symbol, sl)

        # reflejar en el log de apertura
        for rec in reversed(self.results):
            if rec.get("ticket") == ticket and rec.get("status") == "open":
                if tp is not None:
                    rec["tp"] = pos["tp"]
                if sl is not None:
                    rec["sl"] = pos["sl"]
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
