# backtest/managers/position_manager.py

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from backtest.utils.normalization import (
    normalize_lot as util_normalize_lot,
    normalize_price as util_normalize_price,
)


class PositionManager:
    def __init__(self, balance, symbol_points_mapping=None, commission_per_lot_side: float = 0.0):
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
        self.commission_per_lot_side = float(commission_per_lot_side or 0.0)

    # -------- Normalizadores internos -------- #
    def _normalize_lot(self, lot: float) -> float:
        return util_normalize_lot(lot)

    def _normalize_price(self, symbol: str, price):
        return util_normalize_price(symbol, price, self.symbol_points_mapping)

    # -------- Utilidades P/L en dólares -------- #
    @staticmethod
    def _calc_pl_usd(entry_price: float, exit_price: float, point: float, point_value: float, lot_size: float, direction: int) -> float:
        """
        Calcula P/L bruto en dólares para una posición individual.

        - direction: +1 para long, -1 para short.
        - entry_price: precio de entrada (BUY=Ask, SELL=Bid).
        - exit_price: precio de salida (BUY=Bid, SELL=Ask, o nivel TP/SL correspondiente).
        - point: tamaño de punto del símbolo.
        - point_value: valor en dinero por 1 punto y 1 lote.
        - lot_size: tamaño de la posición en lotes.
        """
        if not point or not point_value or not lot_size:
            return 0.0
        diff = direction * (exit_price - entry_price)
        raw = (diff / point) * point_value * lot_size
        return float(Decimal(raw).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

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
        tick_profit = meta.get("point_value_profit", tick)
        tick_loss = meta.get("point_value_loss", tick)
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
            "tick_profit": tick_profit,
            "tick_loss": tick_loss,
            "dir": direction,
            "magic": magic,
            "digits": digits,
            "commission_per_lot_side": self.commission_per_lot_side,
        }
        self.positions[self.ticket_counter] = pos

        # Comisión en apertura (por lado)
        commission_open = float(
            Decimal(self.commission_per_lot_side * lot_size).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        ) if self.commission_per_lot_side else 0.0
        if commission_open:
            # Cargo inmediato a balance
            self.balance = float(
                Decimal(self.balance - commission_open).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            )

        # log de apertura
        # Estimar P/L en dólares a TP/SL (bruto), según niveles actuales
        tp_pl_usd = None
        sl_pl_usd = None
        try:
            if tp is not None and point and tick:
                # elegir valor por punto según signo esperado
                sign_tp = direction * (float(tp) - entry_price)
                pv = tick_profit if sign_tp > 0 else tick_loss
                tp_pl_usd = self._calc_pl_usd(entry_price, float(tp), point, float(pv), lot_size, direction)
            if sl is not None and point and tick:
                sign_sl = direction * (float(sl) - entry_price)
                pv = tick_profit if sign_sl > 0 else tick_loss
                sl_pl_usd = self._calc_pl_usd(entry_price, float(sl), point, float(pv), lot_size, direction)
        except Exception:
            tp_pl_usd = tp_pl_usd if tp_pl_usd is not None else 0.0
            sl_pl_usd = sl_pl_usd if sl_pl_usd is not None else 0.0

        tp_points = None if tp is None or not point else int(round(abs((entry_price - float(tp)) / point)))
        sl_points = None if sl is None or not point else int(round(abs((entry_price - float(sl)) / point)))

        result = {
            "count": self.global_counter + 1,
            "symbol": symbol,
            "ticket": self.ticket_counter,
            "type": position_type,
            "entry": entry_price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "tp_points": tp_points,
            "sl_points": sl_points,
            "tp_pl_usd": tp_pl_usd,
            "sl_pl_usd": sl_pl_usd,
            "open_time": open_time,
            "status": "open",
            "sym_idx": sym_idx,
            "point": point,
            "tick": tick,
            "dir": direction,
            "magic": magic,
            "commission_open": commission_open,
            "commission": commission_open,
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
        tick_profit = position.get("tick_profit") or position.get("tick")
        tick_loss = position.get("tick_loss") or position.get("tick")
        tick = position.get("tick") or (tick_profit if tick_profit else tick_loss)

        # Profit realizado por posición (independiente)
        if position["position"] == "long":
            diff = current_price - position["entry_price"]
        else:
            diff = position["entry_price"] - current_price

        # Seguridad: si falta tick/point, no afectar balance
        if not point or (not tick_profit and not tick_loss):
            profit_gross = 0.0
        else:
            per_point_value = float(tick_profit) if diff >= 0 else float(tick_loss)
            raw_profit = (diff / point) * per_point_value * position["lot_size"]
            profit_gross = float(
                Decimal(raw_profit).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )

        # Comisión en cierre (por lado)
        commission_close = float(
            Decimal(self.commission_per_lot_side * position["lot_size"]).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        ) if self.commission_per_lot_side else 0.0

        # Comisión en apertura (mismo cálculo por lote guardado en open)
        commission_open_equiv = float(
            Decimal(self.commission_per_lot_side * position["lot_size"]).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        ) if self.commission_per_lot_side else 0.0

        # Profit que impacta el balance en el cierre (ya se debitó la apertura)
        profit_after_close_comm = float(
            Decimal(profit_gross - commission_close).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )

        # Profit neto por trade (incluye comisiones de apertura y cierre)
        profit_net = float(
            Decimal(profit_gross - commission_open_equiv - commission_close).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )

        # Balance con 2 decimales
        new_balance = float(
            Decimal(self.balance + profit_after_close_comm).quantize(
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
            "profit_gross": profit_gross,
            "profit": profit_after_close_comm,  # compat: neto tras comisión de cierre
            "profit_net": profit_net,           # neto tras comisiones open+close
            "balance": self.balance,
            "status": "closed",
            "sym_idx": position.get("sym_idx"),
            "point": point,
            "tick": tick,
            "dir": position.get("dir"),
            "magic": position.get("magic"),
            "commission_close": commission_close,
            "commission_open_equiv": commission_open_equiv,
            "commission": commission_close,
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
                # recalcular P/L estimado a TP/SL (bruto)
                try:
                    point = float(pos.get("point") or 0.0)
                    tick_profit = float(pos.get("tick_profit") or pos.get("tick") or 0.0)
                    tick_loss = float(pos.get("tick_loss") or pos.get("tick") or 0.0)
                    lot = float(pos.get("lot_size") or 0.0)
                    direction = int(pos.get("dir") or 0)
                    entry = float(pos.get("entry_price") or 0.0)
                    if rec.get("tp") is not None and point and lot and direction:
                        exit_tp = float(rec["tp"])
                        sign_tp = direction * (exit_tp - entry)
                        pv = tick_profit if sign_tp > 0 else tick_loss
                        rec["tp_pl_usd"] = self._calc_pl_usd(entry, exit_tp, point, pv, lot, direction)
                        rec["tp_points"] = int(round(abs((entry - exit_tp) / point)))
                    if rec.get("sl") is not None and point and lot and direction:
                        exit_sl = float(rec["sl"])
                        sign_sl = direction * (exit_sl - entry)
                        pv = tick_profit if sign_sl > 0 else tick_loss
                        rec["sl_pl_usd"] = self._calc_pl_usd(entry, exit_sl, point, pv, lot, direction)
                        rec["sl_points"] = int(round(abs((entry - exit_sl) / point)))
                except Exception:
                    pass
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
