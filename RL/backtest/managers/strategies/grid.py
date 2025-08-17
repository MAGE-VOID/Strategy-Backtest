# backtest/managers/strategies/grid.py
from __future__ import annotations
import numpy as np
from .base import BaseEntryStrategy
from .registry import register_strategy

# --------- GRID BUY --------- #
@register_strategy("grid_buy")
class GridBuyStrategy(BaseEntryStrategy):
    def __init__(self, name, params, context):
        super().__init__(name, params, context)
        self._last_price = {}    # (symbol, magic) -> last bid
        self._grid_count = {}    # (symbol, magic) -> int

    def clear_symbol_data(self, symbol: str, magic: int) -> None:
        key = (symbol, magic)
        self._last_price.pop(key, None)
        self._grid_count.pop(key, None)

    def on_candle(self, symbol, signal_buy, signal_sell, current_bid, date):
        if not signal_buy:
            return
        key = (symbol, self.magic)
        point = self.ctx.get_symbol_points(symbol)

        longs = [
            p for p in self.ctx.position_manager.positions.values()
            if p["symbol"] == symbol and p["position"] == "long" and p["magic"] == self.magic
        ]

        if not longs:
            self.ctx.open_position(symbol, current_bid, date, is_buy=True, params=self.params)
            self._last_price[key] = current_bid
            self._grid_count[key] = 1
        else:
            last = self._last_price.get(key)
            if last is None:
                return
            if current_bid <= last - self.params["grid_distance"] * point:
                self._add_grid_position(symbol, current_bid, date)

    def _add_grid_position(self, symbol, current_bid, date):
        key = (symbol, self.magic)
        count = self._grid_count.get(key, 0)
        lot = self.params["initial_lot_size"] * (self.params["lot_multiplier"] ** count)
        self.ctx.open_position(symbol, current_bid, date, is_buy=True, params=self.params, lot_size=lot)
        self._last_price[key] = current_bid
        self._grid_count[key] = count + 1
        self._update_group_tp_sl(symbol)

    def _update_group_tp_sl(self, symbol: str):
        """
        Recalcula TP/SL unificado de todas las posiciones LONG (symbol, magic)
        usando el precio medio ponderado por lotes (convertido a BID de disparo).
        """
        longs = [
            p for p in self.ctx.position_manager.positions.values()
            if p["symbol"] == symbol and p["position"] == "long" and p["magic"] == self.magic
        ]
        if not longs:
            return

        prices = np.array([p["entry_price"] for p in longs])  # Ask de entrada (BUY)
        lots = np.array([p["lot_size"] for p in longs])
        tot = lots.sum()
        if tot == 0:
            return

        point = self.ctx.get_symbol_points(symbol)
        spread_move = self.ctx.spread_points * point

        avg_ask = float(np.dot(prices, lots) / tot)
        bid_equiv = avg_ask - spread_move  # nivel de disparo correcto

        new_tp = bid_equiv + self.params["tp_distance"] * point
        new_sl = (
            bid_equiv - self.params["sl_distance"] * point
            if self.params["sl_distance"] is not None
            else None
        )
        self.ctx.position_manager.update_symbol_tp_sl(symbol, magic=self.magic, tp=new_tp, sl=new_sl)


# --------- GRID SELL --------- #
@register_strategy("grid_sell")
class GridSellStrategy(BaseEntryStrategy):
    def __init__(self, name, params, context):
        super().__init__(name, params, context)
        self._last_price = {}    # (symbol, magic) -> last bid
        self._grid_count = {}    # (symbol, magic) -> int

    def clear_symbol_data(self, symbol: str, magic: int) -> None:
        key = (symbol, magic)
        self._last_price.pop(key, None)
        self._grid_count.pop(key, None)

    def on_candle(self, symbol, signal_buy, signal_sell, current_bid, date):
        if not signal_sell:
            return
        key = (symbol, self.magic)
        point = self.ctx.get_symbol_points(symbol)

        shorts = [
            p for p in self.ctx.position_manager.positions.values()
            if p["symbol"] == symbol and p["position"] == "short" and p["magic"] == self.magic
        ]

        if not shorts:
            self.ctx.open_position(symbol, current_bid, date, is_buy=False, params=self.params)
            self._last_price[key] = current_bid
            self._grid_count[key] = 1
        else:
            last = self._last_price.get(key)
            if last is None:
                return
            # Para SHORT agregamos si el precio sube (movimiento adverso)
            if current_bid >= last + self.params["grid_distance"] * point:
                self._add_grid_position(symbol, current_bid, date)

    def _add_grid_position(self, symbol, current_bid, date):
        key = (symbol, self.magic)
        count = self._grid_count.get(key, 0)
        lot = self.params["initial_lot_size"] * (self.params["lot_multiplier"] ** count)
        self.ctx.open_position(symbol, current_bid, date, is_buy=False, params=self.params, lot_size=lot)
        self._last_price[key] = current_bid
        self._grid_count[key] = count + 1
        self._update_group_tp_sl(symbol)

    def _update_group_tp_sl(self, symbol: str):
        """
        Recalcula TP/SL unificado de todas las posiciones SHORT (symbol, magic)
        usando el precio medio ponderado por lotes (con disparo correcto en ASK).
        """
        shorts = [
            p for p in self.ctx.position_manager.positions.values()
            if p["symbol"] == symbol and p["position"] == "short" and p["magic"] == self.magic
        ]
        if not shorts:
            return

        prices = np.array([p["entry_price"] for p in shorts])  # Bid de entrada (SELL)
        lots = np.array([p["lot_size"] for p in shorts])
        tot = lots.sum()
        if tot == 0:
            return

        point = self.ctx.get_symbol_points(symbol)
        spread_move = self.ctx.spread_points * point

        avg_bid = float(np.dot(prices, lots) / tot)
        ask_equiv = avg_bid + spread_move  # nivel de disparo correcto (SELL dispara en ASK)

        new_tp = ask_equiv - self.params["tp_distance"] * point
        new_sl = (
            ask_equiv + self.params["sl_distance"] * point
            if self.params["sl_distance"] is not None
            else None
        )
        self.ctx.position_manager.update_symbol_tp_sl(symbol, magic=self.magic, tp=new_tp, sl=new_sl)
