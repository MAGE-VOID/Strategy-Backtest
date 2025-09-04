# backtest/managers/strategies/simple.py
from __future__ import annotations
from .base import BaseEntryStrategy
from .registry import register_strategy

@register_strategy("simple_buy")
class SimpleBuyStrategy(BaseEntryStrategy):
    def on_candle(self, symbol, signal_buy, signal_sell, current_bid, date):
        if signal_buy and not self.ctx.has_open(symbol, "long", self.magic):
            self.ctx.open_position(symbol, current_bid, date, is_buy=True, params=self.params)

@register_strategy("simple_sell")
class SimpleSellStrategy(BaseEntryStrategy):
    def on_candle(self, symbol, signal_buy, signal_sell, current_bid, date):
        if signal_sell and not self.ctx.has_open(symbol, "short", self.magic):
            self.ctx.open_position(symbol, current_bid, date, is_buy=False, params=self.params)
