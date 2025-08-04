# strategies/signals.py

import numpy as np


class StrategySignal:
    def __init__(self, input_data, variant: int = 0):
        self.variant = variant
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values

    def generate_signals_for_candle(self, index: int):

        signal_buy = True
        signal_sell = True
    
        if self.variant == 0:
            return signal_buy, signal_sell
        if self.variant == 1:
            return signal_buy, signal_sell
