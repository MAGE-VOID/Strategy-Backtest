# strategies/signals.py
import numpy as np

class StrategySignal:
    """
    Genera señales para cada vela utilizando arrays precalculados.
    Permite la optimización al recibir parámetros opcionales que se almacenan internamente.
    """
    def __init__(self, input_data, optimization_params=None):
        self.optimization_params = optimization_params or {}
        self.required_data_length = self.optimization_params.get("required_data_length", 1000)
        
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values
        self.volumes = input_data["Volume"].values

    def generate_signals_for_candle(self, index: int):
        if index < self.required_data_length:
            return False, False
        signal_buy = self.close_prices[index] > self.open_prices[index]
        signal_sell = self.close_prices[index] < self.open_prices[index]
        return signal_buy, signal_sell
