import numpy as np


class StrategySignal:
    def __init__(self, InputData):
        # Pre-calculemos los arrays para acelerar cada llamada
        self.required_data_length = 1000  # Número de velas necesarias para el cálculo
        self.open_prices = InputData["Open"].values
        self.high_prices = InputData["High"].values
        self.low_prices = InputData["Low"].values
        self.close_prices = InputData["Close"].values
        self.volumes = InputData["Volume"].values

    def generate_signals_for_candle(self, index):
        """
        Calcula las señales de compra y venta para una vela específica
        utilizando los arrays precalculados.
        """
        if index < self.required_data_length:
            return False, False

        # Ejemplo de cálculo simple de señales:
        # Compra si la vela es alcista (Close > Open)
        signal_buy = self.close_prices[index] > self.open_prices[index]
        # Venta si la vela es bajista (Close < Open)
        signal_sell = self.close_prices[index] < self.open_prices[index]

        return signal_buy, signal_sell
