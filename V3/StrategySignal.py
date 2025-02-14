import numpy as np


class StrategySignal:
    def __init__(self, InputData):
        # Pre-calculemos los arrays para acelerar cada llamada
        self.required_data_length = 1000  # Número de velas necesarias para el cálculo
        self.open_prices = InputData["Open"].values
        self.close_prices = InputData["Close"].values

    def generate_signals_for_candle(self, index):
        """
        Calcula las señales de compra y venta para una vela específica
        utilizando los arrays precalculados.
        """
        if index < self.required_data_length:
            return False, False

        # Ejemplo de cálculo simple de señales:
        signal_buy = (
            self.close_prices[index] > self.open_prices[index]
        )  # Compra si la vela es alcista
        signal_sell = (
            self.close_prices[index] < self.open_prices[index]
        )  # Venta si la vela es bajista

        return signal_buy, signal_sell
