import numpy as np


class StrategySignal:
    def __init__(self, InputData):
        self.InputData = InputData
        self.required_data_length = 1000

    def generate_signals(self):
        open_prices = self.InputData["Open"].values

        # Si no hay suficientes datos, retornamos arrays vacíos
        if len(open_prices) < self.required_data_length:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

        # Inicialización de los arrays de señales con valores en falso
        signal_buy = np.zeros(len(open_prices), dtype=bool)
        signal_sell = np.zeros(len(open_prices), dtype=bool)

        signal_buy[:] = True
        signal_sell[:] = True

        # Retornamos las señales de compra y venta
        return signal_buy, signal_sell
