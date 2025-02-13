import numpy as np


class StrategySignal:
    def __init__(self, InputData):
        self.InputData = InputData

    def generate_signals(self):
        open_prices = self.InputData["Open"].values

        # Inicialización de los arrays de señales con valores en falso
        signal_buy = np.zeros(len(open_prices), dtype=bool)
        signal_sell = np.zeros(len(open_prices), dtype=bool)

        signal_buy[:] = True
        signal_sell[:] = True

        # Retornamos las señales de compra y venta
        return signal_buy, signal_sell
