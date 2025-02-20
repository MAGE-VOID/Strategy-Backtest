# strategies/signals.py
import numpy as np

class StrategySignal:
    """
    Genera señales para cada vela utilizando arrays precalculados.
    """
    def __init__(self, input_data):
        self.required_data_length = 1000  # Cuántos datos mínimos necesitamos
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values
        self.volumes = input_data["Volume"].values

    def generate_signals_for_candle(self, index: int):
        """
        Genera señales de compra y venta para una vela en particular.
        """
        if index < self.required_data_length:
            return False, False
        
        # Nueva condición: las últimas 5 barras son alcistas
        if self._is_bullish_trend(index):
            signal_buy = True
        else:
            signal_buy = False
        
        # Condición de venta: si la vela es bajista
        signal_sell = self.close_prices[index] < self.open_prices[index]
        
        return signal_buy, signal_sell

    def _is_bullish_trend(self, index: int):
        """
        Verifica si las últimas 5 velas son alcistas.
        """
        # Verificamos que no estemos fuera de los límites de datos
        if index < 5:
            return False
        
        # Revisa las últimas 5 velas: debe ser alcista cada una
        for i in range(index - 5, index):
            if self.close_prices[i] <= self.open_prices[i]:
                return False
        return True
