# strategies/signals.py
import numpy as np


class StrategySignal:
    """
    Genera señales para cada vela utilizando arrays precalculados.
    Permite la optimización al recibir parámetros opcionales que se almacenan internamente.
    Los parámetros de optimización por defecto se definen en `default_optimization_params`.
    """

    default_optimization_params = {
        # Se optimiza el parámetro "required_data_length" desde 1000 hasta 2000 en pasos de 500.
        "required_data_length": (1000, 500, 2000),
        # Se pueden definir otros parámetros por defecto:
        # "otro_param": (valor_inicial, paso, valor_final),
        # "flag_param": [True, False]
    }

    def __init__(self, input_data, optimization_params=None):
        # Si se pasan parámetros, se combinan con los por defecto.
        if optimization_params is None:
            self.optimization_params = self.default_optimization_params.copy()
        else:
            # Los parámetros pasados tienen prioridad sobre los por defecto.
            self.optimization_params = {
                **self.default_optimization_params,
                **optimization_params,
            }

        # Extraer el parámetro a optimizar
        self.required_data_length = self.optimization_params.get(
            "required_data_length", 1000
        )
        # Si es una tupla, usamos el primer valor como required_data_length
        if isinstance(self.required_data_length, tuple):
            self.required_data_length = self.required_data_length[0]

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
