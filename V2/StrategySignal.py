import numpy as np


class StrategySignal:
    def __init__(self, InputData, manipulation_threshold=0.8, manipulation_weight=0.8):
        self.InputData = InputData
        self.required_data_length = 1000
        self.manipulation_threshold = (
            manipulation_threshold  # Factor para definir manipulación
        )
        self.manipulation_weight = (
            manipulation_weight  # Peso reducido para velas manipuladas
        )

    def is_manipulated(self, price_diff_pct, mean_diff_pct):
        # Detecta si una vela es manipulada comparando el % de cambio con el promedio
        return abs(price_diff_pct) > self.manipulation_threshold * mean_diff_pct

    def generate_signals(self):
        open_prices = self.InputData["Open"].values

        # Si no hay suficientes datos, retornamos arrays vacíos
        if len(open_prices) < self.required_data_length:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

        # Inicialización de los arrays de señales con valores en falso
        signal_buy = np.zeros(len(open_prices), dtype=bool)
        signal_sell = np.zeros(len(open_prices), dtype=bool)

        # Variables para sumar los porcentajes de subidas y bajadas
        sum_rise_pct = 0
        sum_fall_pct = 0

        # Lista para almacenar los porcentajes de cambio para detectar promedio
        price_diffs = []

        # Iteramos sobre las velas en intervalos de 10 (por ejemplo 0-10, 10-20, etc.)
        for i in range(0, self.required_data_length - 10, 10):
            # Calculamos la diferencia porcentual entre vela i y la vela i + 10
            price_diff_pct = (
                (open_prices[i + 10] - open_prices[i]) / open_prices[i]
            ) * 100
            price_diffs.append(abs(price_diff_pct))

        # Calculamos el promedio de las diferencias porcentuales (para detectar manipulación)
        mean_diff_pct = np.mean(price_diffs)

        # Iteramos nuevamente para generar las señales, pero ahora considerando manipulación
        for i in range(0, self.required_data_length - 10, 10):
            # Calculamos la diferencia porcentual entre vela actual y la que está 10 velas adelante
            price_diff_pct = (
                (open_prices[i + 10] - open_prices[i]) / open_prices[i]
            ) * 100

            # Verificamos si esta vela es manipulada
            if self.is_manipulated(price_diff_pct, mean_diff_pct):
                price_diff_pct *= (
                    self.manipulation_weight
                )  # Reducimos el impacto si es manipulada

            # Acumulamos las diferencias porcentuales ajustadas
            if price_diff_pct > 0:
                sum_rise_pct += price_diff_pct
                # Señal de compra si la diferencia es positiva
                signal_buy[i : i + 10] = True
            elif price_diff_pct < 0:
                sum_fall_pct += abs(price_diff_pct)
                # Señal de venta si la diferencia es negativa
                signal_sell[i : i + 10] = True

        # Decisión final basada en la suma de diferencias porcentuales
        if sum_rise_pct > sum_fall_pct:
            # Si la suma de subidas es mayor, señal de compra en todo el rango
            signal_buy[:] = True
        else:
            # Si la suma de bajadas es mayor, señal de venta en todo el rango
            signal_sell[:] = True

        signal_buy[:] = True
        signal_sell[:] = True

        # Retornamos las señales de compra y venta
        return signal_buy, signal_sell
