# strategies/signals.py


class StrategySignal_1:
    """
    Señales de ejemplo: BUY si Close>Open, SELL si Close<Open.
    Mantiene input_data y arrays OHLC por claridad.
    """

    def __init__(self, input_data):
        self.input_data = input_data
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values

    def generate_signals_for_candle(self, index: int):
        # Señal simple basada en vela actual
        try:
            o = float(self.open_prices[index])
            c = float(self.close_prices[index])
        except Exception:
            return False, False
        signal_buy = bool(c > o)
        signal_sell = bool(c < o)
        return signal_buy, signal_sell


class StrategySignal_2:
    """
    Señales de ejemplo: BUY si Close>=Open y rango amplio; SELL si Open>Close y rango amplio.
    Mantiene input_data y arrays OHLC.
    """

    def __init__(self, input_data):
        self.input_data = input_data
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values

    def generate_signals_for_candle(self, index: int):
        # Ejemplo con filtro de rango (High-Low)
        try:
            o = float(self.open_prices[index])
            h = float(self.high_prices[index])
            l = float(self.low_prices[index])
            c = float(self.close_prices[index])
        except Exception:
            return False, False
        rng = h - l
        min_range = 0.0  # ajusta si deseas un filtro de rango
        buy = (c >= o) and (rng >= min_range)
        sell = (o > c) and (rng >= min_range)
        return bool(buy), bool(sell)
