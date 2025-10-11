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
        signal_buy = True
        signal_sell = True
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
        buy = True
        sell = True
        return bool(buy), bool(sell)
