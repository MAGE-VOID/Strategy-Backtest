# strategies/signals.py
class StrategySignal:
    def __init__(self, input_data):
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values
        self.volumes = input_data["Volume"].values

    def generate_signals_for_candle(self, index: int):

        signal_buy = True
        signal_sell = True

        return signal_buy, signal_sell
