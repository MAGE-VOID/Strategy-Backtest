# strategies/signals.py
class StrategySignal:
    def __init__(self, input_data):
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values
        self.volumes = input_data["Volume"].values

        # Parámetros de optimización
        self.optimization_params = {
            "params_1": {"start": None, "step": None, "stop": None, "type": "bool"},
            "params_2": {"start": None, "step": None, "stop": None, "type": "bool"},
        }

        self.optimized_params = {key: 0.0 for key in self.optimization_params.keys()}

    # Establece los parámetros optimizados recibidos.
    def set_optimized_params(self, params):
        self.optimized_params.update(params)

    def generate_signals_for_candle(self, index: int):
        if index < 100:
            return False, False

        signal_buy = self.optimized_params.get("params_1")
        signal_sell = self.optimized_params.get("params_2")

        #signal_buy = True
        #signal_sell = True

        return signal_buy, signal_sell
