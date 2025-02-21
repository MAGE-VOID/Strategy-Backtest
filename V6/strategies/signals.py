class StrategySignal:
    def __init__(self, input_data):
        self.required_data_length = 100
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values
        self.volumes = input_data["Volume"].values

        # Parámetros de optimización definidos dentro de la clase
        self.optimization_params = {
            "params_1": {"start": None, "step": None, "stop": None, "type": "bool"},
            "params_2": {"start": None, "step": None, "stop": None, "type": "bool"},
        }

        # Inicialización de los parámetros optimizados
        self.engine_buy = 0.0
        self.engine_sell = 0.0

    def get_optimization_params(self):
        return self.optimization_params

    def set_optimized_params(self, params):
        """
        Establece los parámetros recibidos para usarlos en el backtest.
        """
        self.engine_buy = params["params_1"]
        self.engine_sell = params["params_2"]

    def generate_signals_for_candle(self, index: int):
        if index < self.required_data_length:
            return False, False

        # Generación de señales basada en los parámetros optimizados
        signal_buy = self.engine_buy
        signal_sell = self.engine_sell

        signal_buy = False
        # signal_sell = False

        return signal_buy, signal_sell
