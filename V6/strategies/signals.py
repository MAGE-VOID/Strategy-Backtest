class StrategySignal:
    def __init__(self, input_data):
        self.required_data_length = 1000
        self.open_prices = input_data["Open"].values
        self.high_prices = input_data["High"].values
        self.low_prices = input_data["Low"].values
        self.close_prices = input_data["Close"].values
        self.volumes = input_data["Volume"].values

        # Parámetros de optimización definidos dentro de la clase
        self.optimization_params = {
            "params_1": {
                "start": 0.0,
                "step": 0.01,
                "stop": 0.02,
            },
            "params_2": {
                "start": 0.0,
                "step": 0.01,
                "stop": 0.02,
            },
        }

        # Inicialización de los parámetros optimizados
        self.engine_buy = 0.0
        self.engine_sell = 0.0

    def get_optimization_params(self):
        return self.optimization_params

    def set_optimized_params(self, params):
        # Establece los parámetros optimizados recibidos para usarlos en el backtest
        self.engine_buy = params["params_1"]
        self.engine_sell = params["params_2"]

    def generate_signals_for_candle(self, index: int):
        if index < self.required_data_length:
            return False, False

        # Calculamos la diferencia entre el open de la vela actual y el open de la vela anterior
        open_difference = (
            (self.open_prices[index] - self.open_prices[index - 1])
            / self.open_prices[index - 1]
            * 100
        )

        signal_buy = open_difference > self.engine_buy
        signal_sell = open_difference < -self.engine_sell

        return signal_buy, signal_sell
