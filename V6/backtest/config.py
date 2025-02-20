# backtest/config.py
class BacktestConfig:
    """
    Configuración central para el backtest.
    Centraliza parámetros como el balance inicial, el nombre de la estrategia,
    la clase de señales y el modo de depuración.
    """

    def __init__(
        self,
        initial_balance=1000,
        strategy_name="grid_buy",
        strategy_signal_class=None,
        debug_mode="none",
        mode="single",
    ):
        self.initial_balance = initial_balance
        self.strategy_name = strategy_name
        self.strategy_signal_class = strategy_signal_class
        self.debug_mode = debug_mode  # Opciones: "none", "final", "realtime"
        self.mode = mode  # "single" o "optimization"
