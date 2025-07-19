# backtest/config.py
class BacktestConfig:

    def __init__(
        self,
        initial_balance=1000,
        strategy_name="grid_buy",
        strategy_signal_class=None,
        debug_mode="none",
    ):
        self.initial_balance = initial_balance
        self.strategy_name = strategy_name
        self.strategy_signal_class = strategy_signal_class
        self.debug_mode = debug_mode  # Opciones: "none", "final", "realtime"
