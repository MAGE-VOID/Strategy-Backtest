# backtest/config.py
class BacktestConfig:

    def __init__(
        self,
        initial_balance=1000,
        debug_mode="none",
        strategies_params=None,
        spread_points: int = 2,
    ):
        self.initial_balance = initial_balance
        self.debug_mode = debug_mode  # Opciones: "none", "final", "realtime"
        self.strategies_params = strategies_params or {}
        # Spread en puntos para ASK = BID + spread_points*point
        self.spread_points = int(spread_points)
