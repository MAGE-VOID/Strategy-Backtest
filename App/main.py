# main.py
from datetime import datetime
import os
import MetaTrader5 as mt5
from data import custom as data
from backtest.core.engine import BacktestEngine
from backtest.config import BacktestConfig
from backtest.utils.formatters import format_statistics
from strategies.signals import StrategySignal_1, StrategySignal_2
from visualization.plot import BacktestPlotter

if __name__ == "__main__":
    # Conexión a MT5
    account = int(os.getenv("MT5_ACCOUNT", "51344621"))
    server = os.getenv("MT5_SERVER", "ICMarketsSC-Demo")
    password = os.getenv("MT5_PASSWORD", "bCFNLB9k")
    data.connect_and_login_mt5(account, server, password)

    inp_start_date = datetime(2010, 1, 1)
    inp_end_date = datetime(2024, 12, 31)
    timeframe = mt5.TIMEFRAME_M5
    symbols = [
        # "EURCHF",
        "EURUSD",
        # "GBPUSD",
    ]

    df, _ = data.process_data(
        inp_start_date, inp_end_date, symbols, timeframe
    )

    config = BacktestConfig(
        initial_balance=1000,
        debug_mode="none",  # opciones: "none", "final", "realtime"
        strategies_params={
            "grid_buy": {  # simple_buy simple_sell grid_buy grid_sell
                "strategy_signal_class": StrategySignal_1,
                "tp_distance": 100,
                "sl_distance": None,
                "initial_lot_size": 0.01,
                "grid_distance": 100,
                "lot_multiplier": 1.35,
                "magic": 123456,
            },
            "grid_sell": {  # simple_buy simple_sell grid_buy grid_sell
                "strategy_signal_class": StrategySignal_2,
                "tp_distance": 100,
                "sl_distance": None,
                "initial_lot_size": 0.01,
                "grid_distance": 100,
                "lot_multiplier": 1.35,
                "magic": 234567,
            },
        },
    )

    engine = BacktestEngine(config)
    result_backtest = engine.run_backtest(df)

    stats_df = format_statistics(result_backtest["statistics"])
    print("\n--- Backtest Statistics ---")
    print(stats_df)

    BacktestPlotter().show(result_backtest)
