# main.py
from datetime import datetime
import MetaTrader5 as mt5
from data import custom as data
from backtest.engine import BacktestEngine
from backtest.config import BacktestConfig
from backtest.utils.formatters import format_statistics
from visualization.plot import plot_equity_balance
from strategies.signals import StrategySignal

if __name__ == "__main__":
    # Conexión a MT5
    data.connect_and_login_mt5(51344621, "ICMarketsSC-Demo", "bCFNLB9k")

    inp_start_date = datetime(2010, 1, 1)
    inp_end_date = datetime(2024, 12, 31)
    timeframe = mt5.TIMEFRAME_M5
    symbols = [
        "EURCHF",
        # "EURUSD",
        # "GBPUSD",
    ]

    df, df_standardized = data.process_data(
        inp_start_date, inp_end_date, symbols, timeframe
    )

    # Crear la configuración del backtest
    config = BacktestConfig(
        initial_balance=1000,
        strategy_name="grid_buy",
        strategy_signal_class=StrategySignal,
        debug_mode="none",  # opciones: "none", "final", "realtime"
    )

    engine = BacktestEngine(config)
    result_backtest = engine.run_backtest(df)

    stats_df = format_statistics(result_backtest["statistics"])
    print("\n--- Backtest Statistics ---")
    print(stats_df)

    plot_equity_balance(result_backtest)
