# main.py
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

from data import custom as data
from backtest.engine import BacktestEngine
from backtest.formatters import format_statistics  # Nueva importación
from visualization.plot import plot_equity_balance
from strategies.signals import StrategySignal

# Conexión a MT5
data.connect_and_login_mt5(51344621, "ICMarketsSC-Demo", "bCFNLB9k")

inp_start_date = datetime(2010, 1, 1)
inp_end_date = datetime(2024, 12, 31)
timeframe = mt5.TIMEFRAME_M5
symbols = [
    "AUDNZD",
    "AUDUSD",
    "CADCHF",
    "GBPUSD",
    "EURCHF",
    "EURGBP",
    "GBPCAD",
    "NZDUSD",
    "USDCAD",
]

df, df_standardized = data.process_data(
    inp_start_date, inp_end_date, symbols, timeframe
)

engine = BacktestEngine(initial_balance=1000)
result_backtest = engine.run_backtest(
    df, strategy_name="grid_buy", strategy_signal_class=StrategySignal
)

stats_df = format_statistics(result_backtest["statistics"])
print("\n--- Backtest Statistics ---")
print(stats_df)

plot_equity_balance(result_backtest)
