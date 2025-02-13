import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5
import CustomLibrary as CL
import Backtest_Engine as BT
import visualize as VZ
import StrategySignal as Signal


# Connect to trading account specifying the Number, Server and Password
CL.connect_and_login_mt5(51344621, "ICMarketsSC-Demo", "bCFNLB9k")


# Uso de la función
inp_start_date = datetime(2010, 1, 1)
inp_end_date = datetime(2024, 12, 31)
timeframe = mt5.TIMEFRAME_M5  # M1, M5, M10, M15, M30, H1, H4, D1, W1
# Definir símbolos
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

df, df_standardized = CL.process_data(inp_start_date, inp_end_date, symbols, timeframe)

bt_engine = BT.BacktestEngine(initial_balance=1000)
resultBacktest = bt_engine.run_backtest(
    df_standardized, strategy_name="grid_buy", strategy_signal_class=Signal.StrategySignal
)

# Obtener las estadísticas del backtest
statistics = resultBacktest["statistics"]
stats_df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
stats_df["Value"] = stats_df["Value"].apply(
    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
)
print("\n--- Backtest Statistics ---")
print(stats_df)

# Llamar a la función para graficar equity y balance
VZ.plot_equity_balance(resultBacktest)

# Graficar las señales para un símbolo específico
# VZ.plot_price_with_signals(resultBacktest, df, "EURUSD")
