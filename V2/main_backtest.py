import cProfile
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5
import CustomLibrary as CL
import Backtest_Engine as BT
import visualize as VZ
import StrategySignal as Signal

# Conectar a la cuenta de trading
CL.connect_and_login_mt5(51344621, "ICMarketsSC-Demo", "bCFNLB9k")

# Definir fechas de inicio y fin
inp_start_date = datetime(2010, 1, 1)
inp_end_date = datetime(2024, 12, 31)
timeframe = mt5.TIMEFRAME_M5  # M1, M5, M10, M15, M30, H1, H4, D1, W1

# Definir los símbolos
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

# Descargar y procesar los datos
df, df_standardized = CL.process_data(inp_start_date, inp_end_date, symbols, timeframe)

# Crear instancia del motor de backtest
bt_engine = BT.BacktestEngine(initial_balance=1000)

# Usar cProfile para medir el rendimiento del backtest
profiler = cProfile.Profile()
profiler.enable()

# Ejecutar el backtest
resultBacktest = bt_engine.run_backtest(
    df, strategy_name="grid_buy", strategy_signal_class=Signal.StrategySignal
)

profiler.disable()

# Guardar el resultado del perfilado en un archivo
profiler.dump_stats("backtest_profile.prof")

# Obtener y mostrar las estadísticas del backtest
statistics = resultBacktest["statistics"]
stats_df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
stats_df["Value"] = stats_df["Value"].apply(
    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
)

# Mostrar las estadísticas
print("\n--- Backtest Statistics ---")
print(stats_df)

# Graficar el balance y la equidad
VZ.plot_equity_balance(resultBacktest)
