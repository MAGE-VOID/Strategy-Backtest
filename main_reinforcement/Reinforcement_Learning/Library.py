import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime

def Process_Data(start_date, end_date, symbols, timeframe):
    if not mt5.initialize():
        print("Initialize() failed, error code =", mt5.last_error())
        return None, None, None

    symbols_data = {}
    for symbol in symbols:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            print(f"No data for {symbol}")
            continue
        df = pd.DataFrame(rates)
        df.drop(columns=["spread", "real_volume"], inplace=True)
        df.rename(
            columns={
                "time": "date",
                "tick_volume": "Volume",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            },
            inplace=True,
        )
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df.set_index("date", inplace=True)
        df[["Open", "High", "Low", "Close", "Volume"]] = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].astype("float64")
        symbols_data[symbol] = df

    if not symbols_data:
        print("No data available for the specified symbols and date range.")
        return None, None, None

    # Concatenar los DataFrames de todos los símbolos
    df = pd.concat(
        symbols_data.values(), keys=symbols_data.keys(), names=["Symbol", "date"]
    ).reset_index(level="Symbol")

    # Copias para estandarizaciones
    df_standardized = df.copy()
    df_manual_standardized = df.copy()

    # Estandarización manual
    for symbol in symbols:
        symbol_mask = df_manual_standardized["Symbol"] == symbol
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df_symbol = df_manual_standardized.loc[symbol_mask, cols]

        # Calcular la media y desviación estándar manualmente
        mean = df_symbol.mean()
        std = df_symbol.std()

        # Estandarizar manualmente
        df_manual_standardized.loc[symbol_mask, cols] = (df_symbol - mean) / std

    # Estandarización con StandardScaler para comparar
    from sklearn.preprocessing import StandardScaler
    for symbol in symbols:
        symbol_mask = df_standardized["Symbol"] == symbol
        cols = ["Open", "High", "Low", "Close", "Volume"]
        scaler = StandardScaler()
        df_standardized.loc[symbol_mask, cols] = scaler.fit_transform(
            df_standardized.loc[symbol_mask, cols]
        )

    return df, df_standardized, df_manual_standardized
