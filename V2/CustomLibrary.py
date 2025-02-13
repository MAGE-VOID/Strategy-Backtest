import MetaTrader5 as mt5
import pandas as pd
import numpy as np


def connect_and_login_mt5(account, server, password, print_info=False):
    """
    Initialize the MetaTrader 5 terminal (without specifying a path)
    and log in to a given account. If 'print_info' is True,
    display account and terminal information upon successful login.
    """

    # Attempt to initialize the MetaTrader 5 platform
    if not mt5.initialize():
        error_code = mt5.last_error()
        print(f"MT5 initialization failed, error code: {error_code}")
        return

    # Attempt to log in to the specified account
    if not mt5.login(account, password, server):
        error_code = mt5.last_error()
        print(f"Failed to log in to account #{account}, error code: {error_code}")
        return

    # If successfully connected, optionally display account and terminal info
    if print_info:

        # Display data on the MetaTrader 5 package
        print("MetaTrader5 package author: ", mt5.__author__)
        print("MetaTrader5 package version: ", mt5.__version__)

        account_info = mt5.account_info()
        terminal_info = mt5.terminal_info()

        if account_info is not None:
            print("Successfully connected to the account.")
            print("\nAccount Information:")
            account_info_dict = account_info._asdict()
            max_prop_len = max(len(prop) for prop in account_info_dict)
            for prop, value in account_info_dict.items():
                print(f"  {prop.ljust(max_prop_len)} = {value}")
        else:
            print("No account information available.")

        if terminal_info is not None:
            print("\nTerminal Information:")
            terminal_info_dict = terminal_info._asdict()
            max_prop_len = max(len(prop) for prop in terminal_info_dict)
            for prop, value in terminal_info_dict.items():
                print(f"  {prop.ljust(max_prop_len)} = {value}")
        else:
            print("No terminal information available.")

    else:
        print(f"Successfully connected to account #{account}")


def _fetch_symbol_data(symbol: str, timeframe, start_date, end_date) -> pd.DataFrame:
    """
    Fetches and prepares price data for a given symbol from MetaTrader 5.
    Returns a DataFrame with columns: [Symbol, Open, High, Low, Close, Volume].
    If no data is found, returns an empty DataFrame.
    """
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"No data found for symbol: {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df.drop(columns=["spread", "real_volume"], inplace=True, errors="ignore")
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
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype("float64")
    df.insert(0, "Symbol", symbol)
    return df


def _manual_standardize_numpy(
    df: pd.DataFrame, symbol_col: str = "Symbol"
) -> pd.DataFrame:
    """
    Manually standardize columns (Open, High, Low, Close, Volume) using NumPy
    for improved performance on large datasets.

    Returns a new DataFrame with standardized columns grouped by `symbol_col`.
    """
    # Columns to standardize
    cols_to_std = ["Open", "High", "Low", "Close", "Volume"]

    # Factorize symbols to get integer codes, ignoring the unique symbols array
    symbol_codes = pd.factorize(df[symbol_col])[0]

    # Convert the relevant columns to a NumPy array
    arr = df[cols_to_std].to_numpy(dtype=np.float64)

    # Prepare an array to store standardized data
    arr_std = np.empty_like(arr)

    # Standardize for each unique symbol code
    for code in np.unique(symbol_codes):
        mask = symbol_codes == code
        sub_arr = arr[mask, :]
        mean = sub_arr.mean(axis=0)
        std = sub_arr.std(axis=0, ddof=0)  # ddof=0 for population std
        arr_std[mask, :] = (sub_arr - mean) / std

    # Create a new DataFrame and put the standardized values back
    df_out = df.copy()
    df_out[cols_to_std] = arr_std
    return df_out


def process_data(start_date, end_date, symbols, timeframe):
    """
    Fetches and consolidates OHLCV data for multiple symbols into a single DataFrame.
    Also returns a manually standardized copy of that DataFrame, grouped by symbol.

    Parameters:
    -----------
    start_date : datetime
        The start date for fetching historical data.
    end_date   : datetime
        The end date for fetching historical data.
    symbols    : list[str]
        A list of symbol names (e.g., ["EURUSD", "GBPUSD"]).
    timeframe  : int
        The timeframe constant (e.g., mt5.TIMEFRAME_M5).

    Returns:
    --------
    df_original: pd.DataFrame
        Combined DataFrame of all symbols with columns: [Symbol, Open, High, Low, Close, Volume].
        The 'date' is the DataFrame index.

    df_standardized: pd.DataFrame
        A copy of df_original, but with manual standardization of the OHLCV columns per symbol
        using NumPy factorization (for potentially faster performance on large data).
    """
    all_data = []
    for symbol in symbols:
        df_symbol = _fetch_symbol_data(symbol, timeframe, start_date, end_date)
        if not df_symbol.empty:
            all_data.append(df_symbol)

    if not all_data:
        print("No data available for any specified symbol within the date range.")
        return None, None

    # Concatenate all symbol data into a single DataFrame
    df_original = pd.concat(all_data).sort_index()

    # Manually standardize using NumPy-based approach
    df_standardized = _manual_standardize_numpy(df_original)

    return df_original, df_standardized


def SymbolSync(SelectedSymbols, show_info=True):
    # Obtener la lista de todos los símbolos disponibles
    all_symbols = [symbol.name for symbol in mt5.symbols_get()]

    # Convertir la lista de símbolos seleccionados en un conjunto para una búsqueda más eficiente
    selected_symbols_set = set(SelectedSymbols.split(", "))

    # Crear una lista para mantener la información detallada de los símbolos
    symbol_details = []

    for index, symbol in enumerate(selected_symbols_set, start=1):
        if symbol in all_symbols:
            # Obtener información detallada del símbolo
            symbol_info = mt5.symbol_info(symbol)
            mt5.symbol_select(symbol, True)
            mt5.symbol_info_tick(symbol)

            # Verificar si el símbolo está sincronizado o no
            is_synced = "Yes" if symbol_info is not None else "No"

            # Agregar información a la lista
            symbol_details.append(
                {
                    "Orden": index,
                    "Símbolo": symbol,
                    "Spread": symbol_info.spread,
                    "Sincronizado": is_synced,
                    "Select": symbol_info.select,
                    "Visible": symbol_info.visible,
                    "Time": symbol_info.time,
                    "Digits": symbol_info.digits,
                    "Spread_Float": symbol_info.spread_float,
                    "Trade_Mode": symbol_info.trade_mode,
                    "Trade_ExeMode": symbol_info.trade_exemode,
                    "Swap_Mode": symbol_info.swap_mode,
                    "Swap_Rollover3days": symbol_info.swap_rollover3days,
                    "Expiration_Mode": symbol_info.expiration_mode,
                    "Filling_Mode": symbol_info.filling_mode,
                    "Order_Mode": symbol_info.order_mode,
                    "Bid": symbol_info.bid,
                    "BidHigh": symbol_info.bidhigh,
                    "BidLow": symbol_info.bidlow,
                    "Ask": symbol_info.ask,
                    "AskHigh": symbol_info.askhigh,
                    "AskLow": symbol_info.asklow,
                    "Point": symbol_info.point,
                    "Trade_Tick_Value": symbol_info.trade_tick_value,
                    "Trade_Tick_Value_Profit": symbol_info.trade_tick_value_profit,
                    "Trade_Tick_Value_Loss": symbol_info.trade_tick_value_loss,
                    "Trade_Tick_Size": symbol_info.trade_tick_size,
                    "Volume_Min": symbol_info.volume_min,
                    "Volume_Max": symbol_info.volume_max,
                    "Volume_Step": symbol_info.volume_step,
                    "Swap_Long": symbol_info.swap_long,
                    "Swap_Short": symbol_info.swap_short,
                    "Session_Open": symbol_info.session_open,
                    "Session_Close": symbol_info.session_close,
                    "Price_Change": symbol_info.price_change,
                }
            )

    # Comprobar si todos los símbolos están sincronizados
    all_synced = all(detail["Sincronizado"] == "Yes" for detail in symbol_details)

    # Mostrar el estado de sincronización si show_info es True en un dataframe
    if show_info:
        df = pd.DataFrame(symbol_details)
        pd.set_option("display.max_rows", None)  # Mostrar todas las filas
        pd.set_option("display.max_columns", None)  # Mostrar todas las columnas
        print(df)

    # Devolver la lista de símbolos sincronizados
    if all_synced:
        return list(selected_symbols_set)
    else:
        return "No están sincronizados los símbolos"
