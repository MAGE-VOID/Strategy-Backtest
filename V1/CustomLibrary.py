import MetaTrader5 as mt5
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def account_login(login, password, server):
    if mt5.login(login, password, server):
        print("logged in succesffully")
    else:
        print("login failed, error code: {}".format(mt5.last_error()))


def initialize(login, server, password, path):

    if not mt5.initialize(path):
        print("initialize() failed, error code {}", mt5.last_error())
    else:
        account_login(login, password, server)


def LoginAccount(account, servername, password_account, print_info=False):
    # authorized = mt5.login(account, servername, password_account)
    authorized = mt5.login(account, password_account, servername)
    if authorized:
        account_info = mt5.account_info()
        terminal_info = mt5.terminal_info()
        if print_info:
            # Display trading account data in the form of a list
            print("Successfully connected to account")
            print("\nShow account_info(): ")
            account_info_dict = account_info._asdict()
            max_account_prop_len = max(len(prop) for prop in account_info_dict)

            for prop in account_info_dict:
                formatted_prop = prop.ljust(max_account_prop_len)
                print("  {} = {}".format(formatted_prop, account_info_dict[prop]))

            print("\nShow terminal_info(): ")
            terminal_info_dict = terminal_info._asdict()
            max_terminal_prop_len = max(len(prop) for prop in terminal_info_dict)

            for prop in terminal_info_dict:
                formatted_prop = prop.ljust(max_terminal_prop_len)
                print("  {} = {}".format(formatted_prop, terminal_info_dict[prop]))
        else:
            print("Successfully connected to account #{}".format(account))
    else:
        print(
            "Failed to connect to account #{}, error code: {}".format(
                account, mt5.last_error()
            )
        )


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


def filter_positions(symbol_type, magic_number, position_type):
    # Obtener la lista de posiciones
    positions = mt5.positions_get()

    # Filtrar posiciones según el tipo de símbolo, número mágico y tipo de posición
    filtered_positions = [
        pos
        for pos in positions
        if pos.symbol.startswith(symbol_type)
        and pos.magic == magic_number
        and pos.type == position_type
    ]
    return filtered_positions


def set_type_filling_by_symbol(symbol):
    # Obtener el tipo de llenado posible por símbolo
    symbol_info = mt5.symbol_info(symbol)
    mt5.symbol_select(symbol, True)
    mt5.symbol_info_tick(symbol)
    filling = symbol_info.filling_mode - 1  # Chech with other brokers
    set = -1
    if filling == mt5.ORDER_FILLING_FOK:
        set = 0
    if filling == mt5.ORDER_FILLING_IOC:
        set = 1
    if filling == mt5.ORDER_FILLING_RETURN:
        set = 2
    return set


def open_position(
    symbol, order_type, volume, slippage, sl_points, tp_points, magic, comment
):
    # Obtener información del símbolo
    symbol_info = mt5.symbol_info(symbol)
    mt5.symbol_select(symbol, True)
    mt5.symbol_info_tick(symbol)

    # Si el símbolo no está visible en MarketWatch, intentar agregarlo
    if not symbol_info.visible:
        print(symbol, "no es visible, intentando activarlo")
        if not mt5.symbol_select(symbol, True):
            print("symbol_select({}}) falló, saliendo", symbol)
            mt5.shutdown()
            return

    # Ajustar el order, price, sl y tp multiplicados por los puntos según el tipo de orden
    if order_type == "Buy":
        order_dict = 0
        price_dict = symbol_info.ask
        sl_price = symbol_info.bid - sl_points * symbol_info.point
        tp_price = symbol_info.ask + tp_points * symbol_info.point
    if order_type == "Sell":
        order_dict = 1
        price_dict = symbol_info.bid
        sl_price = symbol_info.ask + sl_points * symbol_info.point
        tp_price = symbol_info.bid - tp_points * symbol_info.point

    # Si sl_points o tp_points es igual a 0, establecer sl_price o tp_price como cadena vacía
    if sl_points == 0:
        sl_price = 0.0
    if tp_points == 0:
        tp_price = 0.0

    # Definir la estructura de la orden
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict,
        "price": price_dict,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": slippage,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": set_type_filling_by_symbol(symbol),
    }

    # Enviar la orden de trading
    result = mt5.order_send(request)

    # Crear un DataFrame a partir del resultado
    result_dict = result._asdict()
    df = pd.DataFrame([result_dict])

    # Mostrar el DataFrame
    pd.set_option("display.max_rows", None)  # Mostrar todas las filas
    pd.set_option("display.max_columns", None)  # Mostrar todas las columnas
    print(df)

    # Verificar el resultado de la ejecución de la orden
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Open Trade Executed")
        print("Ticket Number:", result.order)
    else:
        print("Error, retcode =", result.retcode)
        print("Result:", result)
    return result


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
