# backtest/formatters.py
import pandas as pd
from datetime import datetime, timedelta


def format_statistics(statistics: dict) -> pd.DataFrame:
    """
    Formatea un diccionario de estadísticas en un DataFrame.
    - Números: con dos decimales y separador de miles.
    - Fechas: en formato "YYYY-MM-DD HH:MM:SS".
    - Duraciones (timedelta): en "X days HH:MM:SS".
    - Booleanos: como "Yes" o "No".

    Parameters:
        statistics (dict): Diccionario de estadísticas a formatear.

    Returns:
        pd.DataFrame: DataFrame con el índice nombrado como 'Statistic' y una columna 'Value'
                      con los valores formateados.
    """
    if not statistics:
        return pd.DataFrame()

    def format_value(val):
        if isinstance(val, (int, float)):
            try:
                return f"{val:,.2f}"
            except Exception:
                return val
        elif isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(val, timedelta):
            total_seconds = int(val.total_seconds())
            days, remainder = divmod(total_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{days} days {hours:02d}:{minutes:02d}:{seconds:02d}"
        elif isinstance(val, bool):
            return "Yes" if val else "No"
        else:
            return val

    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
    df.index.name = "Statistic"
    df["Value"] = df["Value"].apply(format_value)
    return df
