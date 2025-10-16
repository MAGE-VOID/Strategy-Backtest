# backtest/utils/formatters.py
import json
import pandas as pd
from datetime import datetime, timedelta


def format_statistics(statistics: dict) -> pd.DataFrame:
    """
    Formatea un diccionario de estadísticas en un DataFrame.
    - Números: con dos decimales y separador de miles.
    - Fechas: en formato "YYYY-MM-DD HH:MM:SS".
    - Duraciones (timedelta): en "X days HH:MM:SS".
    - Booleanos: como "true" o "false".

    Parameters:
        statistics (dict): Diccionario de estadísticas a formatear.

    Returns:
        pd.DataFrame: DataFrame con el índice nombrado como 'Statistic' y una columna 'Value'
                      con los valores formateados.
    """
    if not statistics:
        return pd.DataFrame()

    def format_value(val):
        # Importante: bool es subclass de int, por eso se chequea primero
        if isinstance(val, bool):
            return "true" if val else "false"
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
        else:
            return val

    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
    df.index.name = "Statistic"
    df["Value"] = df["Value"].apply(format_value)
    return df


def _format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} days {hours:02d}:{minutes:02d}:{seconds:02d}"


def statistics_to_json_obj(statistics: dict) -> dict:
    """
    Convierte el diccionario de estadísticas a un objeto JSON-serializable.
    - datetime -> "YYYY-MM-DD HH:MM:SS"
    - timedelta -> "X days HH:MM:SS"
    - bool -> bool nativo
    - números -> float/int nativos
    """
    if not statistics:
        return {}

    def to_jsonable(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(val, timedelta):
            return _format_timedelta(val)
        # numpy types u otros con `.item()`
        try:
            item = getattr(val, "item", None)
            if callable(item):
                return item()
        except Exception:
            pass
        return val

    return {k: to_jsonable(v) for k, v in statistics.items()}


def statistics_to_json(statistics: dict) -> str:
    """Serializa estadísticas a una cadena JSON (UTF-8, sin escape ASCII)."""
    obj = statistics_to_json_obj(statistics)
    return json.dumps(obj, ensure_ascii=False)
