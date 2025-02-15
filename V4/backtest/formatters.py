# backtest/formatters.py
import pandas as pd

def format_statistics(statistics: dict) -> pd.DataFrame:
    """
    Formatea un diccionario de estadísticas en un DataFrame.
    Los valores numéricos se formatean con dos decimales y separador de miles.
    
    Parameters:
        statistics (dict): Diccionario de estadísticas a formatear.
        
    Returns:
        pd.DataFrame: DataFrame con el índice nombrado como 'Statistic' y una columna 'Value'
                      con los valores formateados.
    """
    if not statistics:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
    df.index.name = "Statistic"
    
    def format_value(val):
        if isinstance(val, (int, float)):
            try:
                return f"{val:,.2f}"
            except Exception:
                return val
        return val
    
    df["Value"] = df["Value"].apply(format_value)
    return df
