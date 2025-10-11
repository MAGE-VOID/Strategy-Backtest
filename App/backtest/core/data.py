from __future__ import annotations

import pandas as pd


class DataPrep:
    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("No hay datos para el backtest.")
        return df.sort_index()

    @staticmethod
    def extract_index_symbols(df: pd.DataFrame):
        return df.index.unique(), df["Symbol"].unique()

    @staticmethod
    def map_symbol_points(df: pd.DataFrame, symbols):
        mapping = {}
        for sym in symbols:
            sub = df[df["Symbol"] == sym]
            point = sub["Point"].dropna().iat[0]
            point_value = sub["Point_Value"].dropna().iat[0]
            # Valores diferenciados por signo si están disponibles
            pv_profit = (
                sub["Point_Value_Profit"].dropna().iat[0]
                if "Point_Value_Profit" in sub.columns and not sub["Point_Value_Profit"].dropna().empty
                else point_value
            )
            pv_loss = (
                sub["Point_Value_Loss"].dropna().iat[0]
                if "Point_Value_Loss" in sub.columns and not sub["Point_Value_Loss"].dropna().empty
                else point_value
            )
            if "Digits" in sub.columns:
                digits_series = sub["Digits"].dropna()
                digits = int(digits_series.iat[0]) if not digits_series.empty else None
            else:
                digits = None
            mapping[sym] = {
                "point": point,
                "point_value": point_value,
                "point_value_profit": pv_profit,
                "point_value_loss": pv_loss,
                "digits": digits,
            }
        return mapping
