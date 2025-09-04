from __future__ import annotations

import numpy as np
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
            if "Digits" in sub.columns:
                digits_series = sub["Digits"].dropna()
                digits = int(digits_series.iat[0]) if not digits_series.empty else None
            else:
                digits = None
            mapping[sym] = {
                "point": point,
                "point_value": point_value,
                "digits": digits,
            }
        return mapping

