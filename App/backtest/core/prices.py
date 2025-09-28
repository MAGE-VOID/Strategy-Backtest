from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback when numba missing
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _fill_ohlc_matrices(
    date_idx,
    sym_idx,
    open_values,
    high_values,
    low_values,
    close_values,
    open_mat,
    high_mat,
    low_mat,
    close_mat,
):
    for k in range(len(date_idx)):
        i = date_idx[k]
        j = sym_idx[k]
        if i < 0 or j < 0:
            continue
        open_mat[i, j] = open_values[k]
        high_mat[i, j] = high_values[k]
        low_mat[i, j] = low_values[k]
        close_mat[i, j] = close_values[k]


class PriceMatrixBuilder:
    @staticmethod
    def build(df: pd.DataFrame, dates, symbols):
        n_steps, n_sym = len(dates), len(symbols)
        open_mat = np.full((n_steps, n_sym), np.nan, dtype=np.float64)
        high_mat = np.full((n_steps, n_sym), np.nan, dtype=np.float64)
        low_mat = np.full((n_steps, n_sym), np.nan, dtype=np.float64)
        close_mat = np.full((n_steps, n_sym), np.nan, dtype=np.float64)

        if n_steps == 0 or n_sym == 0 or df.empty:
            return {"open": open_mat, "high": high_mat, "low": low_mat, "close": close_mat}

        date_index = pd.Index(dates)
        sym_index = pd.Index(symbols)

        date_idx = date_index.get_indexer(df.index)
        sym_idx = sym_index.get_indexer(df["Symbol"].to_numpy())

        valid_rows = (date_idx >= 0) & (sym_idx >= 0)
        if not np.all(valid_rows):
            date_idx = date_idx[valid_rows]
            sym_idx = sym_idx[valid_rows]
            mask: slice | np.ndarray = valid_rows  # type: ignore[assignment]
        else:
            mask = slice(None)

        open_vals = df["Open"].to_numpy(dtype=np.float64)[mask]
        high_vals = df["High"].to_numpy(dtype=np.float64)[mask]
        low_vals = df["Low"].to_numpy(dtype=np.float64)[mask]
        close_vals = df["Close"].to_numpy(dtype=np.float64)[mask]

        _fill_ohlc_matrices(
            date_idx,
            sym_idx,
            open_vals,
            high_vals,
            low_vals,
            close_vals,
            open_mat,
            high_mat,
            low_mat,
            close_mat,
        )

        return {"open": open_mat, "high": high_mat, "low": low_mat, "close": close_mat}
