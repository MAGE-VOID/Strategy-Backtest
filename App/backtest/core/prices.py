from __future__ import annotations

import numpy as np
import pandas as pd


class PriceMatrixBuilder:
    @staticmethod
    def build(df: pd.DataFrame, dates, symbols):
        n_steps, n_sym = len(dates), len(symbols)
        mats = {k: np.full((n_steps, n_sym), np.nan, dtype=float) for k in ("open", "high", "low", "close")}
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym].sort_index()
            for fld in ("Open", "High", "Low", "Close"):
                s = grp[fld].reindex(dates)
                mats[fld.lower()][:, j] = s.to_numpy(dtype=float)
        return mats

