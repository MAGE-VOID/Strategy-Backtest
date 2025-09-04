from __future__ import annotations

import pandas as pd


class SignalBuilder:
    @staticmethod
    def build(df: pd.DataFrame, dates, symbols, signal_cls):
        signal_gens = {}
        local_idx_map = {}
        for sym in symbols:
            grp = df[df["Symbol"] == sym]
            signal_gens[sym] = signal_cls(grp)
            local_idx_map[sym] = {dt: idx for idx, dt in enumerate(grp.index)}
        return signal_gens, local_idx_map

