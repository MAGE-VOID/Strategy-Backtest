from __future__ import annotations

import pandas as pd


class SignalBuilder:
    @staticmethod
    def build_multi(df: pd.DataFrame, dates, symbols, strategy_signal_classes: dict):
        """
        Build signal generators per strategy and per symbol.
        - strategy_signal_classes: dict[strategy_name] -> SignalClass
        Returns:
          signal_gens: dict[strategy_name][symbol] -> signal_instance
          local_idx_map: dict[symbol][date] -> local_index
        """
        signal_gens = {}
        local_idx_map = {}
        # Precompute local index map per symbol once
        symbol_groups = {}
        for sym in symbols:
            grp = df[df["Symbol"] == sym]
            symbol_groups[sym] = grp
            local_idx_map[sym] = {dt: idx for idx, dt in enumerate(grp.index)}

        for strat_name, signal_cls in strategy_signal_classes.items():
            signal_gens[strat_name] = {}
            for sym, grp in symbol_groups.items():
                signal_gens[strat_name][sym] = signal_cls(grp)

        return signal_gens, local_idx_map
