from __future__ import annotations

import numpy as np
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
        # Precompute per-symbol DataFrame once
        symbol_groups = {}
        # Build fast global date indexer once (vectorized mapping)
        date_index = pd.Index(dates)
        for sym in symbols:
            grp = df[df["Symbol"] == sym]
            symbol_groups[sym] = grp
            # Build a NumPy array of local indices aligned to global dates
            # For each row in grp (local_idx), find its position in global dates
            # and assign local_idx; otherwise keep -1
            if grp.empty:
                local_idx_map[sym] = np.full(len(dates), -1, dtype=np.int64)
            else:
                # positions of each grp.index within the global date_index
                global_pos = date_index.get_indexer(grp.index)
                local_pos = np.arange(len(grp.index), dtype=np.int64)
                arr = np.full(len(dates), -1, dtype=np.int64)
                mask = global_pos >= 0
                if np.any(mask):
                    arr[global_pos[mask]] = local_pos[mask]
                local_idx_map[sym] = arr

        for strat_name, signal_cls in strategy_signal_classes.items():
            signal_gens[strat_name] = {}
            for sym, grp in symbol_groups.items():
                signal_gens[strat_name][sym] = signal_cls(grp)

        return signal_gens, local_idx_map
