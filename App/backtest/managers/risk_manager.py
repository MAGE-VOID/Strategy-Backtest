from __future__ import annotations

from typing import Optional, Set, Tuple, List
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback when numba unavailable
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator




@njit(cache=True)
def _compute_exits_for_subset(
    sym_idx_arr,
    dir_arr,
    tp_arr,
    sl_arr,
    point_arr,
    open_row,
    low_row,
    high_row,
    close_row,
    spread_points,
):
    n = len(sym_idx_arr)
    exit_flags = np.zeros(n, dtype=np.int8)
    exit_prices = np.full(n, np.nan, dtype=np.float64)

    for k in range(n):
        j = sym_idx_arr[k]
        if j < 0:
            continue
        open_bid = open_row[j]
        low_bid = low_row[j]
        high_bid = high_row[j]
        close_bid = close_row[j]
        if not (
            np.isfinite(open_bid)
            and np.isfinite(low_bid)
            and np.isfinite(high_bid)
            and np.isfinite(close_bid)
        ):
            continue

        point = point_arr[k]
        if not np.isfinite(point) or point <= 0.0:
            continue
        tp = tp_arr[k]
        sl = sl_arr[k]
        sp_move = spread_points * point
        bullish = close_bid >= open_bid
        direction = dir_arr[k]

        if direction > 0:  # long
            if np.isfinite(tp) and open_bid >= tp:
                exit_price = tp
            elif np.isfinite(sl) and open_bid <= sl:
                exit_price = open_bid
            else:
                hit_tp = (np.isfinite(tp)) and (high_bid >= tp)
                hit_sl = (np.isfinite(sl)) and (low_bid <= sl)
                if not (hit_tp or hit_sl):
                    continue
                if hit_tp and hit_sl:
                    exit_price = sl if bullish else tp
                elif hit_tp:
                    exit_price = tp
                else:
                    exit_price = sl
        elif direction < 0:  # short
            open_ask = open_bid + sp_move
            low_ask = low_bid + sp_move
            high_ask = high_bid + sp_move
            if np.isfinite(tp) and open_ask <= tp:
                exit_price = tp
            elif np.isfinite(sl) and open_ask >= sl:
                exit_price = open_ask
            else:
                hit_tp = (np.isfinite(tp)) and (low_ask <= tp)
                hit_sl = (np.isfinite(sl)) and (high_ask >= sl)
                if not (hit_tp or hit_sl):
                    continue
                if hit_tp and hit_sl:
                    exit_price = tp if bullish else sl
                elif hit_tp:
                    exit_price = tp
                else:
                    exit_price = sl
        else:
            continue

        exit_flags[k] = 1
        exit_prices[k] = exit_price

    return exit_flags, exit_prices


class RiskManager:
    """
    TP/SL handling with realistic OHLC path and spread:
      - BUY: TP/SL trigger on BID
      - SELL: TP/SL trigger on ASK = BID + spread

    One-call, two-phase option:
      - Phase 1: close positions opened before the current bar time
      - Phase 2: close positions opened on the current bar time
    """

    def __init__(self, entry_manager, position_manager) -> None:
        self.em = entry_manager
        self.pm = position_manager

    def check_tp_sl(
        self,
        opens,
        lows,
        highs,
        closes,
        date,
        only_opened_before: Optional[object] = None,
        only_opened_on: Optional[object] = None,
    ) -> None:
        """
        Evaluate TP/SL.

        Modes:
          - Filtered (compat): if `only_opened_before` or `only_opened_on` is set,
            process only that subset.
          - Two-phase optimized: if both filters are None, do a single call that
            first closes < date and then == date using one scan partition.
        """

        def _process_subset(items_iter):
            items: List[Tuple[int, dict]] = list(items_iter)
            if not items:
                return set()

            tickets = []
            symbols_local = []
            magics_local = []
            sym_idx_arr = []
            dir_arr = []
            tp_arr = []
            sl_arr = []
            point_arr = []

            for ticket, pos in items:
                idx = pos.get("sym_idx")
                if idx is None:
                    continue
                tickets.append(ticket)
                symbols_local.append(pos.get("symbol"))
                magics_local.append(pos.get("magic"))
                sym_idx_arr.append(int(idx))
                dir_arr.append(float(pos.get("dir") or 0.0))
                tp = pos.get("tp")
                sl = pos.get("sl")
                tp_arr.append(np.nan if tp is None else float(tp))
                sl_arr.append(np.nan if sl is None else float(sl))
                point_arr.append(float(pos.get("point") or 0.0))

            if not tickets:
                return set()

            sym_idx_np = np.asarray(sym_idx_arr, dtype=np.int64)
            dir_np = np.asarray(dir_arr, dtype=np.float64)
            tp_np = np.asarray(tp_arr, dtype=np.float64)
            sl_np = np.asarray(sl_arr, dtype=np.float64)
            point_np = np.asarray(point_arr, dtype=np.float64)

            spread_points = float(self.em.spread_points or 0.0)
            flags, prices = _compute_exits_for_subset(
                sym_idx_np,
                dir_np,
                tp_np,
                sl_np,
                point_np,
                opens,
                lows,
                highs,
                closes,
                spread_points,
            )

            closed_pairs_local: Set[Tuple[str, int]] = set()
            for k, ticket in enumerate(tickets):
                if int(flags[k]) != 1:
                    continue
                raw_exit = float(prices[k])
                if not np.isfinite(raw_exit):
                    continue
                symbol = symbols_local[k]
                exit_price = self.em.normalize_price(symbol, raw_exit)
                self.pm.close_position(ticket, exit_price, date)
                # pair (symbol, magic)
                closed_pairs_local.add((symbol, magics_local[k]))
            return closed_pairs_local

        # ---------- Filtered mode (legacy-compatible) ---------- #
        if (only_opened_before is not None) or (only_opened_on is not None):
            def _should_process(pos) -> bool:
                odt = pos.get("open_dt")
                if only_opened_before is not None:
                    return (odt is None) or (odt < only_opened_before)
                if only_opened_on is not None:
                    return odt == only_opened_on
                return True

            # construir subset filtrado y procesarlo vectorizado
            filtered_items: List[Tuple[int, dict]] = []
            for ticket, pos in list(self.pm.positions.items()):
                if _should_process(pos):
                    filtered_items.append((ticket, pos))

            closed_pairs: Set[Tuple[str, int]] = _process_subset(filtered_items)

            for sym, mag in closed_pairs:
                still_open = any(
                    p for p in self.pm.positions.values() if p["symbol"] == sym and p.get("magic") == mag
                )
                if not still_open:
                    self.em.clear_symbol_data(sym, mag)
            return

        # ---------- Two-phase optimized (single call) ---------- #
        items = list(self.pm.positions.items())
        old_items = []
        new_items = []
        for ticket, pos in items:
            odt = pos.get("open_dt")
            if (odt is None) or (odt < date):
                old_items.append((ticket, pos))
            elif odt == date:
                new_items.append((ticket, pos))

        closed_pairs_total: Set[Tuple[str, int]] = set()
        closed_pairs_total |= _process_subset(old_items)
        closed_pairs_total |= _process_subset(new_items)

        for sym, mag in closed_pairs_total:
            still_open = any(
                p for p in self.pm.positions.values() if p["symbol"] == sym and p.get("magic") == mag
            )
            if not still_open:
                self.em.clear_symbol_data(sym, mag)
