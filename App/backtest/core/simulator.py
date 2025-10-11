from __future__ import annotations

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback when numba unavailable
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

from backtest.utils.progress import BarProgress


@njit(cache=True, fastmath=True)
def _compute_equity_components(
    bid_mark_prices,
    sym_idx,
    directions,
    entry_prices,
    lot_sizes,
    points,
    ticks_profit,
    ticks_loss,
    spread_points,
):
    total_float = 0.0
    total_lots = 0.0
    open_count = 0

    for idx in range(len(sym_idx)):
        sym = sym_idx[idx]
        if sym < 0:
            continue

        bid_price = bid_mark_prices[sym]
        if not np.isfinite(bid_price):
            continue

        point = points[idx]
        tick_p = ticks_profit[idx]
        tick_l = ticks_loss[idx]
        if point <= 0.0 or (tick_p <= 0.0 and tick_l <= 0.0):
            continue

        direction = directions[idx]
        if direction == 0.0:
            continue

        entry_price = entry_prices[idx]
        if not np.isfinite(entry_price):
            continue

        lot = lot_sizes[idx]
        if lot <= 0.0:
            continue

        mark_price = bid_price if direction > 0.0 else bid_price + spread_points * point
        diff = direction * (mark_price - entry_price)
        # Seleccionar valor por punto según signo de P/L
        tick_val = tick_p if diff >= 0.0 else tick_l
        float_pl = (diff / point) * tick_val * lot

        total_float += float_pl
        total_lots += lot
        open_count += 1

    return total_float, total_lots, open_count


class Simulator:
    @staticmethod
    def run(dates, symbols, mats, em, pm, risk, signal_gens, local_idx_map, debug_mode: str, strategies_order=None):
        strategies_seq = (
            tuple(strategies_order)
            if strategies_order is not None
            else tuple(em._strategies.keys())
        )
        symbols = tuple(symbols)
        n = len(dates)
        n_sym = len(symbols)

        equity = np.empty(n, dtype=float)
        balance = np.empty(n, dtype=float)
        floating = np.empty(n, dtype=float)
        open_trades = np.empty(n, dtype=int)
        open_lots = np.empty(n, dtype=float)

        generators_per_symbol = {
            sym: tuple(signal_gens[strat][sym] for strat in strategies_seq)
            for sym in symbols
        }
        # Soporta dos modos de local_idx_map:
        #  - dict[date] -> idx (legacy)
        #  - np.ndarray[global_idx] -> local_idx or -1 (precomputado)
        local_idx_sources = {sym: local_idx_map[sym] for sym in symbols}
        apply_strategy = em.apply_strategy
        spread_points = float(em.spread_points or 0.0)

        last_bid_close = np.full(n_sym, np.nan, dtype=float)

        prog = BarProgress(n)
        for i, date in enumerate(dates):
            open_row = mats["open"][i]
            high_row = mats["high"][i]
            low_row = mats["low"][i]
            close_row = mats["close"][i]

            # (Optimized) defer TP/SL processing to a single two-phase call below

            for j, sym in enumerate(symbols):
                close_val = close_row[j]
                if np.isfinite(close_val):
                    last_bid_close[j] = close_val

                bid_open = open_row[j]
                if not np.isfinite(bid_open) or bid_open <= 0.0:
                    continue

                src = local_idx_sources[sym]
                if hasattr(src, "dtype"):
                    # array de enteros por índice global
                    local_idx = int(src[i])
                    if local_idx < 0:
                        continue
                else:
                    local_idx = src.get(date)
                    if local_idx is None:
                        continue

                bid_open_float = float(bid_open)
                generators = generators_per_symbol[sym]
                for strat, sig_gen in zip(strategies_seq, generators):
                    # Adaptador: instancia con método o tupla de arrays (buy, sell)
                    if hasattr(sig_gen, "generate_signals_for_candle"):
                        buy, sell = sig_gen.generate_signals_for_candle(local_idx)
                    else:
                        buy_arr, sell_arr = sig_gen  # tipo: Tuple[np.ndarray, np.ndarray]
                        buy = bool(buy_arr[local_idx])
                        sell = bool(sell_arr[local_idx])
                    apply_strategy(
                        strat,
                        sym,
                        bool(buy),
                        bool(sell),
                        bid_open_float,
                        i,
                        date,
                    )

            risk.check_tp_sl(
                opens=open_row,
                lows=low_row,
                highs=high_row,
                closes=close_row,
                date=date,
            )

            eq, bal, fpl, cnt, lots = Simulator._compute_equity_balance(
                pm, last_bid_close, spread_points
            )
            equity[i], balance[i], floating[i], open_trades[i], open_lots[i] = (
                eq,
                bal,
                fpl,
                cnt,
                lots,
            )
            prog.update(i + 1)
        prog.stop()

        with np.errstate(divide="ignore", invalid="ignore"):
            cum_max = np.maximum.accumulate(equity)
            dd_equity_pct = np.where(cum_max > 0, -((cum_max - equity) / cum_max * 100.0), 0.0)

        if debug_mode == "final":
            import pandas as pd

            print("\n--- [DEBUG final] Todas las operaciones ---")
            print(pd.DataFrame(pm.results).to_string(index=False))

        records = [
            {
                "date": dt,
                "equity": eqv,
                "balance": balv,
                "floating": fplv,
                "dd_equity_pct": ddv,
                "open_trades": cntv,
                "open_lots": lotv,
            }
            for dt, eqv, balv, fplv, ddv, cntv, lotv in zip(
                dates, equity, balance, floating, dd_equity_pct, open_trades, open_lots
            )
        ]
        em.equity_over_time = records
        return {"records": records, "results": pm.results}

    @staticmethod
    def _compute_equity_balance(pm, bid_mark_prices, spread_points):
        bal = pm.balance

        positions = tuple(pm.positions.values())
        count = len(positions)
        if count:
            sym_idx_arr = np.full(count, -1, dtype=np.int64)
            directions_arr = np.zeros(count, dtype=np.float64)
            entry_prices_arr = np.full(count, np.nan, dtype=np.float64)
            lot_sizes_arr = np.zeros(count, dtype=np.float64)
            points_arr = np.zeros(count, dtype=np.float64)
            ticks_profit_arr = np.zeros(count, dtype=np.float64)
            ticks_loss_arr = np.zeros(count, dtype=np.float64)

            for idx, pos in enumerate(positions):
                pos_sym_idx = pos.get("sym_idx")
                if pos_sym_idx is not None:
                    sym_idx_arr[idx] = int(pos_sym_idx)

                direction = pos.get("dir")
                directions_arr[idx] = float(direction) if direction is not None else 0.0

                entry_price = pos.get("entry_price")
                entry_prices_arr[idx] = float(entry_price) if entry_price is not None else np.nan

                lot_size = pos.get("lot_size")
                lot_sizes_arr[idx] = float(lot_size) if lot_size is not None else 0.0

                point = pos.get("point")
                points_arr[idx] = float(point) if point else 0.0

                tick_p = pos.get("tick_profit") or pos.get("tick")
                tick_l = pos.get("tick_loss") or pos.get("tick")
                ticks_profit_arr[idx] = float(tick_p) if tick_p else 0.0
                ticks_loss_arr[idx] = float(tick_l) if tick_l else 0.0

            total_float, total_lots, open_count = _compute_equity_components(
                bid_mark_prices,
                sym_idx_arr,
                directions_arr,
                entry_prices_arr,
                lot_sizes_arr,
                points_arr,
                ticks_profit_arr,
                ticks_loss_arr,
                spread_points,
            )
        else:
            total_float = 0.0
            total_lots = 0.0
            open_count = 0

        floating = float(Decimal(total_float).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        equity = float(Decimal(bal + floating).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        open_lots = float(Decimal(total_lots).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return equity, bal, floating, open_count, open_lots
