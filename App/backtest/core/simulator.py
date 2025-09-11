from __future__ import annotations

import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from backtest.utils.progress import BarProgress


class Simulator:
    @staticmethod
    def run(dates, symbols, mats, em, pm, risk, signal_gens, local_idx_map, debug_mode: str, strategies_order=None):
        n = len(dates)
        n_sym = len(symbols)

        equity = np.empty(n, dtype=float)
        balance = np.empty(n, dtype=float)
        floating = np.empty(n, dtype=float)
        open_trades = np.empty(n, dtype=int)
        open_lots = np.empty(n, dtype=float)

        last_bid_close = np.full(n_sym, np.nan, dtype=float)

        prog = BarProgress(n)
        for i, date in enumerate(dates):
            o = mats["open"][i]
            h = mats["high"][i]
            l = mats["low"][i]
            c = mats["close"][i]

            risk.check_tp_sl(opens=o, lows=l, highs=h, closes=c, date=date, only_opened_before=date)

            for j, sym in enumerate(symbols):
                bid_open = o[j]
                if not np.isfinite(bid_open) or bid_open <= 0:
                    continue
                local_i = local_idx_map[sym].get(date)
                if local_i is None:
                    continue
                for strat in (strategies_order or list(em._strategies.keys())):
                    # Obtain per-strategy signal generator for this symbol
                    sig_gen = signal_gens[strat][sym]
                    buy, sell = sig_gen.generate_signals_for_candle(local_i)
                    em.apply_strategy(strat, sym, bool(buy), bool(sell), float(bid_open), i, date)

            risk.check_tp_sl(opens=o, lows=l, highs=h, closes=c, date=date, only_opened_on=date)

            bid_for_equity = np.where(np.isfinite(c), c, last_bid_close)
            last_bid_close = np.where(np.isfinite(c), c, last_bid_close)

            eq, bal, fpl, cnt, lots = Simulator._compute_equity_balance(pm, bid_for_equity, em)
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
    def _compute_equity_balance(pm, bid_mark_prices, em):
        bal = pm.balance
        total_float = 0.0
        total_lots = 0.0
        open_count = 0

        for pos in pm.positions.values():
            sym_idx = pos["sym_idx"]
            if sym_idx is None:
                continue
            bid_price = bid_mark_prices[sym_idx]
            if not np.isfinite(bid_price):
                continue

            point = pos.get("point")
            tick = pos.get("tick")
            if not point or not tick:
                continue

            spread_move = em.spread_points * point
            mark_price = bid_price if pos["dir"] > 0 else (bid_price + spread_move)
            diff = pos["dir"] * (mark_price - pos["entry_price"])
            float_pl = (diff / point) * tick * pos["lot_size"]

            total_float += float_pl
            total_lots += pos["lot_size"]
            open_count += 1

        floating = float(Decimal(total_float).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        equity = float(Decimal(bal + floating).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        open_lots = float(Decimal(total_lots).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return equity, bal, floating, open_count, open_lots
