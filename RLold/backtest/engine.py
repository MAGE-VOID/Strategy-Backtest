# backtest/engine.py

import numpy as np
import pandas as pd
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.managers.risk_manager import RiskManager
from backtest.utils.progress import BarProgress


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.debug_mode = config.debug_mode
        self.strategies = list(config.strategies_params.keys())
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        df = self._prepare_dataframe(input_data)
        dates, symbols = self._extract_index_symbols(df)

        price_mats = self._init_price_matrices(len(dates), len(symbols))
        self._fill_price_matrices(df, dates, symbols, price_mats)

        symbol_points = self._map_symbol_points(df, symbols)
        em, pm, risk = self._setup_managers(symbol_points, symbols)

        signal_gens = {}
        local_idx_map = {}
        for sym in symbols:
            grp = df[df["Symbol"] == sym]
            signal_gens[sym] = self.config.strategy_signal_class(grp)
            # fecha global -> índice local (solo si existe vela en ese símbolo)
            local_idx_map[sym] = {dt: idx for idx, dt in enumerate(grp.index)}

        sim = self._simulate_backtest(
            dates, symbols, price_mats, em, pm, risk, signal_gens, local_idx_map
        )
        stats = self._finalize(em, sim)
        return stats

    def _prepare_dataframe(self, df):
        if df is None or df.empty:
            raise ValueError("No hay datos para el backtest.")
        return df.sort_index()

    def _extract_index_symbols(self, df):
        return df.index.unique(), df["Symbol"].unique()

    def _init_price_matrices(self, n_steps, n_sym):
        shape = (n_steps, n_sym)
        return {
            k: np.full(shape, np.nan, dtype=float)
            for k in ("open", "high", "low", "close")
        }

    def _fill_price_matrices(self, df, dates, symbols, mats):
        """
        Alinea cada serie OHLC por símbolo al índice global de fechas (union).
        """
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym].sort_index()
            for fld in ("Open", "High", "Low", "Close"):
                s = grp[fld].reindex(dates)
                mats[fld.lower()][:, j] = s.to_numpy(dtype=float)

    def _map_symbol_points(self, df, symbols):
        return {
            sym: {
                "point": df.loc[df["Symbol"] == sym, "Point"].iat[0],
                "point_value": df.loc[df["Symbol"] == sym, "Point_Value"].iat[0],
            }
            for sym in symbols
        }

    def _setup_managers(self, symbol_points, symbols):
        em = EntryManager(
            self.config.initial_balance,
            strategies_params=self.config.strategies_params,
            symbol_points_mapping=symbol_points,
        )
        pm = em.position_manager
        pm.sym2idx = {sym: idx for idx, sym in enumerate(symbols)}
        risk = RiskManager(em, pm)
        return em, pm, risk

    def _simulate_backtest(
        self, dates, symbols, mats, em, pm, risk, signal_gens, local_idx_map
    ):
        n = len(dates)
        n_sym = len(symbols)

        equity = np.empty(n, dtype=float)
        balance = np.empty(n, dtype=float)
        open_trades = np.empty(n, dtype=int)
        open_lots = np.empty(n, dtype=float)

        # Último BID conocido (para MTM si en una fecha el símbolo no tiene vela)
        last_bid_close = np.full(n_sym, np.nan, dtype=float)

        prog = BarProgress(n)
        for i, date in enumerate(dates):
            o = mats["open"][i]  # Bid Open
            h = mats["high"][i]  # Bid High
            l = mats["low"][i]  # Bid Low
            c = mats["close"][i]  # Bid Close

            # 1) Cierre de posiciones abiertas antes de esta vela (no in-bar)
            risk.check_tp_sl(
                opens=o, lows=l, highs=h, closes=c, date=date, only_opened_before=date
            )

            # 2) Apertura en el OPEN de la vela (BUY a Ask, SELL a Bid)
            for j, sym in enumerate(symbols):
                bid_open = o[j]
                if not np.isfinite(bid_open) or bid_open <= 0:
                    continue
                local_i = local_idx_map[sym].get(date)
                if local_i is None:
                    continue

                buy, sell = signal_gens[sym].generate_signals_for_candle(local_i)
                for strat in self.strategies:
                    # Solo intentamos abrir en el OPEN de la vela
                    em.apply_strategy(
                        strat, sym, bool(buy), bool(sell), float(bid_open), i, date
                    )

            # 3) Permitir cierre in‑bar para posiciones abiertas en esta misma vela
            risk.check_tp_sl(
                opens=o, lows=l, highs=h, closes=c, date=date, only_opened_on=date
            )

            # 4) Mark‑to‑market (BUY→BidClose, SELL→AskClose)
            bid_for_equity = np.where(np.isfinite(c), c, last_bid_close)
            last_bid_close = np.where(np.isfinite(c), c, last_bid_close)

            eq, bal, cnt, lots = self._compute_equity_balance(pm, bid_for_equity, em)
            equity[i], balance[i], open_trades[i], open_lots[i] = eq, bal, cnt, lots
            prog.update(i + 1)
        prog.stop()

        if self.debug_mode == "final":
            import pandas as pd

            print("\n--- [DEBUG final] Todas las operaciones ---")
            print(pd.DataFrame(pm.results).to_string(index=False))

        records = [
            {
                "date": dt,
                "equity": eqv,
                "balance": balv,
                "open_trades": cntv,
                "open_lots": lotv,
            }
            for dt, eqv, balv, cntv, lotv in zip(
                dates, equity, balance, open_trades, open_lots
            )
        ]
        em.equity_over_time = records
        return {"records": records, "results": pm.results}

    def _compute_equity_balance(self, pm, bid_mark_prices, em):
        """
        Equity = Balance + ∑ P/L flotante (por posición, independiente):
         - LONG: marcar a BID
         - SHORT: marcar a ASK = BID + spread
        """
        bal = pm.balance
        eq = bal
        total_lots = 0.0

        for pos in pm.positions.values():
            sym_idx = pos["sym_idx"]
            if sym_idx is None:
                continue

            bid_price = bid_mark_prices[sym_idx]
            if not np.isfinite(bid_price):
                continue

            spread_move = em.spread_points * pos["point"]
            mark_price = bid_price if pos["dir"] > 0 else (bid_price + spread_move)

            diff = pos["dir"] * (mark_price - pos["entry_price"])
            float_pl = (diff / pos["point"]) * pos["tick"] * pos["lot_size"]

            eq += float_pl
            total_lots += pos["lot_size"]

        return eq, bal, len(pm.positions), total_lots

    def _finalize(self, em, sim):
        stats = Statistics(
            em.position_manager.results,
            em.equity_over_time,
            self.config.initial_balance,
        ).calculate_statistics()
        return {
            "trades": sim["results"],
            "equity_over_time": em.equity_over_time,
            "statistics": stats,
        }
