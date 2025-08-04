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
        return {k: np.zeros(shape) for k in ("open", "high", "low", "close")}

    def _fill_price_matrices(self, df, dates, symbols, mats):
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym]
            for fld in ("Open", "High", "Low", "Close"):
                mats[fld.lower()][: len(grp), j] = grp[fld].values

    def _map_symbol_points(self, df, symbols):
        return {
            sym: {
                "point": df.loc[df["Symbol"] == sym, "Point"].iat[0],
                #"tick_value": df.loc[df["Symbol"] == sym, "Tick_Value"].iat[0],
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
        equity = np.empty(n)
        balance = np.empty(n)
        open_trades = np.empty(n, dtype=int)
        open_lots = np.empty(n)

        prog = BarProgress(n)
        for i, date in enumerate(dates):
            o, h, l, c = (
                mats["open"][i],
                mats["high"][i],
                mats["low"][i],
                mats["close"][i],
            )

            # 1) cierres por TP/SL
            risk.check_tp_sl(l, h, symbols, date)

            # 2) aperturas por estrategia
            for j, sym in enumerate(symbols):
                price_o = o[j]
                if price_o <= 0:
                    continue
                local_i = local_idx_map[sym].get(date)
                if local_i is None:
                    continue

                buy, sell = signal_gens[sym].generate_signals_for_candle(local_i)
                for strat in self.strategies:
                    em.apply_strategy(
                        strat, sym, bool(buy), bool(sell), price_o, i, date
                    )

            # 3) snapshot intradía
            eq, bal, cnt, lots = self._compute_equity_balance(pm, c)
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

    def _compute_equity_balance(self, pm, closes):
        bal = pm.balance
        eq = bal
        total_lots = 0
        for pos in pm.positions.values():
            cp = closes[pos["sym_idx"]]
            diff = pos["dir"] * (cp - pos["entry_price"])
            eq += (diff / pos["point"]) * pos["tick"] * pos["lot_size"]
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
