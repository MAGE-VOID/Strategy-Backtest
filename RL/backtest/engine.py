# backtest/engine.py

import numpy as np
import pandas as pd
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.managers.risk_manager import RiskManager
from backtest.utils.progress import BarProgress


class BacktestEngine:
    def __init__(self, config):
        """
        config: instancia de BacktestConfig, debe tener `strategies_params` como
                un dict con una sola clave = nombre de la estrategia.
        """
        self.config = config
        self.equity_over_time = []
        self.strategy = None  # se llenará en _setup_managers

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        # 1) preparar datos
        df = self._prepare_dataframe(input_data)
        dates, symbols = self._extract_index_symbols(df)

        # 2) llenar matrices de precios
        price_matrices = self._init_price_matrices(len(dates), len(symbols))
        self._fill_price_matrices(df, dates, symbols, price_matrices)

        # 3) mapping de puntos/tick_value y setup de managers
        symbol_points = self._map_symbol_points(df, symbols)
        em, pm, risk, sym2idx = self._setup_managers(symbol_points, symbols)

        # 4) crear signal generators y mapeo fecha→índice local
        signal_gens = {}
        local_idx_map = {}
        for sym in symbols:
            grp = df[df["Symbol"] == sym]
            signal_gens[sym] = self.config.strategy_signal_class(grp)
            local_idx_map[sym] = {dt: idx for idx, dt in enumerate(grp.index)}

        # 5) simular barra a barra
        results = self._simulate_backtest(
            dates,
            symbols,
            price_matrices,
            em,
            pm,
            risk,
            sym2idx,
            signal_gens,
            local_idx_map,
        )

        # 6) estadísticas finales
        stats = self._finalize(em, results)
        return stats

    def _prepare_dataframe(self, df):
        if df is None or df.empty:
            raise ValueError("No hay datos de mercado para el backtest.")
        return df.sort_index()

    def _extract_index_symbols(self, df):
        return df.index.unique(), df["Symbol"].unique()

    def _init_price_matrices(self, n_steps, n_sym):
        shape = (n_steps, n_sym)
        return {
            "open": np.zeros(shape),
            "high": np.zeros(shape),
            "low": np.zeros(shape),
            "close": np.zeros(shape),
        }

    def _fill_price_matrices(self, df, dates, symbols, m):
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym]
            length = len(grp)
            for field in ("Open", "High", "Low", "Close"):
                m[field.lower()][:length, j] = grp[field].values

    def _map_symbol_points(self, df, symbols):
        return {
            sym: {
                "point": df.loc[df["Symbol"] == sym, "Point"].iat[0],
                "tick_value": df.loc[df["Symbol"] == sym, "Tick_Value"].iat[0],
            }
            for sym in symbols
        }

    def _setup_managers(self, symbol_points, symbols):
        # 1) determinar nombre de la estrategia
        params_dict = getattr(self.config, "strategies_params", {}) or {}
        if len(params_dict) == 1:
            strat = next(iter(params_dict))
        else:
            raise ValueError(
                "Debe pasar exactamente una estrategia en strategies_params."
            )
        self.strategy = strat
        params = params_dict[strat]

        # 2) crear EntryManager con esos params
        em = EntryManager(
            self.config.initial_balance,
            strategies_params=params,
            symbol_points_mapping=symbol_points,
        )
        pm = em.position_manager

        # 3) parche para añadir meta-info a cada posición abierta
        sym2idx = {sym: i for i, sym in enumerate(symbols)}
        orig_open = pm.open_position

        def open_with_meta(
            symbol, position_type, price, lot_size, sl=None, tp=None, open_date=None
        ):
            orig_open(symbol, position_type, price, lot_size, sl, tp, open_date)
            ticket = pm.ticket_counter - 1
            pos = pm.positions[ticket]
            idx = sym2idx[symbol]
            meta = symbol_points[symbol]
            extra = {
                "sym_idx": idx,
                "point": meta["point"],
                "tick": meta["tick_value"],
                "dir": 1 if position_type == "long" else -1,
            }
            pos.update(extra)
            pm.results[-1].update(extra)

        pm.open_position = open_with_meta
        risk = RiskManager(em, pm)
        return em, pm, risk, sym2idx

    def _simulate_backtest(
        self, dates, symbols, m, em, pm, risk, sym2idx, signal_gens, local_idx_map
    ):
        n = len(dates)
        equity = np.empty(n)
        balance = np.empty(n)
        open_trades = np.empty(n, dtype=int)
        open_lots = np.empty(n)

        progress = BarProgress(n)
        for i, date in enumerate(dates):
            o = m["open"][i]
            h = m["high"][i]
            l = m["low"][i]
            c = m["close"][i]

            # cierres por TP/SL
            risk.check_tp_sl(l, h, symbols, date)

            # aperturas y grids
            for j, sym in enumerate(symbols):
                price_o = o[j]
                if price_o <= 0:
                    continue
                local_i = local_idx_map[sym].get(date)
                if local_i is None:
                    continue
                buy_sig, sell_sig = signal_gens[sym].generate_signals_for_candle(
                    local_i
                )
                em.apply_strategy(
                    self.strategy,
                    sym,
                    bool(buy_sig),
                    bool(sell_sig),
                    price_o,
                    i,
                    date,
                )

            # snapshot intradía
            eq, bal, cnt, lots = self._compute_equity_balance(pm, c)
            equity[i], balance[i], open_trades[i], open_lots[i] = eq, bal, cnt, lots
            progress.update(i + 1)
        progress.stop()

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

    def _finalize(self, em, sim_data):
        stats = Statistics(
            em.position_manager.results,
            em.equity_over_time,
            self.config.initial_balance,
        ).calculate_statistics()
        return {
            "trades": sim_data["results"],
            "equity_over_time": em.equity_over_time,
            "statistics": stats,
        }

    def _compute_equity_balance(self, pm, closes):
        bal = pm.balance
        eq = bal
        total_lots = 0
        for pos in pm.positions.values():
            cp = closes[pos["sym_idx"]]
            if cp <= 0:
                continue
            diff = pos["dir"] * (cp - pos["entry_price"])
            eq += (diff / pos["point"]) * pos["tick"] * pos["lot_size"]
            total_lots += pos["lot_size"]
        return eq, bal, len(pm.positions), total_lots
