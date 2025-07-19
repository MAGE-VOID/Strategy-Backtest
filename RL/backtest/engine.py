# backtest/engine.py

import numpy as np
import pandas as pd
from datetime import datetime
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.config import BacktestConfig
from backtest.utils.progress import BarProgress


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        """
        Punto de entrada: corre un único backtest.
        """
        df = self._prepare_dataframe(input_data)
        dates, symbols = self._extract_index_symbols(df)
        matrices = self._init_price_and_signal_matrices(len(dates), len(symbols))
        self._fill_price_and_signal_matrices(df, dates, symbols, matrices)
        symbol_points = self._map_symbol_points(df, symbols)
        manager, sym2idx = self._setup_entry_and_position_manager(
            symbol_points, symbols
        )
        results = self._simulate_backtest(dates, symbols, matrices, manager, sym2idx)
        stats = self._finalize(manager, results, symbol_points)
        return stats

    # --- Paso 1: preparar datos ---
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordena y valida el DataFrame de entrada."""
        if df is None or df.empty:
            raise ValueError("No hay datos de mercado para el backtest.")
        return df.sort_index()

    def _extract_index_symbols(self, df: pd.DataFrame):
        """Devuelve array de fechas únicas y array de símbolos."""
        dates = df.index.unique()
        symbols = df["Symbol"].unique()
        return dates, symbols

    # --- Paso 2: inicializar matrices de precios y señales ---
    def _init_price_and_signal_matrices(self, n_steps: int, n_sym: int):
        """Crea arrays vacíos para OHLC y señales."""
        shape = (n_steps, n_sym)
        return {
            "open": np.zeros(shape),
            "high": np.zeros(shape),
            "low": np.zeros(shape),
            "close": np.zeros(shape),
            "buy": np.zeros(shape, dtype=bool),
            "sell": np.zeros(shape, dtype=bool),
        }

    # --- Paso 3: rellenar matrices con datos y señales ---
    def _fill_price_and_signal_matrices(self, df, dates, symbols, m):
        """
        Para cada símbolo:
          1) rellena la columna j de open/high/low/close
          2) genera señales buy/sell con la estrategia
        """
        dates_idx = {dt: i for i, dt in enumerate(dates)}
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym]
            length = len(grp)
            # rellenar precios
            for field in ("Open", "High", "Low", "Close"):
                arr = grp[field].values
                m[field.lower()][:length, j] = arr
            # generar señales
            signal_gen = self.config.strategy_signal_class(grp)
            for local_i, dt in enumerate(grp.index):
                global_i = dates_idx.get(dt)
                if global_i is None:
                    continue
                b, s = signal_gen.generate_signals_for_candle(local_i)
                m["buy"][global_i, j] = b
                m["sell"][global_i, j] = s

    # --- Paso 4: obtener puntos/tick_value por símbolo ---
    def _map_symbol_points(self, df, symbols):
        """Devuelve dict {symbol: {'point':…, 'tick_value':…}}."""
        return {
            sym: {
                "point": df.loc[df["Symbol"] == sym, "Point"].iat[0],
                "tick_value": df.loc[df["Symbol"] == sym, "Tick_Value"].iat[0],
            }
            for sym in symbols
        }

    # --- Paso 5: configurar EntryManager y parchear open_position ---
    def _setup_entry_and_position_manager(self, symbol_points, symbols):
        em = EntryManager(
            self.config.initial_balance, symbol_points_mapping=symbol_points
        )
        pm = em.position_manager

        # parche para incluir meta-info en cada posición/resultado
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
        return em, sym2idx

    # --- Paso 6: bucle principal de simulación ---
    def _simulate_backtest(self, dates, symbols, m, em: EntryManager, sym2idx):
        """
        Recorre cada timestamp:
          a) gestiona TP/SL y aplica señales
          b) cierra posiciones por TP/SL
          c) calcula equity/balance/lotes
        """
        n = len(dates)
        equity = np.empty(n)
        balance = np.empty(n)
        open_trades = np.empty(n, dtype=int)
        open_lots = np.empty(n)

        progress = BarProgress(n)
        pm = em.position_manager
        manage = em.manage_tp_sl
        apply = em.apply_strategy

        for i, date in enumerate(dates):
            o = m["open"][i]
            h = m["high"][i]
            l = m["low"][i]
            c = m["close"][i]
            buys = m["buy"][i]
            sells = m["sell"][i]

            # a) señales + gestión TP/SL
            for j, sym in enumerate(symbols):
                price_o = o[j]
                if price_o <= 0:
                    continue
                manage(sym, price_o, date)
                apply(
                    self.config.strategy_name,
                    sym,
                    bool(buys[j]),
                    bool(sells[j]),
                    price_o,
                    i,
                    date,
                )

            # b) cierre TP/SL
            self._close_by_tp_sl(pm, l, h, date)

            # c) cálculo de métricas intradía
            eq, bal, cnt, lots = self._compute_equity_balance(pm, c)
            equity[i], balance[i], open_trades[i], open_lots[i] = eq, bal, cnt, lots

            progress.update(i + 1)
        progress.stop()

        # volcar resultados en em.equity_over_time
        records = []
        for dt, eqv, balv, cntv, lotv in zip(
            dates, equity, balance, open_trades, open_lots
        ):
            records.append(
                {
                    "date": dt,
                    "equity": eqv,
                    "balance": balv,
                    "open_trades": cntv,
                    "open_lots": lotv,
                }
            )
        em.equity_over_time = records  # inyectamos directamente

        return {"records": records, "results": pm.results}

    def _close_by_tp_sl(self, pm, lows, highs, date):
        """Cierra todas las posiciones cuyo TP o SL se hayan alcanzado."""
        for ticket, pos in list(pm.positions.items()):
            idx = pos["sym_idx"]
            tp, sl = pos.get("tp"), pos.get("sl")
            if pos["position"] == "long":
                if sl is not None and lows[idx] <= sl:
                    pm.close_position(ticket, sl, date)
                elif tp is not None and highs[idx] >= tp:
                    pm.close_position(ticket, tp, date)
            else:
                if sl is not None and highs[idx] >= sl:
                    pm.close_position(ticket, sl, date)
                elif tp is not None and lows[idx] <= tp:
                    pm.close_position(ticket, tp, date)

    def _compute_equity_balance(self, pm, closes):
        """
        Retorna (equity, balance, n_positions, total_lots)
        con base en el cierre de candle actual.
        """
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

    # --- Paso 7: cálcula estadísticas y arma el diccionario final ---
    def _finalize(self, em: EntryManager, sim_data, symbol_points):
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
