# backtest/engine.py
import numpy as np
import pandas as pd
from datetime import datetime
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.config import BacktestConfig
from backtest.utils.progress import BarProgress
from backtest.optimization_engine import OptimizationEngine

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy_manager = None
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        if self.config.mode == "optimization":
            return OptimizationEngine(self.config, input_data).run_optimization()
        return self._run_single_backtest(input_data)

    def _run_single_backtest(self, input_data: pd.DataFrame) -> dict:
        df = input_data.sort_index()
        dates   = df.index.unique()
        symbols = df["Symbol"].unique()
        n_steps = len(dates)
        n_sym   = symbols.size

        # --- 1) Matrices OHLC ---
        open_mat  = np.zeros((n_steps, n_sym), dtype=float)
        high_mat  = np.zeros((n_steps, n_sym), dtype=float)
        low_mat   = np.zeros((n_steps, n_sym), dtype=float)
        close_mat = np.zeros((n_steps, n_sym), dtype=float)

        buy_signals  = np.zeros((n_steps, n_sym), dtype=bool)
        sell_signals = np.zeros((n_steps, n_sym), dtype=bool)

        # Para mapear señales a índice global
        dates_idx = {dt:i for i, dt in enumerate(dates)}

        # --- 2) Rellenar matrices y señales ---
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym]
            # Alineamos con reindex para no saltar fechas
            ser_o = grp["Open"].reindex(dates).fillna(0)
            ser_h = grp["High"].reindex(dates).fillna(0)
            ser_l = grp["Low"].reindex(dates).fillna(0)
            ser_c = grp["Close"].reindex(dates).fillna(0)
            open_mat[:,j]  = ser_o.values
            high_mat[:,j]  = ser_h.values
            low_mat[:,j]   = ser_l.values
            close_mat[:,j] = ser_c.values

            # señales: generador usa sólo los índices locales de grp
            gen = self.config.strategy_signal_class(grp)
            for local_i, dt in enumerate(grp.index):
                gi = dates_idx.get(dt, -1)
                if gi < 0: continue
                b, s = gen.generate_signals_for_candle(local_i)
                buy_signals[gi, j]  = b
                sell_signals[gi, j] = s

        # --- 3) Punto/Tick mapping ---
        symbol_points = {
            sym: {
                "point": grp["Point"].iat[0],
                "tick_value": grp["Tick_Value"].iat[0]
            }
            for sym, grp in df.groupby("Symbol")
        }

        # --- 4) Iniciar EntryManager y parche open_position para meta-info ---
        em = EntryManager(self.config.initial_balance,
                          symbol_points_mapping=symbol_points)
        pm = em.position_manager
        self.strategy_manager = em

        sym2idx = {sym:i for i, sym in enumerate(symbols)}
        orig_open = pm.open_position

        def open_with_meta(symbol, position_type, price, lot_size, sl=None, tp=None, open_date=None):
            orig_open(symbol, position_type, price, lot_size, sl, tp, open_date)
            ticket = pm.ticket_counter - 1
            pos    = pm.positions[ticket]
            j      = sym2idx[symbol]
            meta   = symbol_points[symbol]
            pos.update({
                "sym_idx": j,
                "point":   meta["point"],
                "tick":    meta["tick_value"],
                "dir":     1 if position_type=="long" else -1
            })
            pm.results[-1].update({
                "sym_idx": j,
                "point":   meta["point"],
                "tick":    meta["tick_value"],
                "dir":     pos["dir"]
            })

        pm.open_position = open_with_meta

        manage_tp_sl = em.manage_tp_sl
        apply_strat  = em.apply_strategy

        equity_arr  = np.empty(n_steps, dtype=float)
        balance_arr = np.empty(n_steps, dtype=float)
        open_trds   = np.empty(n_steps, dtype=int)
        open_lots   = np.empty(n_steps, dtype=float)

        progress = BarProgress(n_steps)

        # --- 5) Bucle principal: cada vela una vez ---
        for i, date in enumerate(dates):
            opens  = open_mat[i]
            highs  = high_mat[i]
            lows   = low_mat[i]
            closes = close_mat[i]
            buys   = buy_signals[i]
            sells  = sell_signals[i]

            # a) señales + gestión TP/SL en la apertura
            for j, sym in enumerate(symbols):
                price_o = opens[j]
                if price_o == 0: 
                    continue
                manage_tp_sl(sym, price_o, date)
                apply_strat(self.config.strategy_name,
                            sym,
                            bool(buys[j]),
                            bool(sells[j]),
                            price_o, i, date)

            # b) **intrabar**: chequeo de extremos para cerrar stops y take-profits
            #   sólo primera vez que toca cualquiera de los dos
            for ticket, pos in list(pm.positions.items()):
                sym = pos["symbol"]
                j   = pos["sym_idx"]
                tp  = pos.get("tp", None)
                sl  = pos.get("sl", None)
                if pos["position"]=="long":
                    # si toca primero el SL o el TP
                    if sl is not None and lows[j] <= sl:
                        pm.close_position(ticket, sl, date)
                        continue
                    if tp is not None and highs[j] >= tp:
                        pm.close_position(ticket, tp, date)
                else:  # short
                    if sl is not None and highs[j] >= sl:
                        pm.close_position(ticket, sl, date)
                        continue
                    if tp is not None and lows[j] <= tp:
                        pm.close_position(ticket, tp, date)

            # c) calcular equity real-time
            bal = pm.balance
            eq  = bal
            lots = 0
            for pos in pm.positions.values():
                j   = pos["sym_idx"]
                cp  = closes[j]
                if cp == 0:
                    continue
                diff = pos["dir"] * (cp - pos["entry_price"])
                eq  += (diff / pos["point"]) * pos["tick"] * pos["lot_size"]
                lots += pos["lot_size"]

            equity_arr[i]  = eq
            balance_arr[i] = bal
            open_trds[i]   = len(pm.positions)
            open_lots[i]   = lots

            progress.update(i+1)
        progress.stop()

        # --- 6) Resultados y estadísticas ---
        df_out = pd.DataFrame({
            "date":        dates,
            "equity":      equity_arr,
            "balance":     balance_arr,
            "open_trades": open_trds,
            "open_lots":   open_lots
        })
        self.equity_over_time = df_out.to_dict("records")

        stats = Statistics(pm.results,
                           self.equity_over_time,
                           self.config.initial_balance).calculate_statistics()

        return {
            "trades":           pm.results,
            "equity_over_time": self.equity_over_time,
            "statistics":       stats,
        }
