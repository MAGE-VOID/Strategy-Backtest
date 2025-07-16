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
        self.strategy_manager = None
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        return self._run_single_backtest(input_data)

    def _run_single_backtest(self, input_data: pd.DataFrame) -> dict:
        df = input_data.sort_index()
        dates = df.index.unique()
        symbols = df["Symbol"].unique()
        n_steps = len(dates)
        n_sym = symbols.size

        # --- 1) Pre-cálculo de matrices OHLC y señales (optimización de reindexado) ---
        open_mat = np.zeros((n_steps, n_sym), dtype=float)
        high_mat = np.zeros((n_steps, n_sym), dtype=float)
        low_mat = np.zeros((n_steps, n_sym), dtype=float)
        close_mat = np.zeros((n_steps, n_sym), dtype=float)
        buy_signals = np.zeros((n_steps, n_sym), dtype=bool)
        sell_signals = np.zeros((n_steps, n_sym), dtype=bool)

        # Optimización: Utilizar un diccionario con índices para acceder más rápidamente
        dates_idx = {dt: i for i, dt in enumerate(dates)}

        # Optimizar el cálculo de las señales y los valores OHLC
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym]
            ser_o = grp["Open"].values
            ser_h = grp["High"].values
            ser_l = grp["Low"].values
            ser_c = grp["Close"].values

            # Usamos numpy.pad para evitar reindex y optimizar el rendimiento
            open_mat[:, j] = np.pad(ser_o, (0, n_steps - len(ser_o)), mode="constant")
            high_mat[:, j] = np.pad(ser_h, (0, n_steps - len(ser_h)), mode="constant")
            low_mat[:, j] = np.pad(ser_l, (0, n_steps - len(ser_l)), mode="constant")
            close_mat[:, j] = np.pad(ser_c, (0, n_steps - len(ser_c)), mode="constant")

            # Usamos numpy para calcular las señales de forma más eficiente
            gen = self.config.strategy_signal_class(grp)
            for local_i, dt in enumerate(grp.index):
                gi = dates_idx.get(dt, -1)
                if gi >= 0:
                    b, s = gen.generate_signals_for_candle(local_i)
                    buy_signals[gi, j] = b
                    sell_signals[gi, j] = s

        # --- 2) Mapeo de puntos/ticks ---
        # Optimización: Usamos `numpy` para evitar un ciclo de acceso costoso
        symbol_points = {
            sym: {
                "point": df.loc[df["Symbol"] == sym, "Point"].iat[0],
                "tick_value": df.loc[df["Symbol"] == sym, "Tick_Value"].iat[0],
            }
            for sym in symbols
        }

        # --- 3) Iniciar EntryManager y parche open_position para meta-info ---
        em = EntryManager(
            self.config.initial_balance, symbol_points_mapping=symbol_points
        )
        pm = em.position_manager
        self.strategy_manager = em

        sym2idx = {sym: i for i, sym in enumerate(symbols)}
        orig_open = pm.open_position

        # Optimización de open_position
        def open_with_meta(
            symbol, position_type, price, lot_size, sl=None, tp=None, open_date=None
        ):
            orig_open(symbol, position_type, price, lot_size, sl, tp, open_date)
            ticket = pm.ticket_counter - 1
            pos = pm.positions[ticket]
            j = sym2idx[symbol]
            meta = symbol_points[symbol]
            pos.update(
                {
                    "sym_idx": j,
                    "point": meta["point"],
                    "tick": meta["tick_value"],
                    "dir": 1 if position_type == "long" else -1,
                }
            )
            pm.results[-1].update(
                {
                    "sym_idx": j,
                    "point": meta["point"],
                    "tick": meta["tick_value"],
                    "dir": pos["dir"],
                }
            )

        pm.open_position = open_with_meta
        manage_tp_sl = em.manage_tp_sl
        apply_strat = em.apply_strategy

        equity_arr = np.empty(n_steps, dtype=float)
        balance_arr = np.empty(n_steps, dtype=float)
        open_trds = np.empty(n_steps, dtype=int)
        open_lots = np.empty(n_steps, dtype=float)

        progress = BarProgress(n_steps)

        # --- 4) Bucle principal optimizado (procesamiento por lotes) ---
        for i, date in enumerate(dates):
            opens = open_mat[i]
            highs = high_mat[i]
            lows = low_mat[i]
            closes = close_mat[i]
            buys = buy_signals[i]
            sells = sell_signals[i]

            # a) Señales + gestión TP/SL
            for j, sym in enumerate(symbols):
                price_o = opens[j]
                if price_o == 0:
                    continue
                manage_tp_sl(sym, price_o, date)
                apply_strat(
                    self.config.strategy_name,
                    sym,
                    bool(buys[j]),
                    bool(sells[j]),
                    price_o,
                    i,
                    date,
                )

            # b) Cierre de posiciones por TP/SL
            positions_copy = list(pm.positions.items())
            for ticket, pos in positions_copy:
                sym = pos["symbol"]
                j = pos["sym_idx"]
                tp = pos.get("tp", None)
                sl = pos.get("sl", None)
                if pos["position"] == "long":
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

            # c) Calcular equity y balance en tiempo real
            bal = pm.balance
            eq = bal
            lots = 0
            for pos in pm.positions.values():
                j = pos["sym_idx"]
                cp = closes[j]
                if cp == 0:
                    continue
                diff = pos["dir"] * (cp - pos["entry_price"])
                eq += (diff / pos["point"]) * pos["tick"] * pos["lot_size"]
                lots += pos["lot_size"]

            equity_arr[i] = eq
            balance_arr[i] = bal
            open_trds[i] = len(pm.positions)
            open_lots[i] = lots

            progress.update(i + 1)

        progress.stop()

        # --- 5) Resultados y estadísticas ---
        df_out = pd.DataFrame(
            {
                "date": dates,
                "equity": equity_arr,
                "balance": balance_arr,
                "open_trades": open_trds,
                "open_lots": open_lots,
            }
        )
        self.equity_over_time = df_out.to_dict("records")

        stats = Statistics(
            pm.results, self.equity_over_time, self.config.initial_balance
        ).calculate_statistics()

        return {
            "trades": pm.results,
            "equity_over_time": self.equity_over_time,
            "statistics": stats,
        }
