# backtest/optimization_engine.py

import numpy as np
import pandas as pd
import copy
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.utils.barprogress_multi import MultiProgress
from backtest.algorithm.parameter_combinations import generate_combinations
from backtest.algorithm.metric_config import select_best_result


class OptimizationEngine:
    """
    Motor de optimización que maneja el proceso de backtesting con múltiples combinaciones de parámetros.
    """

    def __init__(self, config, input_data: pd.DataFrame):
        self.config = config
        self.input_data = input_data
        self.equity_over_time = []

    def run_optimization(self):
        # Instanciar la señal para obtener los parámetros de optimización
        strategy_signal = self.config.strategy_signal_class(self.input_data)
        optimization_params = strategy_signal.optimization_params

        param_combinations = generate_combinations(optimization_params)
        total_combinations = len(param_combinations)
        print(f"\nTotal de combinaciones de parámetros: {total_combinations}\n")

        # Preprocesado: ordenar índice
        preprocessed_data = self.input_data.sort_index()

        results = []
        max_workers = min(4, multiprocessing.cpu_count())
        with multiprocessing.Manager() as manager:
            progress_updates = manager.dict()  # dict compartido para progresos
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, param_values in enumerate(param_combinations):
                    params_dict = dict(zip(optimization_params.keys(), param_values))
                    nucleo_id = idx % max_workers
                    futures.append(
                        executor.submit(
                            _run_single_backtest_parallel,
                            (
                                self.config,
                                preprocessed_data,
                                params_dict,
                                False,  # full_output=False en optimización
                                nucleo_id,
                                progress_updates,
                            ),
                        )
                    )

                # Barra de progreso multi-núcleo
                multi_prog = MultiProgress(total_tasks=total_combinations)
                multi_prog.start()
                while True:
                    done = sum(f.done() for f in futures)
                    multi_prog.update_overall(done)
                    for nid, upd in progress_updates.items():
                        curr = upd.get("progress", 0)
                        total = upd.get("total", 1)
                        if nid not in multi_prog.worker_tasks:
                            multi_prog.add_worker(nid, f"Núcleo {nid}", total=total)
                        multi_prog.update_worker(nid, curr)
                    if done == total_combinations:
                        break
                    time.sleep(0.5)
                multi_prog.stop()

                for f in as_completed(futures):
                    results.append(f.result())

        best = select_best_result(results)
        best_params = best.get("optimized_params", {})

        print("\nBacktest final - Mejor combinación de parámetros:")
        print(best_params)

        # Backtest final con salida completa
        final_signal = self.config.strategy_signal_class(preprocessed_data)
        final_signal.set_optimized_params(best_params)
        final_result = self._run_single_backtest(
            preprocessed_data,
            final_signal,
            full_output=True,
        )
        final_result["optimized_params"] = best_params
        return final_result

    def _run_single_backtest(
        self,
        input_data: pd.DataFrame,
        strategy_signal,
        full_output=False,
        nucleo_id=None,
        progress_updates=None,
    ) -> dict:
        # --- 1) Preparar matrices OHLC y señales ---
        df = input_data
        dates = df.index.unique()
        symbols = df["Symbol"].unique()
        n_steps = len(dates)
        n_sym = len(symbols)

        open_mat = np.zeros((n_steps, n_sym), dtype=float)
        high_mat = np.zeros((n_steps, n_sym), dtype=float)
        low_mat = np.zeros((n_steps, n_sym), dtype=float)
        close_mat = np.zeros((n_steps, n_sym), dtype=float)
        buy_mat = np.zeros((n_steps, n_sym), dtype=bool)
        sell_mat = np.zeros((n_steps, n_sym), dtype=bool)

        # Mapa rápido de fecha → índice
        date2idx = {dt: i for i, dt in enumerate(dates)}

        # Rellenar matrices y señales
        for j, sym in enumerate(symbols):
            grp = df[df["Symbol"] == sym]
            o = grp["Open"].values
            h = grp["High"].values
            l = grp["Low"].values
            c = grp["Close"].values

            open_mat[:, j] = np.pad(
                o, (0, n_steps - len(o)), mode="constant", constant_values=0
            )
            high_mat[:, j] = np.pad(
                h, (0, n_steps - len(h)), mode="constant", constant_values=0
            )
            low_mat[:, j] = np.pad(
                l, (0, n_steps - len(l)), mode="constant", constant_values=0
            )
            close_mat[:, j] = np.pad(
                c, (0, n_steps - len(c)), mode="constant", constant_values=0
            )

            # Generador de señales por vela
            for local_i, dt in enumerate(grp.index):
                gi = date2idx.get(dt, -1)
                if gi >= 0:
                    b, s = strategy_signal.generate_signals_for_candle(local_i)
                    buy_mat[gi, j] = b
                    sell_mat[gi, j] = s

        # --- 2) Mapeo de punto y tick_value ---
        symbol_points = {
            sym: {
                "point": df.loc[df["Symbol"] == sym, "Point"].iat[0],
                "tick_value": df.loc[df["Symbol"] == sym, "Tick_Value"].iat[0],
            }
            for sym in symbols
        }

        # --- 3) Inicializar EntryManager y parche open_position ---
        em = EntryManager(
            self.config.initial_balance, symbol_points_mapping=symbol_points
        )
        pm = em.position_manager
        self.strategy_manager = em

        sym2idx = {sym: i for i, sym in enumerate(symbols)}
        orig_open = pm.open_position

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
                    "tick_value": meta["tick_value"],
                    "dir": 1 if position_type == "long" else -1,
                }
            )
            pm.results[-1].update(
                {
                    "sym_idx": j,
                    "point": meta["point"],
                    "tick_value": meta["tick_value"],
                    "dir": pos["dir"],
                }
            )

        pm.open_position = open_with_meta
        manage_tp_sl = em.manage_tp_sl
        apply_strat = em.apply_strategy

        # --- 4) Arrays para resultados y barra de progreso ---
        equity_arr = np.empty(n_steps, dtype=float)
        balance_arr = np.empty(n_steps, dtype=float)
        open_trades_arr = np.empty(n_steps, dtype=int)
        open_lots_arr = np.empty(n_steps, dtype=float)

        if nucleo_id is None:
            # Backtest final: barra simple
            from backtest.utils.progress import BarProgress

            prog = BarProgress(n_steps, description="Final Backtest Progress")
            sample_interval = 1
        else:
            # Optimización: muestreo y actualizaciones compartidas
            progress_updates[nucleo_id] = {"progress": 0, "total": n_steps}
            sample_interval = max(1, n_steps // 1000)

        # --- 5) Bucle principal ---
        for i, date in enumerate(dates):
            opens = open_mat[i]
            highs = high_mat[i]
            lows = low_mat[i]
            closes = close_mat[i]
            buys = buy_mat[i]
            sells = sell_mat[i]

            # a) Señales y gestión TP/SL
            for j, sym in enumerate(symbols):
                if opens[j] == 0:
                    continue
                manage_tp_sl(sym, opens[j], date)
                apply_strat(
                    self.config.strategy_name,
                    sym,
                    bool(buys[j]),
                    bool(sells[j]),
                    opens[j],
                    i,
                    date,
                )

            # b) Cierre por TP/SL
            for ticket, pos in list(pm.positions.items()):
                sym = pos["symbol"]
                j = pos["sym_idx"]
                tp = pos.get("tp")
                sl = pos.get("sl")
                if pos["position"] == "long":
                    if sl is not None and lows[j] <= sl:
                        pm.close_position(ticket, sl, date)
                        continue
                    if tp is not None and highs[j] >= tp:
                        pm.close_position(ticket, tp, date)
                else:
                    if sl is not None and highs[j] >= sl:
                        pm.close_position(ticket, sl, date)
                        continue
                    if tp is not None and lows[j] <= tp:
                        pm.close_position(ticket, tp, date)

            # c) Calcular equity y balance
            bal = pm.balance
            eq = bal
            lots = 0
            for pos in pm.positions.values():
                j = pos["sym_idx"]
                cp = closes[j]
                if cp == 0:
                    continue
                diff = pos["dir"] * (cp - pos["entry_price"])
                eq += (diff / pos["point"]) * pos["tick_value"] * pos["lot_size"]
                lots += pos["lot_size"]

            equity_arr[i] = eq
            balance_arr[i] = bal
            open_trades_arr[i] = len(pm.positions)
            open_lots_arr[i] = lots

            # Actualizar progreso
            if nucleo_id is None:
                prog.update(i + 1)
            else:
                if (i % sample_interval == 0) or (i == n_steps - 1):
                    progress_updates[nucleo_id] = {"progress": i + 1, "total": n_steps}

        # Detener barra local
        if nucleo_id is None:
            prog.stop()

        # --- 6) Recoger series de equity_over_time ---
        if full_output:
            df_out = pd.DataFrame(
                {
                    "date": dates,
                    "equity": equity_arr,
                    "balance": balance_arr,
                    "open_trades": open_trades_arr,
                    "open_lots": open_lots_arr,
                }
            )
            self.equity_over_time = df_out.to_dict("records")
        else:
            sampled = []
            for idx in range(0, n_steps, sample_interval):
                sampled.append(
                    {
                        "date": dates[idx],
                        "equity": equity_arr[idx],
                        "balance": balance_arr[idx],
                        "open_trades": open_trades_arr[idx],
                        "open_lots": open_lots_arr[idx],
                    }
                )
            if (n_steps - 1) % sample_interval != 0:
                sampled.append(
                    {
                        "date": dates[-1],
                        "equity": equity_arr[-1],
                        "balance": balance_arr[-1],
                        "open_trades": open_trades_arr[-1],
                        "open_lots": open_lots_arr[-1],
                    }
                )
            self.equity_over_time = sampled

        # --- 7) Estadísticas ---
        stats = Statistics(
            pm.results, self.equity_over_time, self.config.initial_balance
        ).calculate_statistics()

        result = {"statistics": copy.deepcopy(stats)}
        if full_output:
            result["trades"] = copy.deepcopy(pm.results)
            result["equity_over_time"] = copy.deepcopy(self.equity_over_time)
        return result


def _run_single_backtest_parallel(args):
    config, data, params, full_out, nid, prog_upd = args
    engine = OptimizationEngine(config, data)
    sig = config.strategy_signal_class(data)
    sig.set_optimized_params(params)
    res = engine._run_single_backtest(
        data,
        sig,
        full_output=full_out,
        nucleo_id=nid,
        progress_updates=prog_upd,
    )
    res["optimized_params"] = params
    return res
