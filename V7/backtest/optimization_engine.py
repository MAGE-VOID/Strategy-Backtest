# backtest/optimization_engine.py
import numpy as np
import pandas as pd
import copy
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.utils.barprogress_multi import MultiProgress  # Nuestro nuevo módulo


class OptimizationEngine:
    """
    Motor de optimización que maneja el proceso de backtesting con múltiples combinaciones de parámetros.
    """

    def __init__(self, config, input_data: pd.DataFrame):
        self.config = config
        self.input_data = input_data
        self.best_result = None
        self.best_stats = None
        self.equity_over_time = []

    def generate_combinations(self, optimization_params):
        param_values = []
        for i, (param_key, param_info) in enumerate(
            optimization_params.items(), start=1
        ):
            param_type = param_info.get("type", "float")
            if param_type in ["int", "float"]:
                start = param_info.get("start")
                stop = param_info.get("stop")
                step = param_info.get("step")
                if start is None or stop is None or step is None:
                    raise ValueError(
                        f"Parámetros de rango no definidos correctamente para {param_key}"
                    )
                generated_values = np.arange(
                    start, stop, step, dtype=float if param_type == "float" else int
                )
                print(
                    f"Lista {i} para {param_key} (tipo {param_type}): {generated_values}"
                )
                param_values.append(generated_values)
            elif param_type == "bool":
                generated_values = [True, False]
                print(f"Lista {i} para {param_key} (tipo bool): {generated_values}")
                param_values.append(generated_values)
            else:
                raise ValueError(
                    f"Tipo de parámetro desconocido para {param_key}: {param_type}"
                )
        return list(product(*param_values))

    def run_optimization(self):
        # Se crea una instancia de la señal para obtener los parámetros de optimización
        strategy_signal = self.config.strategy_signal_class(self.input_data)
        optimization_params = strategy_signal.optimization_params

        param_combinations = self.generate_combinations(optimization_params)
        total_combinations = len(param_combinations)
        print(f"\nTotal de combinaciones de parámetros: {total_combinations}\n")

        preprocessed_data = self._preprocess_data(self.input_data.copy())

        results = []
        # Usamos un Manager para compartir el diccionario de progreso entre procesos
        with multiprocessing.Manager() as manager:
            progress_updates = (
                manager.dict()
            )  # Cada tarea actualizará: { task_id: {"progress": x, "total": y} }
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = []
                # Asignamos un id único (que usaremos para la barra individual) a cada combinación
                for idx, param_values in enumerate(param_combinations):
                    params_dict = dict(zip(optimization_params.keys(), param_values))
                    task_id = idx  # identificador único para la tarea
                    futures.append(
                        executor.submit(
                            _run_single_backtest_parallel,
                            (
                                self.config,
                                preprocessed_data,
                                params_dict,
                                False,
                                task_id,
                                progress_updates,
                            ),
                        )
                    )

                # Creamos la barra de progreso multi-núcleo
                multi_progress = MultiProgress(total_tasks=total_combinations)
                multi_progress.start()
                # Bucle de monitoreo: se actualiza la barra general y las individuales
                while True:
                    done_count = sum([future.done() for future in futures])
                    multi_progress.update_overall(done_count)
                    # Para cada tarea que haya iniciado (clave presente en progress_updates)
                    for task_id, update in progress_updates.items():
                        current = update.get("progress", 0)
                        total = update.get("total", 1)
                        # Si no se ha creado aún la barra individual para ese task_id, la agregamos
                        if task_id not in multi_progress.worker_tasks:
                            multi_progress.add_worker(
                                task_id, f"Tarea {task_id}", total=total
                            )
                        multi_progress.update_worker(task_id, current)
                    if done_count == total_combinations:
                        break
                    time.sleep(0.5)
                multi_progress.stop()

                # Se recogen los resultados de cada tarea conforme van terminando
                for future in as_completed(futures):
                    result_backtest = future.result()
                    results.append(result_backtest)

        # Se determina el mejor resultado basándose en el "Win Rate [%]"
        best_backtest_result = None
        best_win_rate = -float("inf")
        best_params = None
        for res in results:
            win_rate = res["statistics"].get("Win Rate [%]", 0)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_backtest_result = res
                best_params = res["optimized_params"]

        # Una vez obtenida la mejor combinación, se realiza un backtest final con salida completa
        final_engine = OptimizationEngine(self.config, preprocessed_data)
        final_strategy_signal = self.config.strategy_signal_class(preprocessed_data)
        final_strategy_signal.set_optimized_params(best_params)
        final_result = final_engine._run_single_backtest(
            preprocessed_data, final_strategy_signal, full_output=True
        )
        final_result["optimized_params"] = best_params

        return final_result

    def update_best_result(self, result_backtest: dict) -> dict:
        stats = result_backtest["statistics"]
        if not isinstance(stats, dict):
            print("Error: las estadísticas no están en formato diccionario.")
            return self.best_result

        win_rate = stats.get("Win Rate [%]", 0)
        if self.best_result is None or win_rate > self.best_stats:
            self.best_result = {
                "trades": result_backtest["trades"],
                "equity_over_time": result_backtest["equity_over_time"],
                "statistics": result_backtest["statistics"],
            }
            self.best_stats = win_rate

        return self.best_result

    def _run_single_backtest(
        self,
        input_data: pd.DataFrame,
        strategy_signal,
        full_output=False,
        task_id=None,
        progress_updates=None,
    ) -> dict:
        symbol_points_mapping = self._build_symbol_points_mapping(input_data)
        self._init_managers(symbol_points_mapping)
        self.equity_over_time.clear()

        # Preparar datos
        all_dates, filled_data, signals = self._prepare_data(
            input_data, lambda df: strategy_signal
        )
        symbol_lengths = {symbol: arr.shape[0] for symbol, arr in filled_data.items()}

        total_steps = len(all_dates)
        # Inicializa el progreso para esta tarea, si se ha proporcionado el diccionario compartido
        if progress_updates is not None and task_id is not None:
            progress_updates[task_id] = {"progress": 0, "total": total_steps}

        # Bucle principal: iterar sobre cada vela
        for i, date in enumerate(all_dates):
            current_prices = {
                symbol: arr[i]
                for symbol, arr in filled_data.items()
                if i < symbol_lengths[symbol]
            }
            if not current_prices:
                continue

            for symbol, price in current_prices.items():
                signal_buy, signal_sell = signals[symbol].generate_signals_for_candle(i)
                self.strategy_manager.manage_tp_sl(symbol, price, date)
                self.strategy_manager.apply_strategy(
                    self.config.strategy_name,
                    symbol,
                    signal_buy,
                    signal_sell,
                    price,
                    i,
                    date,
                )

            self._update_equity(current_prices, date)
            # Actualizamos el progreso de esta tarea
            if progress_updates is not None and task_id is not None:
                progress_updates[task_id] = {"progress": i + 1, "total": total_steps}

        # Cálculo de estadísticas
        statistics_calculator = Statistics(
            self.strategy_manager.get_results(),
            self.equity_over_time,
            self.config.initial_balance,
        )
        stats = statistics_calculator.calculate_statistics()

        if full_output:
            trades_copy = copy.deepcopy(self.strategy_manager.get_results())
            equity_copy = copy.deepcopy(self.equity_over_time)
            stats_copy = copy.deepcopy(stats)
            return {
                "trades": trades_copy,
                "equity_over_time": equity_copy,
                "statistics": stats_copy,
            }
        else:
            stats_copy = copy.deepcopy(stats)
            return {"statistics": stats_copy}

    # ------------------- Funciones Auxiliares ------------------- #

    def _build_symbol_points_mapping(self, input_data: pd.DataFrame) -> dict:
        symbol_mapping = {}
        for symbol, group in input_data.groupby("Symbol"):
            point = group["Point"].iloc[0]
            tick_value = group["Tick_Value"].iloc[0]
            symbol_mapping[symbol] = {"point": point, "tick_value": tick_value}
        return symbol_mapping

    def _init_managers(self, symbol_points_mapping: dict):
        self.strategy_manager = EntryManager(
            self.config.initial_balance, symbol_points_mapping=symbol_points_mapping
        )

    def _preprocess_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        input_data.sort_index(inplace=True)
        return input_data

    def _prepare_data(self, input_data: pd.DataFrame, strategy_signal_factory):
        all_dates = input_data.index.unique()
        grouped_data = input_data.groupby("Symbol")
        filled_data = {}
        signal_generators = {}

        for symbol, group in grouped_data:
            filled_data[symbol] = group["Open"].values
            signal_generators[symbol] = strategy_signal_factory(group)

        return all_dates, filled_data, signal_generators

    def _update_equity(self, current_prices: dict, date: datetime):
        positions = self.strategy_manager.get_positions()
        balance = self.strategy_manager.get_balance()
        equity = self._calculate_equity(positions, current_prices, balance)
        open_trades = len(positions)
        open_lots = sum(pos["lot_size"] for pos in positions.values())

        self.equity_over_time.append(
            {
                "date": date,
                "equity": equity,
                "balance": balance,
                "open_trades": open_trades,
                "open_lots": open_lots,
            }
        )

    def _calculate_equity(
        self, positions: dict, current_prices: dict, balance: float
    ) -> float:
        symbol_points_map = self.strategy_manager.symbol_points_mapping
        equity = balance

        for pos in positions.values():
            symbol = pos["symbol"]
            cp = current_prices.get(symbol, 0)
            if cp == 0:
                continue

            point = symbol_points_map[symbol]["point"]
            tick_value = symbol_points_map[symbol]["tick_value"]

            if pos["position"] == "long":
                price_diff = cp - pos["entry_price"]
            else:
                price_diff = pos["entry_price"] - cp

            floating_profit = (price_diff / point) * tick_value * pos["lot_size"]
            equity += floating_profit

        return equity


# Función auxiliar para ejecución en paralelo.
# Recibe una tupla con (config, input_data, params_dict, full_output, task_id, progress_updates),
# crea una instancia nueva del motor, establece los parámetros optimizados y ejecuta el backtest.
def _run_single_backtest_parallel(args):
    config, input_data, params_dict, full_output, task_id, progress_updates = args
    engine = OptimizationEngine(config, input_data)
    strategy_signal = config.strategy_signal_class(input_data)
    strategy_signal.set_optimized_params(params_dict)
    result = engine._run_single_backtest(
        input_data,
        strategy_signal,
        full_output=full_output,
        task_id=task_id,
        progress_updates=progress_updates,
    )
    result["optimized_params"] = params_dict
    return result
