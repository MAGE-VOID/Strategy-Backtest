import numpy as np
import pandas as pd
import copy
from datetime import datetime
from itertools import product
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.progress import BarProgress


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

    def run_optimization(self):
        strategy_signal = self.config.strategy_signal_class(self.input_data)
        optimization_params = strategy_signal.optimization_params

        param_combinations = self.generate_combinations(optimization_params)
        print(f"\nTotal de combinaciones de parámetros: {len(param_combinations)}\n")

        preprocessed_data = self._preprocess_data(self.input_data.copy())

        best_backtest_result = None

        for idx, param_values in enumerate(param_combinations, start=1):
            print(f"Combinación {idx}: {param_values}")
            params_dict = dict(zip(optimization_params.keys(), param_values))
            strategy_signal.set_optimized_params(params_dict)

            result_backtest = self._run_single_backtest(
                preprocessed_data, strategy_signal
            )
            best_backtest_result = self.update_best_result(result_backtest)

        return {
            "trades": best_backtest_result["trades"],
            "equity_over_time": best_backtest_result["equity_over_time"],
            "statistics": best_backtest_result["statistics"],
        }

    def _run_single_backtest(self, input_data: pd.DataFrame, strategy_signal) -> dict:
        symbol_points_mapping = self._build_symbol_points_mapping(input_data)
        self._init_managers(symbol_points_mapping)
        self.equity_over_time.clear()

        # Preparar datos
        all_dates, filled_data, signals = self._prepare_data(
            input_data, lambda df: strategy_signal
        )
        symbol_lengths = {symbol: arr.shape[0] for symbol, arr in filled_data.items()}

        total_steps = len(all_dates)
        progress_bar = BarProgress(total_steps)
        progress_current = 0

        # Iteramos cada vela
        for i, date in enumerate(all_dates):
            current_prices = {
                symbol: arr[i]
                for symbol, arr in filled_data.items()
                if i < symbol_lengths[symbol]
            }
            if not current_prices:
                continue

            # Para cada símbolo, generamos señales y aplicamos la estrategia
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

            # Actualizamos la equidad
            self._update_equity(current_prices, date)

            # Barra de progreso
            progress_current += 1
            progress_bar.update(progress_current + 1)

        progress_bar.stop()

        # Calcular estadísticas
        statistics_calculator = Statistics(
            self.strategy_manager.get_results(),
            self.equity_over_time,
            self.config.initial_balance,
        )
        stats = statistics_calculator.calculate_statistics()

        trades_copy = copy.deepcopy(self.strategy_manager.get_results())
        equity_copy = copy.deepcopy(self.equity_over_time)
        stats_copy = copy.deepcopy(stats)

        return {
            "trades": trades_copy,
            "equity_over_time": equity_copy,
            "statistics": stats_copy,
        }

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

        self.equity_over_time.append(
            {"date": date, "equity": equity, "balance": balance}
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
