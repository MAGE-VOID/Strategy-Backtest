import numpy as np
import pandas as pd
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
        """
        Genera todas las combinaciones posibles de los parámetros de optimización.
        """
        param_1_values = np.arange(
            optimization_params["params_1"]["start"],
            optimization_params["params_1"]["stop"],
            optimization_params["params_1"]["step"],
        )
        param_2_values = np.arange(
            optimization_params["params_2"]["start"],
            optimization_params["params_2"]["stop"],
            optimization_params["params_2"]["step"],
        )

        # Genera todas las combinaciones posibles entre los parámetros
        return list(product(param_1_values, param_2_values))

    def run_optimization(self):
        """
        Ejecuta la optimización: evalúa todas las combinaciones de parámetros.
        """
        # Obtiene los parámetros de optimización de la clase StrategySignal
        strategy_signal = self.config.strategy_signal_class(self.input_data)
        optimization_params = strategy_signal.get_optimization_params()

        # Genera todas las combinaciones posibles de parámetros
        param_combinations = self.generate_combinations(optimization_params)

        print(f"Total de combinaciones de parámetros: {len(param_combinations)}")

        # Listado para almacenar los resultados de cada backtest
        results = []

        for param_1_value, param_2_value in param_combinations:
            # Establecer los parámetros optimizados en la estrategia
            strategy_signal.set_optimized_params(
                {"params_1": param_1_value, "params_2": param_2_value}
            )

            # Realizar el backtest con los parámetros optimizados
            result_backtest = self._run_single_backtest(self.input_data)

            # Guardar el resultado de este backtest
            results.append(result_backtest)

            # Extraer estadísticas y comparar si es el mejor resultado
            stats = result_backtest["statistics"]

            # Verificar si stats es un diccionario
            if isinstance(stats, dict):
                win_rate = stats.get("Win Rate [%]", 0)
            else:
                print("Error: las estadísticas no están en formato diccionario.")
                continue  # Saltamos a la siguiente iteración del backtest

            # Comparar con el mejor resultado basado en Win Rate
            if self.best_result is None or win_rate > self.best_stats:
                self.best_result = result_backtest
                self.best_stats = win_rate

        stats = self.best_result["statistics"]

        return {
            "trades": self.best_result["trades"],
            "equity_over_time": self.best_result["equity_over_time"],
            "statistics": stats,
        }

    def _run_single_backtest(self, input_data: pd.DataFrame) -> dict:
        """
        Realiza un backtest simple con los parámetros actuales de la estrategia.
        """
        input_data = self._preprocess_data(input_data)

        symbol_points_mapping = self._build_symbol_points_mapping(input_data)
        self._init_managers(symbol_points_mapping)
        self.equity_over_time.clear()

        all_dates, filled_data, signals = self._prepare_data(
            input_data, self.config.strategy_signal_class
        )
        if signals is None:
            raise ValueError("No se proporcionó una clase para generar señales.")

        symbol_lengths = {symbol: arr.shape[0] for symbol, arr in filled_data.items()}
        total_steps = len(all_dates)
        prev_trade_count = 0

        progress_bar = BarProgress(total_steps)
        progress_current = 0

        # Bucle principal del backtest
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

            progress_current += 1
            progress_bar.update(progress_current + 1)

        progress_bar.stop()

        statistics_calculator = Statistics(
            self.strategy_manager.get_results(),
            self.equity_over_time,
            self.config.initial_balance,
        )
        stats = statistics_calculator.calculate_statistics()

        return {
            "trades": self.strategy_manager.get_results(),
            "equity_over_time": self.equity_over_time,
            "statistics": stats,
        }

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

    def _prepare_data(self, input_data: pd.DataFrame, strategy_signal_class):
        all_dates = input_data.index.unique()
        grouped_data = input_data.groupby("Symbol")
        filled_data = {}
        signal_generators = {} if strategy_signal_class else None

        for symbol, group in grouped_data:
            filled_data[symbol] = group["Open"].values
            if strategy_signal_class:
                signal_generators[symbol] = strategy_signal_class(group)

        return all_dates, filled_data, signal_generators

    def _update_equity(self, current_prices: dict, date: datetime):
        positions = self.strategy_manager.get_positions()
        balance = self.strategy_manager.get_balance()
        equity = self._calculate_equity(positions, current_prices, balance)

        self.equity_over_time.append(
            {
                "date": date,
                "equity": equity,
                "balance": balance,
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
