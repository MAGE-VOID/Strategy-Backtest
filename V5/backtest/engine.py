# backtest/engine.py
import numpy as np
import pandas as pd
from datetime import datetime
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.config import BacktestConfig
from barprogress import init_progress_bar, update_progress_bar, stop_progress_bar


class BacktestEngine:
    """
    Motor del backtest que utiliza un objeto de configuración para centralizar parámetros.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy_manager = None
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
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

        init_progress_bar(total_steps)
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

            if self.config.debug_mode == "realtime":
                current_trade_count = len(self.strategy_manager.get_results())
                if current_trade_count > prev_trade_count:
                    for trade in self.strategy_manager.get_results()[prev_trade_count:]:
                        print(trade)
                    prev_trade_count = current_trade_count

            progress_current += 1
            update_progress_bar(progress_current)

        stop_progress_bar()

        if self.config.debug_mode == "final":
            self._debug_positions()

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
        """
        Construye un diccionario que mapea cada símbolo a sus valores de point y tick_value.
        Ejemplo:
            {
                'EURUSD': {
                    'point': 1e-5,
                    'tick_value': 1.0
                },
                'GBPUSD': {
                    'point': 1e-5,
                    'tick_value': 1.2
                },
                ...
            }
        """
        symbol_mapping = {}
        for symbol, group in input_data.groupby("Symbol"):
            point = group["Point"].iloc[0]
            tick_value = group["Tick_Value"].iloc[0]  # <-- Asegúrate de haberlo descargado
            symbol_mapping[symbol] = {
                "point": point,
                "tick_value": tick_value
            }
        return symbol_mapping


    def _init_managers(self, symbol_points_mapping: dict):
        self.strategy_manager = EntryManager(
            self.config.initial_balance,
            symbol_points_mapping=symbol_points_mapping
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
        equity = self._calculate_equity(current_prices)
        self.equity_over_time.append(
            {
                "date": date,
                "equity": equity,
                "balance": self.strategy_manager.get_balance(),
            }
        )

    def _calculate_equity(self, current_prices: dict) -> float:
        equity = self.strategy_manager.get_balance()
        positions = self.strategy_manager.get_positions()
        symbol_points_map = self.strategy_manager.symbol_points_mapping  # Para fácil acceso

        for pos in positions.values():
            symbol = pos["symbol"]
            cp = current_prices.get(symbol, 0)  # current_price
            if cp:
                point = symbol_points_map[symbol]["point"]
                tick_value = symbol_points_map[symbol]["tick_value"]
                if pos["position"] == "long":
                    price_diff = cp - pos["entry_price"]
                else:  # short
                    price_diff = pos["entry_price"] - cp

                floating_profit = (price_diff / point) * tick_value * pos["lot_size"]
                equity += floating_profit

        return equity

    def _debug_positions(self):
        for trade in self.strategy_manager.get_results():
            print(trade)
