# backtest/engine.py
import numpy as np
import pandas as pd
from datetime import datetime
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.config import BacktestConfig
from backtest.counter import create_progress_counter


class BacktestEngine:
    """
    Motor del backtest que utiliza un objeto de configuración para centralizar parámetros.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy_manager = None
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame, worker_id: int = 1) -> dict:
        input_data = self._preprocess_data(input_data)
        symbol_points_mapping = self._build_symbol_points_mapping(input_data)
        self._init_managers(symbol_points_mapping)
        self.equity_over_time.clear()

        all_dates, filled_data, signals = self._prepare_data(
            input_data, self.config.strategy_signal_class
        )
        if signals is None:
            raise ValueError("No se proporcionó una clase para generar señales.")

        total_steps = len(all_dates)

        if self.config.mode == "single":
            counter = create_progress_counter(self.config.mode, total_steps)
        else:
            counter = create_progress_counter(self.config.mode, total_steps, worker_id)

        for i, date in enumerate(all_dates):
            for symbol, arr in filled_data.items():
                if i < arr.shape[0]:
                    price = arr[i]
                    signal_buy, signal_sell = signals[
                        symbol
                    ].generate_signals_for_candle(i)
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
            self._update_equity(
                {
                    symbol: arr[i]
                    for symbol, arr in filled_data.items()
                    if i < arr.shape[0]
                },
                date,
            )
            counter.update(1)
        counter.close()

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
        return {
            symbol: group["Point"].iloc[0]
            for symbol, group in input_data.groupby("Symbol")
        }

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
                optimization_params = getattr(self.config, "optimization_params", {})
                signal_generators[symbol] = strategy_signal_class(
                    group, optimization_params=optimization_params
                )
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
        for pos in positions.values():
            cp = current_prices.get(pos["symbol"], 0)
            if cp:
                floating_profit = (
                    (cp - pos["entry_price"]) * pos["lot_size"]
                    if pos["position"] == "long"
                    else (pos["entry_price"] - cp) * pos["lot_size"]
                )
                equity += floating_profit
        return equity

    def _debug_positions(self):
        for trade in self.strategy_manager.get_results():
            print(trade)
