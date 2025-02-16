# backtest/engine.py
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.config import BacktestConfig  # NUEVA IMPORTACIÓN


class BacktestEngine:
    """
    Motor del backtest que utiliza un objeto de configuración para centralizar parámetros.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy_manager = None
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        # Preprocesar y ordenar datos
        input_data = self._preprocess_data(input_data)

        # Construir el diccionario de puntos (se asume que 'Point' ya está en el DataFrame)
        symbol_points_mapping = self._build_symbol_points_mapping(input_data)
        self._init_managers(symbol_points_mapping)
        self.equity_over_time.clear()

        # Preparar datos y señales
        all_dates, filled_data, signals = self._prepare_data(
            input_data, self.config.strategy_signal_class
        )
        if signals is None:
            raise ValueError("No se proporcionó una clase para generar señales.")

        symbol_lengths = {symbol: arr.shape[0] for symbol, arr in filled_data.items()}
        total_steps = len(all_dates)
        prev_trade_count = 0

        # Bucle principal del backtest
        for i, date in enumerate(
            tqdm(
                all_dates,
                total=total_steps,
                desc="Running backtest",
                unit="step",
                ascii=True,
            )
        ):
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
        """Construye { 'símbolo': point } a partir del DataFrame."""
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
