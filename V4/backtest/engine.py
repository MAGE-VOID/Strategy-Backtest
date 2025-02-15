# backtest/engine.py
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager


class BacktestEngine:
    """
    Clase que gestiona la ejecución del backtest.
    """

    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance
        self.strategy_manager = None
        self.equity_over_time = []
        self.debug_mode = "none"  # "none", "final" o "realtime"

    def run_backtest(
        self,
        input_data: pd.DataFrame,
        strategy_name: str = "grid_buy",
        strategy_signal_class=None,
        debug_mode: str = "none",  # nuevos: "none", "final" o "realtime"
    ) -> dict:
        self.debug_mode = debug_mode
        self._initialize_managers()
        self.equity_over_time.clear()

        # Preprocesamiento
        input_data = self._preprocess_data(input_data)
        all_dates, filled_data, signals = self._prepare_data(
            input_data, strategy_signal_class
        )

        if signals is None:
            print("No se pasó ninguna clase de señales. Terminando el backtest.")
            return {}

        symbol_lengths = {symbol: arr.shape[0] for symbol, arr in filled_data.items()}
        strategy_manager = self.strategy_manager
        total_steps = len(all_dates)

        prev_trade_count = 0  # Para el modo "realtime"

        with tqdm(
            total=total_steps, desc="Running backtest", unit="step", ascii=True
        ) as pbar:
            for i, date in enumerate(all_dates):
                current_prices = {
                    symbol: arr[i]
                    for symbol, arr in filled_data.items()
                    if i < symbol_lengths[symbol]
                }
                if not current_prices:
                    pbar.update(1)
                    continue

                for symbol, price in current_prices.items():
                    strategy_obj = signals[symbol]
                    signal_buy, signal_sell = strategy_obj.generate_signals_for_candle(
                        i
                    )
                    strategy_manager.manage_tp_sl(symbol, price, date)
                    strategy_manager.apply_strategy(
                        strategy_name, symbol, signal_buy, signal_sell, price, i, date
                    )

                self._update_equity(current_prices, date)
                pbar.update(1)

                # Si el modo es "realtime", imprime los nuevos eventos (trades) desde la última iteración
                if self.debug_mode == "realtime":
                    current_trade_count = len(strategy_manager.get_results())
                    if current_trade_count > prev_trade_count:
                        for trade in strategy_manager.get_results()[prev_trade_count:]:
                            print(trade)
                        prev_trade_count = current_trade_count

        # Si se escogió "final", se imprimen todos los eventos al final.
        if self.debug_mode == "final":
            self._debug_positions()

        statistics_calculator = Statistics(
            strategy_manager.get_results(), self.equity_over_time, self.initial_balance
        )
        statistics = statistics_calculator.calculate_statistics()

        return {
            "trades": strategy_manager.get_results(),
            "equity_over_time": self.equity_over_time,
            "statistics": statistics,
        }

    def _initialize_managers(self):
        self.strategy_manager = EntryManager(self.initial_balance)

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

    def _debug_positions(self):
        for trade in self.strategy_manager.get_results():
            print(trade)

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
