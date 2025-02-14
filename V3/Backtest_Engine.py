import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import Entry_Strategy_Manager as ES
from Statistics import Statistics


class BacktestEngine:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.strategy_manager = None
        self.equity_over_time = []

    def run_backtest(
        self,
        InputData,
        strategy_name="grid_buy",
        strategy_signal_class=None,
        DebugPositions=False,
    ):
        self._initialize_managers()
        self.equity_over_time.clear()

        InputData = self._preprocess_data(InputData)
        all_dates, filled_data, signals = self._prepare_data(
            InputData, strategy_signal_class
        )

        if signals is None:
            print("No se pasó ninguna clase de señales. Terminando el backtest.")
            return

        with tqdm(
            total=len(all_dates), desc="Running backtest", unit="step", ascii=True
        ) as pbar:
            for i, date in enumerate(all_dates):
                current_prices = self._get_current_prices(filled_data, i)
                if not current_prices:
                    pbar.update(1)
                    continue

                self._process_signals_and_apply_strategy(
                    strategy_name, signals, current_prices, date, i
                )

                self._update_equity(current_prices, date)
                pbar.update(1)

        if DebugPositions:
            self._debug_positions()

        statistics_calculator = Statistics(
            self.strategy_manager.get_results(),
            self.equity_over_time,
            self.initial_balance,
        )
        statistics = statistics_calculator.calculate_statistics()

        return {
            "trades": self.strategy_manager.get_results(),
            "equity_over_time": self.equity_over_time,
            "statistics": statistics,
        }

    def _initialize_managers(self):
        self.strategy_manager = ES.EntryManager(self.initial_balance)

    def _preprocess_data(self, InputData):
        InputData.sort_index(inplace=True)
        return InputData

    def _prepare_data(self, InputData, strategy_signal_class):
        """Genera los datos y señales utilizando la clase de señal proporcionada."""
        all_dates = InputData.index.unique()
        grouped_data = InputData.groupby("Symbol")
        filled_data = {symbol: group["Open"].values for symbol, group in grouped_data}

        if strategy_signal_class:
            signal_generators = {
                symbol: strategy_signal_class(group) for symbol, group in grouped_data
            }

            # Modificación aquí: Se cambió `generate_signals()` por `generate_signals_for_candle(index)`
            signals = {
                symbol: signal_generators[symbol] for symbol in signal_generators.keys()
            }
            return all_dates, filled_data, signals

        return all_dates, filled_data, None


    def _get_current_prices(self, filled_data, index):
        return {
            symbol: filled_data[symbol][index]
            for symbol in filled_data
            if index < len(filled_data[symbol])
        }

    def _process_signals_and_apply_strategy(
        self, strategy_name, signals, current_prices, date, index
    ):
        for symbol, price in current_prices.items():
            # Llamada a StrategySignal para generar las señales de compra y venta
            signal_buy, signal_sell = signals[symbol].generate_signals_for_candle(index)
            self.strategy_manager.manage_tp_sl(symbol, price, date)
            self.strategy_manager.apply_strategy(
                strategy_name, symbol, signal_buy, signal_sell, price, index, date
            )


    def _update_equity(self, current_prices, date):
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

    def _calculate_equity(self, current_prices):
        """Calcula el equity basado en las posiciones abiertas y los precios actuales."""
        equity = self.strategy_manager.get_balance()
        positions = self.strategy_manager.get_positions()

        for pos in positions.values():
            current_price = current_prices.get(pos["symbol"], 0)
            if current_price:
                floating_profit = (
                    (current_price - pos["entry_price"]) * pos["lot_size"]
                    if pos["position"] == "long"
                    else (pos["entry_price"] - current_price) * pos["lot_size"]
                )
                equity += floating_profit
        return equity
