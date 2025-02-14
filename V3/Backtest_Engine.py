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

        # Preprocesamiento (ordenar, etc.)
        InputData = self._preprocess_data(InputData)
        all_dates, filled_data, signals = self._prepare_data(
            InputData, strategy_signal_class
        )

        if signals is None:
            print("No se pasó ninguna clase de señales. Terminando el backtest.")
            return

        # Precalcular las longitudes de cada serie para evitar múltiples llamadas a len()
        symbol_lengths = {symbol: arr.shape[0] for symbol, arr in filled_data.items()}

        # Guardar en variables locales objetos usados en el loop
        strategy_manager = self.strategy_manager
        total_steps = len(all_dates)

        with tqdm(
            total=total_steps, desc="Running backtest", unit="step", ascii=True
        ) as pbar:
            for i, date in enumerate(all_dates):
                # Obtención de precios actuales usando los arrays y las longitudes precalculadas
                current_prices = {
                    symbol: arr[i]
                    for symbol, arr in filled_data.items()
                    if i < symbol_lengths[symbol]
                }
                if not current_prices:
                    pbar.update(1)
                    continue

                # Procesamos señales y ejecutamos estrategias
                for symbol, price in current_prices.items():
                    # Se guarda en variable local el objeto de señales para evitar búsquedas repetidas
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

        if DebugPositions:
            self._debug_positions()

        statistics_calculator = Statistics(
            strategy_manager.get_results(),
            self.equity_over_time,
            self.initial_balance,
        )
        statistics = statistics_calculator.calculate_statistics()

        return {
            "trades": strategy_manager.get_results(),
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
        filled_data = {}
        signal_generators = {} if strategy_signal_class else None

        for symbol, group in grouped_data:
            filled_data[symbol] = group["Open"].values
            if strategy_signal_class:
                signal_generators[symbol] = strategy_signal_class(group)

        return all_dates, filled_data, signal_generators


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
            cp = current_prices.get(pos["symbol"], 0)
            if cp:
                floating_profit = (
                    (cp - pos["entry_price"]) * pos["lot_size"]
                    if pos["position"] == "long"
                    else (pos["entry_price"] - cp) * pos["lot_size"]
                )
                equity += floating_profit
        return equity
