from tqdm import tqdm
from strategy.entry_strategy_manager import EntryManager
from statistics import Statistics

class BacktestEngine:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.strategy_manager = None
        self.equity_over_time = []

    def run(self, input_data, strategy_name="grid_buy", strategy_signal_class=None, debug_positions=False):
        self._initialize_managers()
        self.equity_over_time.clear()

        input_data = self._preprocess_data(input_data)
        all_dates, filled_data, signals = self._prepare_data(input_data, strategy_signal_class)

        if signals is None:
            raise ValueError("No signals class passed. Ending backtest.")
        
        with tqdm(total=len(all_dates), desc="Running backtest", unit="step", ascii=True) as pbar:
            for i, date in enumerate(all_dates):
                current_prices = self._get_current_prices(filled_data, i)
                if not current_prices:
                    pbar.update(1)
                    continue

                self._process_signals_and_apply_strategy(strategy_name, signals, current_prices, date, i)
                self._update_equity(current_prices, date)
                pbar.update(1)

        if debug_positions:
            self._debug_positions()

        return self._generate_statistics()

    def _initialize_managers(self):
        self.strategy_manager = EntryManager(self.initial_balance)

    def _preprocess_data(self, input_data):
        input_data.sort_index(inplace=True)
        return input_data

    def _prepare_data(self, input_data, strategy_signal_class):
        all_dates = input_data.index.unique()
        grouped_data = input_data.groupby("Symbol")
        filled_data = {symbol: group["Open"].values for symbol, group in grouped_data}

        signals = {symbol: strategy_signal_class(group).generate_signals() for symbol, group in grouped_data} if strategy_signal_class else None
        return all_dates, filled_data, signals

    def _get_current_prices(self, filled_data, index):
        return {symbol: filled_data[symbol][index] for symbol in filled_data if index < len(filled_data[symbol])}

    def _process_signals_and_apply_strategy(self, strategy_name, signals, current_prices, date, index):
        for symbol, price in current_prices.items():
            signal_buy, signal_sell = signals.get(symbol, (None, None))
            self.strategy_manager.manage_tp_sl(symbol, price, date)
            self.strategy_manager.apply_strategy(strategy_name, symbol, signal_buy, signal_sell, price, index, date)

    def _update_equity(self, current_prices, date):
        equity = self._calculate_equity(current_prices)
        self.equity_over_time.append({"date": date, "equity": equity, "balance": self.strategy_manager.get_balance()})

    def _debug_positions(self):
        for trade in self.strategy_manager.get_results():
            print(trade)

    def _calculate_equity(self, current_prices):
        equity = self.strategy_manager.get_balance()
        positions = self.strategy_manager.get_positions()
        for pos in positions.values():
            current_price = current_prices.get(pos["symbol"], 0)
            if current_price:
                floating_profit = (current_price - pos["entry_price"]) * pos["lot_size"] if pos["position"] == "long" else (pos["entry_price"] - current_price) * pos["lot_size"]
                equity += floating_profit
        return equity

    def _generate_statistics(self):
        stats_calculator = Statistics(self.strategy_manager.get_results(), self.equity_over_time, self.initial_balance)
        return stats_calculator.calculate_statistics()
