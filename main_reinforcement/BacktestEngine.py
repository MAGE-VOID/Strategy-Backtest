import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import Entry_Strategy_Manager_RF as ES
from sklearn.linear_model import LinearRegression
from statistics import StatisticsCalculator


class PositionManager:
    def __init__(self, balance):
        self.positions = {}
        self.balance = balance
        self.initial_balance = balance
        self.ticket_counter = 0
        self.global_counter = 0
        self.results = []

    def open_position(
        self, symbol, position_type, price, lot_size, sl=None, tp=None, open_date=None
    ):
        open_time = (
            open_date.strftime("%Y-%m-%d %H:%M:%S")
            if open_date
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        position = {
            "symbol": symbol,
            "position": position_type,
            "entry_price": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "open_time": open_time,
            "status": "open",
        }
        self.positions[self.ticket_counter] = position
        self.results.append(
            {
                "count": self.global_counter + 1,
                "symbol": symbol,
                "ticket": self.ticket_counter,
                "type": position_type,
                "entry": price,
                "lot_size": lot_size,
                "sl": sl,
                "tp": tp,
                "open_time": open_time,
                "status": "open",
            }
        )
        self.ticket_counter += 1
        self.global_counter += 1

    def close_position(self, ticket, current_price, close_date):
        position = self.positions.pop(ticket, None)
        if not position or position["status"] == "closed":
            return

        entry_price = position["entry_price"]
        lot_size = position["lot_size"]
        profit = (
            (current_price - entry_price) * lot_size
            if position["position"] == "long"
            else (entry_price - current_price) * lot_size
        )
        self.balance += profit
        close_time = close_date.strftime("%Y-%m-%d %H:%M:%S")
        self.results.append(
            {
                "count": self.global_counter + 1,
                "symbol": position["symbol"],
                "ticket": ticket,
                "type": position["position"],
                "entry": entry_price,
                "close_time": close_time,
                "exit": current_price,
                "profit": profit,
                "balance": self.balance,
                "status": "closed",
            }
        )
        self.global_counter += 1

    def calculate_equity(self, current_prices):
        equity = self.balance
        for pos in self.positions.values():
            symbol = pos["symbol"]
            if symbol in current_prices:
                current_price = current_prices[symbol]
                floating_profit = (
                    (current_price - pos["entry_price"]) * pos["lot_size"]
                    if pos["position"] == "long"
                    else (pos["entry_price"] - current_price) * pos["lot_size"]
                )
                equity += floating_profit
        return equity

    def get_results(self):
        return self.results

    def get_balance(self):
        return self.balance

    def get_positions(self, symbol=None, position_type=None):
        if symbol:
            return {
                ticket: pos
                for ticket, pos in self.positions.items()
                if pos["symbol"] == symbol
                and (position_type is None or pos["position"] == position_type)
            }
        return self.positions

    def calculate_statistics(self, equity_over_time):
        """Calcula las métricas estadísticas del backtest."""
        closed_trades = [trade for trade in self.results if trade["status"] == "closed"]

        if not closed_trades:
            return {}

        initial_balance = self.initial_balance
        final_balance = self.get_balance()

        # Usar el StatisticsCalculator para calcular todas las métricas
        stats_calculator = StatisticsCalculator(equity_over_time)
        return stats_calculator.calculate_statistics(
            closed_trades, initial_balance, final_balance
        )


# Modificaciones necesarias en BacktestEngine.py
class BacktestEngineRL:
    def __init__(self, initial_balance=1000, num_agents=1):
        self.initial_balance = initial_balance
        self.num_agents = num_agents
        self.position_managers = [PositionManager(initial_balance) for _ in range(num_agents)]

    def run_backtest(
        self, InputData, entry_manager, agents, window_size=50, DebugPositions=False
    ):
        equity_over_time = [[] for _ in range(self.num_agents)]
        actions_taken = [[] for _ in range(self.num_agents)]
        all_dates = InputData.index.unique()

        # Agrupar datos por símbolo
        grouped_data = InputData.groupby("Symbol")
        filled_data = {symbol: group["Close"].values for symbol, group in grouped_data}

        # Verificamos que las fechas estén ordenadas
        all_dates = sorted(all_dates)

        with tqdm(
            total=len(all_dates), desc="Running RL Backtest", unit="step", ascii=True
        ) as pbar:
            for i, date in enumerate(all_dates):
                if i < window_size:
                    pbar.update(1)
                    continue

                # Obtener precios actuales
                current_prices = {
                    symbol: filled_data[symbol][i]
                    for symbol in filled_data
                    if i < len(filled_data[symbol])
                }

                # Asegurarse de que todas las ventanas tengan la misma longitud
                valid_symbols = [
                    symbol
                    for symbol in filled_data
                    if len(filled_data[symbol][i - window_size: i]) == window_size
                ]

                # Iterar sobre cada agente
                for agent_idx, agent in enumerate(agents):
                    # Obtener ventana de las últimas 'window_size' velas para el estado
                    state = np.array(
                        [
                            filled_data[symbol][i - window_size: i]
                            for symbol in valid_symbols
                        ]
                    )

                    # El agente toma una acción
                    action = agent.act(state)
                    actions_taken[agent_idx].append(action)

                    # Aplicar la estrategia si la acción es comprar
                    for symbol in valid_symbols:
                        if symbol in current_prices:
                            entry_manager.apply_rl_strategy(
                                symbol, action, current_prices[symbol], date, agent_idx
                            )

                    # Gestionar las posiciones abiertas
                    for symbol in valid_symbols:
                        if symbol in current_prices:
                            entry_manager.manage_tp_sl(symbol, current_prices[symbol], date, agent_idx)

                    # Calcular el equity y guardar el balance para este agente
                    equity = self.position_managers[agent_idx].calculate_equity(current_prices)
                    equity_over_time[agent_idx].append(
                        {
                            "date": date,
                            "equity": equity,
                            "balance": self.position_managers[agent_idx].get_balance(),
                        }
                    )

                pbar.update(1)

        # Ordenar equity_over_time por fecha para cada agente
        for agent_eq in equity_over_time:
            agent_eq.sort(key=lambda x: x["date"])

        if DebugPositions:
            for idx, pm in enumerate(self.position_managers):
                print(f"Agente {idx}:")
                for trade in pm.get_results():
                    print(trade)

        # Calcular estadísticas para cada agente
        all_statistics = []
        for agent_idx, pm in enumerate(self.position_managers):
            statistics = pm.calculate_statistics(equity_over_time[agent_idx])
            all_statistics.append(statistics)

        return {
            "trades": [pm.get_results() for pm in self.position_managers],
            "equity_over_time": equity_over_time,
            "actions_taken": actions_taken,
            "statistics": all_statistics,
        }

