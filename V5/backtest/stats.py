# backtest/stats.py
import numpy as np


class Statistics:
    def __init__(self, closed_trades, equity_over_time, initial_balance):
        self.closed_trades = closed_trades
        self.equity_over_time = equity_over_time
        self.initial_balance = initial_balance

    def calculate_statistics(self):
        if not self.closed_trades:
            return {}

        # Ganancias y pérdidas
        profits, losses = self._separate_profits_losses()

        total_profit = sum(profits)
        total_loss = sum(losses)
        net_profit = total_profit + total_loss

        # Cantidad de operaciones ganadoras y perdedoras
        win_trades = len(profits)
        lose_trades = len(losses)
        total_trades = win_trades + lose_trades

        # Winrate (porcentaje de operaciones ganadoras)
        winrate = (win_trades / total_trades) * 100 if total_trades > 0 else 0

        # Profit factor (ganancia total / pérdida total)
        profit_factor = (
            abs(total_profit / total_loss) if total_loss != 0 else float("inf")
        )

        # Calcular el Maximal Drawdown y el Relative Drawdown en base a la equidad
        equity_maximal_drawdown, equity_relative_drawdown = self._calculate_drawdowns()

        # Recovery factor (ganancia neta / drawdown máximo absoluto)
        recovery_factor = (
            net_profit / equity_maximal_drawdown
            if equity_maximal_drawdown > 0
            else float("inf")
        )

        # Mejor y peor operación
        best_trade = max(profits) if profits else 0
        worst_trade = min(losses) if losses else 0

        # Promedio de ganancia/pérdida por operación
        avg_profit_per_trade = (total_profit / win_trades) if win_trades > 0 else 0
        avg_loss_per_trade = (total_loss / lose_trades) if lose_trades > 0 else 0

        # Desviación estándar de las ganancias
        profit_std_dev = np.std(
            [trade.get("profit", 0) for trade in self.closed_trades]
        )

        # Ratio Ganancia/Pérdida (Promedio de ganancias / Promedio de pérdidas)
        ratio_gain_loss = (
            avg_profit_per_trade / abs(avg_loss_per_trade)
            if avg_loss_per_trade != 0
            else float("inf")
        )

        # Balance inicial y final
        final_balance = self.initial_balance + net_profit

        # Contar operaciones Buy y Sell
        buy_trades = len(
            [trade for trade in self.closed_trades if trade.get("type") == "long"]
        )
        sell_trades = len(
            [trade for trade in self.closed_trades if trade.get("type") == "short"]
        )

        # Calcular Value at Risk (VaR) al 95% utilizando retornos logarítmicos
        confidence_level = 95
        alpha = 100 - confidence_level  # Por ejemplo, 5 para un nivel del 95%
        equity_values = np.array([record["equity"] for record in self.equity_over_time])
        if len(equity_values) > 1:
            # Evitar valores <= 0 para el cálculo del logaritmo
            safe_equity = np.clip(equity_values, 1e-6, None)
            log_returns = np.diff(np.log(safe_equity))
            var_log = np.percentile(log_returns, alpha)
            var_pct = 1 - np.exp(var_log)
            var_95 = final_balance * var_pct
        else:
            var_95 = 0.0

        return {
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "net_profit": net_profit,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "equity_maximal_drawdown": equity_maximal_drawdown,
            "equity_relative_drawdown": equity_relative_drawdown,
            "recovery_factor": recovery_factor,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_profit_per_trade": avg_profit_per_trade,
            "avg_loss_per_trade": avg_loss_per_trade,
            "profit_std_dev": profit_std_dev,
            "ratio_gain_loss": ratio_gain_loss,
            "var_95": var_95,
        }

    def _separate_profits_losses(self):
        profits = [
            trade.get("profit", 0)
            for trade in self.closed_trades
            if trade.get("profit", 0) > 0
        ]
        losses = [
            trade.get("profit", 0)
            for trade in self.closed_trades
            if trade.get("profit", 0) < 0
        ]
        return profits, losses

    def _calculate_drawdowns(self):
        equity_values = [record["equity"] for record in self.equity_over_time]

        max_equity = equity_values[0]
        max_drawdown_abs = 0
        max_drawdown_pct = 0

        for equity in equity_values:
            if equity > max_equity:
                max_equity = equity
            drawdown_abs = max_equity - equity
            drawdown_pct = drawdown_abs / max_equity if max_equity > 0 else 0

            if drawdown_abs > max_drawdown_abs:
                max_drawdown_abs = drawdown_abs
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

        return max_drawdown_abs, max_drawdown_pct * 100
