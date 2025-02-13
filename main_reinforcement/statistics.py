import numpy as np
from sklearn.linear_model import LinearRegression


class StatisticsCalculator:
    def __init__(self, equity_over_time):
        self.equity_over_time = equity_over_time

    def calculate_lr_correlation(self, key):
        """Calcula la correlación lineal (LR correlation) sobre una serie temporal (equidad o balance)"""
        time_points = np.array(range(len(self.equity_over_time))).reshape(-1, 1)
        values = np.array([record[key] for record in self.equity_over_time]).reshape(
            -1, 1
        )

        model = LinearRegression()
        model.fit(time_points, values)
        slope = model.coef_[0][0]
        return slope

    def calculate_drawdowns(self):
        """Calcula la serie temporal de drawdowns"""
        drawdowns = []
        max_equity = self.equity_over_time[0]["equity"]

        for record in self.equity_over_time:
            current_equity = record["equity"]
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = max_equity - current_equity
            drawdowns.append(drawdown)

        return drawdowns

    def calculate_drawdown_metrics(self):
        """Calcula métricas avanzadas sobre drawdowns, incluidos picos extremos."""
        equity_values = [record["equity"] for record in self.equity_over_time]
        max_equity = equity_values[0]
        drawdown_depths = []
        drawdown_start = None
        max_drawdown_abs = 0
        equity_maximal_drawdown = (
            0  # Nueva variable para calcular el maximal equity drawdown
        )

        for i, equity in enumerate(equity_values):
            if equity > max_equity:
                max_equity = equity
                if drawdown_start is not None:
                    drawdown_start = None
            else:
                drawdown_abs = max_equity - equity
                if drawdown_start is None:
                    drawdown_start = i
                max_drawdown_abs = max(max_drawdown_abs, drawdown_abs)
                drawdown_depths.append(drawdown_abs)

                # Calcular el maximal equity drawdown
                equity_maximal_drawdown = max(equity_maximal_drawdown, drawdown_abs)

        # Si hay drawdowns, obtenemos los picos más extremos
        if drawdown_depths:
            max_drawdown_peak = max(drawdown_depths)
            top_5_percentile_drawdowns = np.percentile(drawdown_depths, 95)
            drawdown_peak_intensity = (
                max_drawdown_peak / np.mean(drawdown_depths)
                if np.mean(drawdown_depths) != 0
                else 0
            )
        else:
            max_drawdown_peak = 0
            top_5_percentile_drawdowns = 0
            drawdown_peak_intensity = 0

        drawdown_frequency = len(drawdown_depths)
        average_drawdown_depth = np.mean(drawdown_depths) if drawdown_depths else 0
        drawdown_depth_std = np.std(drawdown_depths) if drawdown_depths else 0

        max_possible_drawdown = max(equity_values) - min(equity_values)
        normalized_drawdown_depth = (
            average_drawdown_depth / max_possible_drawdown
            if max_possible_drawdown != 0
            else 0
        )
        normalized_drawdown_std = (
            drawdown_depth_std / max_possible_drawdown
            if max_possible_drawdown != 0
            else 0
        )

        weight_depth = 0.4
        weight_peak_intensity = 0.3
        weight_frequency = 0.2
        weight_std = 0.1

        composite_drawdown_metric = (
            weight_depth * normalized_drawdown_depth
            + weight_peak_intensity * drawdown_peak_intensity
            + weight_frequency * (drawdown_frequency / len(equity_values))
            + weight_std * normalized_drawdown_std
        )

        return {
            "composite_drawdown_metric": composite_drawdown_metric,
            "max_drawdown_peak": max_drawdown_peak,
            "top_5_percentile_drawdowns": top_5_percentile_drawdowns,
            "drawdown_peak_intensity": drawdown_peak_intensity,
            "drawdown_depth_std": drawdown_depth_std,
            "equity_maximal_drawdown": equity_maximal_drawdown,  # Agregar el maximal equity drawdown
        }

    def calculate_statistics(self, closed_trades, initial_balance, final_balance):
        """Calcula las métricas estadísticas generales."""
        profits = [trade["profit"] for trade in closed_trades if trade["profit"] > 0]
        losses = [trade["profit"] for trade in closed_trades if trade["profit"] < 0]

        total_profit = sum(profits)
        total_loss = sum(losses)
        net_profit = total_profit + total_loss

        win_trades = len(profits)
        lose_trades = len(losses)
        total_trades = win_trades + lose_trades

        winrate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = (
            abs(total_profit / total_loss) if total_loss != 0 else float("inf")
        )

        best_trade = max(profits) if profits else 0
        worst_trade = min(losses) if losses else 0

        avg_profit_per_trade = total_profit / win_trades if win_trades > 0 else 0
        avg_loss_per_trade = total_loss / lose_trades if lose_trades > 0 else 0

        profit_std_dev = np.std([trade["profit"] for trade in closed_trades])
        ratio_gain_loss = (
            avg_profit_per_trade / abs(avg_loss_per_trade)
            if avg_loss_per_trade != 0
            else float("inf")
        )

        lr_correlation_equity = self.calculate_lr_correlation("equity")
        lr_correlation_balance = self.calculate_lr_correlation("balance")
        drawdown_metrics = self.calculate_drawdown_metrics()

        # Contar operaciones de compra (buy) y venta (sell)
        buy_trades = len([trade for trade in closed_trades if trade["type"] == "long"])
        sell_trades = len(
            [trade for trade in closed_trades if trade["type"] == "short"]
        )

        return {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "net_profit": net_profit,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "total_trades": total_trades,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_profit_per_trade": avg_profit_per_trade,
            "avg_loss_per_trade": avg_loss_per_trade,
            "profit_std_dev": profit_std_dev,
            "ratio_gain_loss": ratio_gain_loss,
            "buy_trades": buy_trades,  # Número de operaciones de compra
            "sell_trades": sell_trades,  # Número de operaciones de venta
            "lr_correlation_equity": lr_correlation_equity,
            "lr_correlation_balance": lr_correlation_balance,
            **drawdown_metrics,  # Incluir todas las métricas de drawdown
        }
