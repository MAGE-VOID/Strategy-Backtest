# backtest/stats.py
import numpy as np
from datetime import datetime, timedelta


class Statistics:
    def __init__(self, closed_trades, equity_over_time, initial_balance):
        """
        Parámetros:
          - closed_trades: lista de diccionarios con información de cada trade cerrado.
            Cada trade debe incluir:
              - "profit": beneficio (positivo o negativo)
              - "open_time": cadena o datetime (apertura)
              - "close_time": cadena o datetime (cierre)
              - "entry": precio de entrada
              - "exit": precio de salida
              - "type": "long" o "short"
          - equity_over_time: lista de diccionarios con la evolución de la equidad,
            donde cada registro debe incluir:
              - "date": datetime
              - "equity": valor de la equidad
              - "balance": balance (sin incluir beneficios flotantes)
          - initial_balance: balance inicial
        """
        self.closed_trades = closed_trades
        self.equity_over_time = equity_over_time
        self.initial_balance = initial_balance

    # -------------------------
    # Métodos auxiliares
    # -------------------------
    @staticmethod
    def _parse_time(time_input):
        """Convierte time_input a datetime. Si ya es datetime, lo retorna."""
        if isinstance(time_input, datetime):
            return time_input
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(time_input, fmt)
            except Exception:
                continue
        raise ValueError(f"No se pudo parsear la fecha: {time_input}")

    def _separate_profits_losses(self):
        """Separa los beneficios y pérdidas de los trades cerrados."""
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
        """
        Calcula el máximo drawdown basado en la evolución de la equidad.
        Retorna:
          - max_dd_abs: Drawdown máximo absoluto [$] (la mayor caída desde un pico)
          - max_dd_pct: Drawdown máximo relativo [%] (expresado de forma positiva, para luego mostrarlo negativo)
        """
        equity_array = np.array([record["equity"] for record in self.equity_over_time])
        cum_max = np.maximum.accumulate(equity_array)
        drawdowns = cum_max - equity_array
        max_dd_abs = drawdowns.max()
        max_dd_pct = (drawdowns / cum_max).max() * 100 if cum_max.max() > 0 else 0
        return max_dd_abs, max_dd_pct

    def _calculate_var(self, final_balance):
        """Calcula el Value at Risk (VaR) al 95% usando log-retornos vectorizados."""
        equity_values = np.array([record["equity"] for record in self.equity_over_time])
        if equity_values.size > 1:
            safe_eq = np.clip(equity_values, 1e-6, None)
            log_returns = np.diff(np.log(safe_eq))
            alpha = 5  # Percentil 5 para 95% de confianza
            var_log = np.percentile(log_returns, alpha)
            var_pct = 1 - np.exp(var_log)
            return final_balance * var_pct
        return 0.0

    # -------------------------
    # Cálculo de estadísticas esenciales
    # -------------------------
    def calculate_statistics(self):
        if not self.closed_trades or not self.equity_over_time:
            return {}

        # --- Métricas de operaciones ---
        profits, losses = self._separate_profits_losses()
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
        avg_profit = total_profit / win_trades if win_trades > 0 else 0
        avg_loss = total_loss / lose_trades if lose_trades > 0 else 0
        profit_std_dev = np.std(
            [trade.get("profit", 0) for trade in self.closed_trades]
        )
        ratio_gain_loss = avg_profit / abs(avg_loss) if avg_loss != 0 else float("inf")
        final_balance = self.initial_balance + net_profit

        # --- Métricas basadas en la evolución de la equidad y balance ---
        equity_values = np.array([record["equity"] for record in self.equity_over_time])
        min_eq = equity_values.min()
        max_eq = equity_values.max()

        # Drawdown basado en el balance inicial (balance drawdown)
        balance_dd_abs = (
            self.initial_balance - min_eq if min_eq < self.initial_balance else 0
        )
        balance_dd_pct = (balance_dd_abs / self.initial_balance) * 100

        # Drawdown basado en la evolución de la equidad (equity drawdown)
        max_dd_abs, max_dd_pct = self._calculate_drawdowns()
        # Se expresa el drawdown relativo como negativo (típico en informes)
        equity_dd_pct = -max_dd_pct

        account_broken = min_eq <= 0

        # Factores de recuperación (Recovery Factor)
        recovery_equity = net_profit / max_dd_abs if max_dd_abs > 0 else float("inf")
        recovery_balance = (
            net_profit / balance_dd_abs if balance_dd_abs > 0 else float("inf")
        )

        # --- Métricas de retorno y riesgo ---
        start_time = self.equity_over_time[0]["date"]
        end_time = self.equity_over_time[-1]["date"]
        duration = end_time - start_time

        ret_pct = ((final_balance / self.initial_balance) - 1) * 100
        years = duration.total_seconds() / (365.25 * 24 * 3600)
        cagr = (
            (final_balance / self.initial_balance) ** (1 / years) - 1
            if years > 0
            else 0
        )
        return_ann_pct = cagr * 100

        log_returns = np.diff(np.log(np.clip(equity_values, 1e-6, None)))
        if log_returns.size > 0:
            avg_period = duration.total_seconds() / (len(equity_values) - 1)
            periods_per_year = (365.25 * 24 * 3600) / avg_period
            vol_ann = np.std(log_returns) * np.sqrt(periods_per_year) * 100
        else:
            vol_ann = 0

        sharpe_ratio = return_ann_pct / vol_ann if vol_ann != 0 else float("inf")
        downside = log_returns[log_returns < 0]
        if downside.size > 0:
            downside_vol = np.std(downside) * np.sqrt(periods_per_year) * 100
        else:
            downside_vol = 0
        sortino_ratio = (
            return_ann_pct / downside_vol if downside_vol != 0 else float("inf")
        )
        calmar_ratio = return_ann_pct / max_dd_pct if max_dd_pct != 0 else float("inf")

        # --- Métricas individuales de trade ---
        best_trade_pct = (best_trade / self.initial_balance) * 100
        worst_trade_pct = (worst_trade / self.initial_balance) * 100
        avg_trade_pct = (
            ((net_profit / total_trades) / self.initial_balance) * 100
            if total_trades > 0
            else 0
        )

        expectancy = (
            (
                (win_trades / total_trades) * avg_profit
                - ((total_trades - win_trades) / total_trades) * abs(avg_loss)
            )
            if total_trades > 0
            else 0
        )
        expectancy_pct = (expectancy / self.initial_balance) * 100

        profits_array = np.array(
            [trade.get("profit", 0) for trade in self.closed_trades]
        )
        if total_trades > 0 and np.std(profits_array) != 0:
            sqn = (profits_array.mean() / np.std(profits_array)) * np.sqrt(total_trades)
        else:
            sqn = float("inf")
        kelly = (
            (win_trades / total_trades)
            - ((1 - (win_trades / total_trades)) / (avg_profit / abs(avg_loss)))
            if avg_loss != 0
            else float("inf")
        )

        # --- Trades por tipo ---
        buy_trades = sum(
            1 for trade in self.closed_trades if trade.get("type") == "long"
        )
        sell_trades = sum(
            1 for trade in self.closed_trades if trade.get("type") == "short"
        )

        # --- Valor en riesgo ---
        var_95 = self._calculate_var(final_balance)

        # --- Resumen de métricas ---
        summary = {
            # Fechas y duración
            "Start": start_time,
            "End": end_time,
            "Duration": duration,
            # Evolución de la cuenta
            "Equity Final [$]": final_balance,
            "Equity Peak [$]": max_eq,
            "Return [%]": ret_pct,
            "Return (Ann.) [%]": return_ann_pct,
            # Riesgo y volatilidad
            "Volatility (Ann.) [%]": vol_ann,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            # Drawdowns basados en EQUITY
            "Equity Drawdown Absolute [$]": max_dd_abs,
            "Equity Drawdown Relative [%]": equity_dd_pct,
            "Recovery Factor (Equity)": recovery_equity,
            # Drawdowns basados en BALANCE
            "Balance Drawdown Absolute [$]": balance_dd_abs,
            "Balance Drawdown Relative [%]": balance_dd_pct,
            "Recovery Factor (Balance)": recovery_balance,
            # Operaciones
            "Win Rate [%]": winrate,
            "Buy Trades": buy_trades,
            "Sell Trades": sell_trades,
            "# Total Trades": buy_trades + sell_trades,
            "Best Trade [%]": best_trade_pct,
            "Worst Trade [%]": worst_trade_pct,
            "Avg. Trade [%]": avg_trade_pct,
            "Profit Factor": profit_factor,
            "Expectancy [%]": expectancy_pct,
            "SQN": sqn,
            "Kelly Criterion": kelly,
            "VaR 95% [$]": var_95,
            # Detalles internos
            "initial_balance": self.initial_balance,
            "net_profit": net_profit,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "account_broken": account_broken,
        }
        return summary
