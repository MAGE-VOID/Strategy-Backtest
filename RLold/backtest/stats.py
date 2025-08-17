# backtest/stats.py
import numpy as np
from datetime import datetime, timedelta


class Statistics:
    def __init__(self, all_trades, equity_over_time, initial_balance):
        """
        all_trades: lista de eventos (apertura/cierre)
        equity_over_time: [{"date","equity","balance","open_trades","open_lots"}, ...]
        initial_balance: balance inicial
        """
        self.closed_trades = [t for t in all_trades if t.get("status") == "closed"]
        self.equity_over_time = equity_over_time or []
        self.initial_balance = initial_balance

    @staticmethod
    def _dd_series(values: np.ndarray):
        """
        Devuelve (max_dd_abs, max_dd_pct, series_dd_pct_negativa)
        dd_pct negativa: 0 en máximo, valores negativos en drawdown
        """
        if values.size == 0:
            return 0.0, 0.0, np.array([])
        cum_max = np.maximum.accumulate(values)
        dd_abs = cum_max - values
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = np.where(cum_max > 0, dd_abs / cum_max * 100.0, 0.0)
        max_dd_abs = float(dd_abs.max())
        max_dd_pct = float(dd_pct.max())
        dd_pct_negative = -dd_pct  # negativa (por convención)
        return max_dd_abs, max_dd_pct, dd_pct_negative

    def _calculate_var(self, final_equity):
        eq = np.array([rec["equity"] for rec in self.equity_over_time], dtype=float)
        if eq.size > 1:
            safe = np.clip(eq, 1e-6, None)
            log_returns = np.diff(np.log(safe))
            alpha = 5
            var_log = np.percentile(log_returns, alpha)
            var_pct = 1 - np.exp(var_log)
            return float(final_equity * var_pct)
        return 0.0

    def calculate_statistics(self):
        if not self.equity_over_time:
            return {}

        # Series de tiempo
        eq_vals = np.array(
            [rec["equity"] for rec in self.equity_over_time], dtype=float
        )
        bal_vals = np.array(
            [rec["balance"] for rec in self.equity_over_time], dtype=float
        )
        dates = [rec["date"] for rec in self.equity_over_time]

        # Finales y picos
        final_equity = float(eq_vals[-1])
        final_balance = float(bal_vals[-1])
        equity_peak = (
            float(np.nanmax(eq_vals)) if eq_vals.size else self.initial_balance
        )
        balance_peak = (
            float(np.nanmax(bal_vals)) if bal_vals.size else self.initial_balance
        )

        # Drawdowns
        eq_dd_abs, eq_dd_pct, _ = self._dd_series(eq_vals)
        bal_dd_abs, bal_dd_pct, _ = self._dd_series(bal_vals)

        # Beneficios realiz. por trades cerrados
        profits = [t["profit"] for t in self.closed_trades if t["profit"] > 0]
        losses = [t["profit"] for t in self.closed_trades if t["profit"] < 0]
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_profit = (
            total_profit + total_loss
        )  # debería ser ~ final_balance - initial_balance

        # Trades
        win_trades = len(profits)
        lose_trades = len(losses)
        total_trades = win_trades + lose_trades
        winrate = (win_trades / total_trades) * 100 if total_trades > 0 else 0.0
        profit_factor = (
            abs(total_profit / total_loss) if total_loss != 0 else float("inf")
        )
        best_trade = max(profits) if profits else 0.0
        worst_trade = min(losses) if losses else 0.0
        avg_profit = (total_profit / win_trades) if win_trades > 0 else 0.0
        avg_loss = (total_loss / lose_trades) if lose_trades > 0 else 0.0

        # Periodo
        start_time = dates[0]
        end_time = dates[-1]
        duration = end_time - start_time

        # Retornos
        ret_pct = ((final_balance / self.initial_balance) - 1.0) * 100.0
        growth = (
            (final_equity / self.initial_balance) if self.initial_balance > 0 else 0.0
        )
        years = (
            duration.total_seconds() / (365.25 * 24 * 3600)
            if duration.total_seconds() > 0
            else 0.0
        )
        if years > 0 and growth > 0:
            cagr = growth ** (1 / years) - 1
        else:
            cagr = 0.0
        return_ann_pct = cagr * 100.0

        # Volatilidad/Sharpe/Sortino con equity MTM
        if eq_vals.size > 1:
            safe_eq = np.clip(eq_vals, 1e-6, None)
            log_returns = np.diff(np.log(safe_eq))
            avg_period = duration.total_seconds() / (len(eq_vals) - 1)
            periods_per_year = (
                (365.25 * 24 * 3600) / avg_period if avg_period > 0 else 0.0
            )
            vol_ann = (
                (np.std(log_returns) * np.sqrt(periods_per_year) * 100.0)
                if periods_per_year > 0
                else 0.0
            )
            downside = log_returns[log_returns < 0]
            downside_vol = (
                (np.std(downside) * np.sqrt(periods_per_year) * 100.0)
                if (downside.size > 0 and periods_per_year > 0)
                else 0.0
            )
        else:
            vol_ann = 0.0
            downside_vol = 0.0
            log_returns = np.array([])

        sharpe_ratio = (return_ann_pct / vol_ann) if vol_ann != 0 else float("inf")
        sortino_ratio = (
            (return_ann_pct / downside_vol) if downside_vol != 0 else float("inf")
        )
        calmar_ratio = (return_ann_pct / eq_dd_pct) if eq_dd_pct != 0 else float("inf")

        # Métricas por trade
        best_trade_pct = (
            (best_trade / self.initial_balance) * 100.0
            if self.initial_balance > 0
            else 0.0
        )
        worst_trade_pct = (
            (worst_trade / self.initial_balance) * 100.0
            if self.initial_balance > 0
            else 0.0
        )
        avg_trade_pct = (
            (((net_profit / total_trades) / self.initial_balance) * 100.0)
            if (total_trades > 0 and self.initial_balance > 0)
            else 0.0
        )

        profits_array = np.array([t["profit"] for t in self.closed_trades], dtype=float)
        if total_trades > 0 and np.std(profits_array) != 0:
            sqn = (profits_array.mean() / np.std(profits_array)) * np.sqrt(total_trades)
        else:
            sqn = float("inf")

        kelly = (
            (
                (win_trades / total_trades)
                - ((1 - (win_trades / total_trades)) / (avg_profit / abs(avg_loss)))
            )
            if (avg_loss != 0 and total_trades > 0)
            else float("inf")
        )

        buy_trades = sum(1 for t in self.closed_trades if t["type"] == "long")
        sell_trades = sum(1 for t in self.closed_trades if t["type"] == "short")

        var_95 = self._calculate_var(final_equity)

        account_broken = (eq_vals.min() <= 0) if eq_vals.size else False

        return {
            "Start": start_time,
            "End": end_time,
            "Duration": duration,
            # Finales / Picos
            "Equity Final [$]": final_equity,
            "Balance Final [$]": final_balance,
            "Equity Peak [$]": equity_peak,
            "Balance Peak [$]": balance_peak,
            # Returns & riesgos
            "Return [%]": ret_pct,  # vs balance inicial
            "Return (Ann.) [%]": return_ann_pct,  # basado en equity MTM
            "Volatility (Ann.) [%]": vol_ann,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            # Drawdowns
            "Equity Drawdown Absolute [$]": eq_dd_abs,
            "Equity Drawdown Relative [%]": -eq_dd_pct,  # negativo por convención
            "Balance Drawdown Absolute [$]": bal_dd_abs,
            "Balance Drawdown Relative [%]": -bal_dd_pct,  # negativo por convención
            "Win Rate [%]": winrate,
            "Buy Trades": buy_trades,
            "Sell Trades": sell_trades,
            "# Total Trades": total_trades,
            "Best Trade [%]": best_trade_pct,
            "Worst Trade [%]": worst_trade_pct,
            "Avg. Trade [%]": avg_trade_pct,
            "Profit Factor": profit_factor,
            "Expectancy [%]": (
                (
                    (win_trades / total_trades) * avg_profit
                    - ((total_trades - win_trades) / total_trades) * abs(avg_loss)
                )
                if total_trades > 0
                else 0.0
            ),
            "SQN": sqn,
            "Kelly Criterion": kelly,
            "VaR 95% [$]": var_95,
            # Extras útiles para chequeo
            "initial_balance": self.initial_balance,
            "net_profit_closed": net_profit,  # P/L cerrado acumulado
            "account_broken": account_broken,
        }
