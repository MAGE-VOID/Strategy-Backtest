# backtest/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
from datetime import datetime, timedelta


# ============================== Utilidades básicas ==============================

_YEAR_SECONDS = 365.25 * 24 * 3600.0


def _to_datetime(x: Any) -> Optional[datetime]:
    """
    Convierte tipos comunes (datetime, pandas.Timestamp, numpy.datetime64, str ISO)
    a datetime naive (sin tz). Devuelve None si no se puede parsear con seguridad.
    """
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.replace(tzinfo=None)

    # pandas.Timestamp
    if hasattr(x, "to_pydatetime"):
        try:
            dt = x.to_pydatetime()  # type: ignore[attr-defined]
            if isinstance(dt, datetime):
                return dt.replace(tzinfo=None)
        except Exception:
            pass

    # numpy.datetime64
    if isinstance(x, np.datetime64):
        try:
            us = (x - np.datetime64("1970-01-01T00:00:00")).astype("timedelta64[us]").astype(int)
            return datetime.utcfromtimestamp(us / 1_000_000.0)
        except Exception:
            pass

    # str ISO
    if isinstance(x, str):
        s = x.strip().replace("Z", "")
        if "." in s:
            head, tail = s.split(".", 1)
            tail = "".join(ch for ch in tail if ch.isdigit())[:6]  # a micros
            s = f"{head}.{tail}" if tail else head
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass

    return None


def _ratio(num: float, den: float) -> float:
    """Devuelve num/den, o 0.0 si den es 0/None (evita 'inf')."""
    return float(num / den) if (den not in (0, 0.0, None)) else 0.0


def _np_float_array(seq: List[float]) -> np.ndarray:
    return np.asarray(seq, dtype=float) if seq else np.array([], dtype=float)


# ============================== Estructuras internas =============================

@dataclass
class Trade:
    """
    Estructura mínima: sólo lo que se usa para estadísticas agregadas.
    """
    ticket: int
    side: str                 # "long" | "short"
    profit: float
    open_time: datetime
    close_time: datetime

    @property
    def duration(self) -> timedelta:
        return self.close_time - self.open_time


# ================================== Núcleo =====================================

class Statistics:
    """
    Calcula métricas del backtest a partir de:
      - Eventos de posiciones (para métricas por trade).
      - Serie de equity/balance (para DD y ratios de riesgo).
    Devuelve UN solo resultado agregado (sin desgloses por símbolo/magic).
    """

    def __init__(
        self,
        all_trades: List[Dict[str, Any]],
        equity_over_time: List[Dict[str, Any]],
        initial_balance: float,
    ):
        self.initial_balance = float(initial_balance or 0.0)
        # Guardamos los eventos crudos para métricas adicionales (comisiones)
        self.events: List[Dict[str, Any]] = list(all_trades or [])
        self.trades: List[Trade] = self._reconstruct_closed_trades(all_trades or [])
        self.eq_arr, self.bal_arr, self.dt_arr = self._extract_time_series(equity_over_time or [])

    # --------------------------- Reconstrucción de trades ---------------------------

    def _reconstruct_closed_trades(self, events: List[Dict[str, Any]]) -> List[Trade]:
        """
        Une 'open' y 'closed' por ticket. Ignora entradas inconsistentes o incompletas.
        """
        by_ticket: Dict[int, Dict[str, Any]] = {}
        for ev in events:
            t = ev.get("ticket")
            if t is None:
                continue
            rec = by_ticket.get(t, {})
            status = str(ev.get("status", "")).lower()

            if status == "open":
                rec.update(
                    ticket=t,
                    side=str(ev.get("type", "")).lower(),
                    open_time=_to_datetime(ev.get("open_time")),
                )
            elif status == "closed":
                rec.setdefault("side", str(ev.get("type", "")).lower())
                rec.update(
                    close_time=_to_datetime(ev.get("close_time")),
                    profit=float(ev.get("profit", 0.0) or 0.0),
                )
            by_ticket[t] = rec

        out: List[Trade] = []
        for rec in by_ticket.values():
            ot, ct = rec.get("open_time"), rec.get("close_time")
            if ot is None or ct is None:
                continue
            side = str(rec.get("side", "")).lower()
            if side not in ("long", "short"):
                continue
            try:
                out.append(
                    Trade(
                        ticket=int(rec.get("ticket")),
                        side=side,
                        profit=float(rec.get("profit", 0.0) or 0.0),
                        open_time=ot,
                        close_time=ct,
                    )
                )
            except Exception:
                # Si algo vino corrupto, lo omitimos
                continue
        out.sort(key=lambda tr: tr.close_time)
        return out

    # ------------------------------ Serie temporal --------------------------------

    def _extract_time_series(self, rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eq, bal, dt = [], [], []
        for r in rows:
            e, b, t = r.get("equity"), r.get("balance"), r.get("date")
            if e is None or b is None or t is None:
                continue
            try:
                ee, bb = float(e), float(b)
            except Exception:
                continue
            ts = _to_datetime(t)
            if ts is None:
                continue
            eq.append(ee), bal.append(bb), dt.append(ts)

        if not dt:
            return np.array([]), np.array([]), np.array([])

        dt_np = np.asarray(dt, dtype="datetime64[ns]")
        order = np.argsort(dt_np)
        eq_np = np.asarray(eq, dtype=float)[order]
        bal_np = np.asarray(bal, dtype=float)[order]
        dt_np = dt_np[order]
        mask = ~(np.isnan(eq_np) | np.isnan(bal_np))
        return eq_np[mask], bal_np[mask], dt_np[mask]

    # ------------------------------ Drawdown utils --------------------------------

    @staticmethod
    def _dd_from_series(vals: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Devuelve (max_dd_abs, max_dd_pct_pos, dd_series_pct_neg)."""
        if vals.size == 0:
            return 0.0, 0.0, np.array([])
        m = np.maximum.accumulate(vals)
        dd_abs = m - vals
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = np.where(m > 0.0, dd_abs / m * 100.0, 0.0)
        return float(dd_abs.max()), float(dd_pct.max()), -dd_pct

    # --------------------------- Exposición / concurrencia ------------------------

    def _exposure_and_concurrency(self) -> Tuple[timedelta, float, int, float]:
        """
        Unión de intervalos para exposición (% del tiempo con al menos 1 trade),
        concurrencia máxima y concurrencia promedio ponderada en el tiempo.
        """
        if not self.trades or self.dt_arr.size == 0:
            return timedelta(0), 0.0, 0, 0.0

        # Intervalos (ordenados)
        intervals = [(tr.open_time, tr.close_time) for tr in self.trades]
        intervals.sort(key=lambda x: x[0])

        # Unión
        merged: List[Tuple[datetime, datetime]] = []
        cs, ce = intervals[0]
        for s, e in intervals[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))

        total_exposed = sum((e - s for s, e in merged), timedelta(0))

        # Duración total (segundos)
        try:
            period_seconds = float((self.dt_arr[-1] - self.dt_arr[0]) / np.timedelta64(1, "s"))
        except Exception:
            start = _to_datetime(str(self.dt_arr[0]))
            end = _to_datetime(str(self.dt_arr[-1]))
            period_seconds = (end - start).total_seconds() if (start and end) else 0.0

        exposure_pct = (total_exposed.total_seconds() / period_seconds * 100.0) if period_seconds > 0 else 0.0

        # Concurrencia (line sweep). Cierre antes que apertura si coinciden.
        events: List[Tuple[datetime, int, int]] = []
        for s, e in intervals:
            events.append((s, 1, +1))
            events.append((e, 0, -1))
        events.sort(key=lambda x: (x[0], x[1]))

        peak, cur, area = 0, 0, 0.0
        last_t = events[0][0]
        for t, _, dlt in events:
            dt_sec = (t - last_t).total_seconds()
            if dt_sec > 0:
                area += cur * dt_sec
            cur += dlt
            peak = max(peak, cur)
            last_t = t

        avg_conc = (area / period_seconds) if period_seconds > 0 else 0.0
        return total_exposed, exposure_pct, peak, avg_conc

    # ------------------------------ Retornos y ratios -----------------------------

    def _returns_metrics(self, eq: np.ndarray, dt: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Retornos simples 1-periodo, volatilidad anualizada (%), Sharpe y Sortino (rf=0).
        Devuelve: (ret_1p, vol_ann_pct, sharpe, sortino)
        """
        if eq.size <= 1:
            return np.array([]), 0.0, 0.0, 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(eq[:-1] > 0.0, (eq[1:] / eq[:-1]) - 1.0, 0.0)

        dt_seconds = np.diff(dt.astype("datetime64[s]").astype(np.int64)).astype(float)
        avg_period = float(np.mean(dt_seconds)) if dt_seconds.size else 0.0
        periods_per_year = (_YEAR_SECONDS / avg_period) if avg_period > 0 else 0.0

        mu = float(np.mean(r)) if r.size else 0.0
        sigma = float(np.std(r, ddof=0)) if r.size else 0.0
        downside = r[r < 0.0]
        sigma_down = float(np.std(downside, ddof=0)) if downside.size else 0.0

        vol_ann_pct = (sigma * math.sqrt(periods_per_year) * 100.0) if periods_per_year > 0 else 0.0
        sharpe = (mu / sigma * math.sqrt(periods_per_year)) if (sigma > 0 and periods_per_year > 0) else 0.0
        sortino = (mu / sigma_down * math.sqrt(periods_per_year)) if (sigma_down > 0 and periods_per_year > 0) else 0.0

        return r, vol_ann_pct, sharpe, sortino

    @staticmethod
    def _var_95(final_equity: float, ret_series: np.ndarray) -> float:
        """VaR 95% 1-periodo (magnitude en $) a partir del percentil 5% de retornos."""
        if ret_series.size == 0:
            return 0.0
        q = float(np.percentile(ret_series, 5))
        return abs(q) * float(final_equity)

    @staticmethod
    def _ulcer_index(dd_pct_negative: np.ndarray) -> float:
        """Ulcer Index (usar dd% negativos)."""
        if dd_pct_negative.size == 0:
            return 0.0
        return float(math.sqrt(np.mean((dd_pct_negative) ** 2)))

    # --------------------------------- Cálculo principal -------------------------

    def calculate_statistics(self) -> Dict[str, Any]:
        # --- Serie de equity/balance
        eq, bal, dt = self.eq_arr, self.bal_arr, self.dt_arr
        if dt.size == 0:
            return {}

        start_time = _to_datetime(str(dt[0]))
        end_time = _to_datetime(str(dt[-1]))
        duration = (end_time - start_time) if (start_time and end_time) else timedelta(0)

        final_equity = float(eq[-1]) if eq.size else self.initial_balance
        final_balance = float(bal[-1]) if bal.size else self.initial_balance
        equity_peak = float(np.nanmax(eq)) if eq.size else self.initial_balance
        balance_peak = float(np.nanmax(bal)) if bal.size else self.initial_balance

        # Drawdowns
        eq_dd_abs, eq_dd_pct_pos, eq_dd_series = self._dd_from_series(eq)
        bal_dd_abs, bal_dd_pct_pos, _ = self._dd_from_series(bal)

        # Retornos y ratios
        ret_1p, vol_ann_pct, sharpe_ratio, sortino_ratio = self._returns_metrics(eq, dt)

        # CAGR / Calmar
        eq0 = float(eq[0]) if eq.size else self.initial_balance
        years = (duration.total_seconds() / _YEAR_SECONDS) if duration.total_seconds() > 0 else 0.0
        cagr = ((final_equity / eq0) ** (1 / years) - 1) if (years > 0 and eq0 > 0) else 0.0
        return_ann_pct = cagr * 100.0
        calmar_ratio = _ratio(cagr, (eq_dd_pct_pos / 100.0))  # CAGR / MaxDD

        # P/L cerrado por trades
        profits = _np_float_array([tr.profit for tr in self.trades])
        total_profit = float(profits[profits > 0].sum()) if profits.size else 0.0
        total_loss = float(profits[profits < 0].sum()) if profits.size else 0.0
        net_profit_closed = total_profit + total_loss
        net_profit_balance = final_balance - self.initial_balance

        # Comisiones (del log de eventos)
        comm_open_total = 0.0
        comm_close_total = 0.0
        try:
            for ev in self.events:
                st = str(ev.get("status", "")).lower()
                if st == "open":
                    comm_open_total += float(ev.get("commission_open", 0.0) or 0.0)
                elif st == "closed":
                    comm_close_total += float(ev.get("commission_close", 0.0) or 0.0)
        except Exception:
            # Si el log no contiene campos de comisión, mantenemos 0.0
            pass
        comm_total = float(comm_open_total + comm_close_total)
        # P/L antes de todas las comisiones (usando balance final)
        net_profit_before_commissions = float(net_profit_balance + comm_total)
        # P/L cerrado antes de comisión de cierre (profit de trades ya descuenta cierre)
        closed_pl_before_close_commission = float(net_profit_closed + comm_close_total)

        # Métricas por trade
        total_trades = int(len(self.trades))
        wins_mask = profits > 0
        losses_mask = profits < 0
        win_trades = int(wins_mask.sum()) if profits.size else 0
        lose_trades = int(losses_mask.sum()) if profits.size else 0
        winrate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_win = float(profits[wins_mask].mean()) if win_trades > 0 else 0.0
        avg_loss = float(profits[losses_mask].mean()) if lose_trades > 0 else 0.0
        payoff = _ratio(avg_win, abs(avg_loss)) if avg_loss != 0 else 0.0
        profit_factor = _ratio(total_profit, abs(total_loss)) if total_loss != 0 else 0.0
        best_trade = float(profits.max()) if profits.size else 0.0
        worst_trade = float(profits.min()) if profits.size else 0.0
        avg_trade = float(profits.mean()) if profits.size else 0.0

        # Percentiles por trade ($)
        if profits.size:
            p5, p50, p95 = (float(np.percentile(profits, q)) for q in (5, 50, 95))
        else:
            p5 = p50 = p95 = 0.0

        # Duraciones
        if self.trades:
            dsecs = _np_float_array([t.duration.total_seconds() for t in self.trades])
            avg_hold = timedelta(seconds=float(dsecs.mean()))
            med_hold = timedelta(seconds=float(np.median(dsecs)))
            max_hold = timedelta(seconds=float(dsecs.max()))
        else:
            avg_hold = med_hold = max_hold = timedelta(0)

        # Exposición / concurrencia
        exposure_td, exposure_pct, peak_conc, avg_conc = self._exposure_and_concurrency()

        # Rachas
        max_wins = max_losses = cw = cl = 0
        for tr in self.trades:
            if tr.profit > 0:
                cw += 1; cl = 0
            elif tr.profit < 0:
                cl += 1; cw = 0
            max_wins = max(max_wins, cw)
            max_losses = max(max_losses, cl)

        # Lados
        buy_trades = sum(1 for t in self.trades if t.side == "long")
        sell_trades = total_trades - buy_trades
        long_wins = sum(1 for t in self.trades if t.side == "long" and t.profit > 0)
        short_wins = sum(1 for t in self.trades if t.side == "short" and t.profit > 0)
        long_wr = (long_wins / buy_trades * 100.0) if buy_trades > 0 else 0.0
        short_wr = (short_wins / sell_trades * 100.0) if sell_trades > 0 else 0.0

        # VaR y Ulcer
        var_95 = self._var_95(final_equity, ret_1p)
        ulcer_idx = self._ulcer_index(eq_dd_series)

        # Sanidad
        open_pl_end = final_equity - final_balance
        account_broken = bool(eq.size > 0 and np.nanmin(eq) <= 0.0)

        # SQN (Van Tharp). Usamos ddof=1 si hay >=2 trades
        if profits.size >= 2:
            std_pl = float(np.std(profits, ddof=1))
            out_sqn = float((profits.mean() / std_pl) * math.sqrt(len(profits))) if std_pl > 0 else 0.0
        else:
            out_sqn = 0.0

        # Kelly (aprox). Si no es definible, devolvemos 0.0
        if total_trades > 0 and avg_loss != 0:
            p = winrate / 100.0
            R = payoff  # avg_win / |avg_loss|
            out_kelly = float(p - (1 - p) / R) if R > 0 else 0.0
        else:
            out_kelly = 0.0

        # ---------------------------- Salida plana ----------------------------
        out: Dict[str, Any] = {
            "Start": start_time,
            "End": end_time,
            "Duration": duration,
            # Finales / Picos
            "Equity Final [$]": final_equity,
            "Balance Final [$]": final_balance,
            "Equity Peak [$]": equity_peak,
            "Balance Peak [$]": balance_peak,
            # Retornos y riesgos (equity MTM)
            "Return [%]": ((final_balance / self.initial_balance) - 1.0) * 100.0 if self.initial_balance > 0 else 0.0,
            "Return (Ann.) [%]": return_ann_pct,
            "Volatility (Ann.) [%]": vol_ann_pct,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            # Drawdowns
            "Equity Drawdown Absolute [$]": eq_dd_abs,
            "Equity Drawdown Relative [%]": -eq_dd_pct_pos,  # negativo por convención
            "Balance Drawdown Absolute [$]": bal_dd_abs,
            "Balance Drawdown Relative [%]": -bal_dd_pct_pos,
            "Ulcer Index [%]": ulcer_idx,
            # P/L cerrado y sanity
            "Net Profit [$]": net_profit_balance,          # balance final - inicial
            "Net Profit (Closed) [$]": net_profit_closed,  # suma de profits por trade
            # Comisiones
            "Commissions Open [$]": comm_open_total,
            "Commissions Close [$]": comm_close_total,
            "Commissions Total [$]": comm_total,
            # P/L bruto (útil para comparar sin comisiones)
            "Net Profit (Before Commissions) [$]": net_profit_before_commissions,
            "Net Profit (Closed, Before Close Comm.) [$]": closed_pl_before_close_commission,
            "Open P/L at End [$]": open_pl_end,
            # Trades
            "# Total Trades": total_trades,
            "Win Rate [%]": winrate,
            "Profit Factor": profit_factor,
            "Payoff Ratio": payoff,
            "Best Trade [$]": best_trade,
            "Worst Trade [$]": worst_trade,
            "Avg. Trade [$]": avg_trade,
            "P5 Trade [$]": p5,
            "Median Trade [$]": p50,
            "P95 Trade [$]": p95,
            "Avg. Win [$]": avg_win,
            "Avg. Loss [$]": avg_loss,
            "Buy Trades": buy_trades,
            "Sell Trades": sell_trades,
            "Win Rate Long [%]": long_wr,
            "Win Rate Short [%]": short_wr,
            "Max Consecutive Wins": max_wins,
            "Max Consecutive Losses": max_losses,
            # Duraciones / exposición
            "Avg Holding Period": avg_hold,
            "Median Holding Period": med_hold,
            "Max Holding Period": max_hold,
            "Exposure Time": exposure_td,
            "Exposure [%]": exposure_pct,
            "Peak Concurrency": peak_conc,
            "Avg Concurrency": avg_conc,
            # Riesgo
            "VaR 95% [$]": var_95,
            # Extras útiles
            "initial_balance": self.initial_balance,
            "account_broken": account_broken,
            # Calidad de sistema
            "SQN": out_sqn,
            "Kelly Criterion": out_kelly,
        }

        # Métricas relativas al balance inicial
        if self.initial_balance > 0:
            out.update({
                "Best Trade [%]": (best_trade / self.initial_balance) * 100.0,
                "Worst Trade [%]": (worst_trade / self.initial_balance) * 100.0,
                "Avg. Trade [%]": (avg_trade / self.initial_balance) * 100.0,
                "Expectancy [%]": (avg_trade / self.initial_balance) * 100.0,
            })
        else:
            out.update({
                "Best Trade [%]": 0.0, "Worst Trade [%]": 0.0,
                "Avg. Trade [%]": 0.0, "Expectancy [%]": 0.0,
            })

        return out
