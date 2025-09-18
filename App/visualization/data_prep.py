from __future__ import annotations

"""
data_prep.py

Preparación de datos para el dashboard de backtest: parsing, ordenado, limpieza,
downsample y derivadas (floating, drawdown, descomposición long/short).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pandas como último recurso para parsear fechas atípicas
    import pandas as pd  # type: ignore  # noqa: F401

    HAS_PANDAS = True
except Exception:  # pragma: no cover - entorno sin pandas
    HAS_PANDAS = False

from .gpu_utils import (
    HAS_CUPY,
    GPU_MIN_POINTS,
    GPU_USE_FP32,
    all_nan,
    choose_idx,
    to_numpy,
    to_xp,
)


@dataclass(frozen=True)
class PreparedData:
    dates: np.ndarray
    equity: np.ndarray
    balance: np.ndarray
    floating: np.ndarray
    floating_init_pct: np.ndarray
    initial_balance: float
    deposit_load_pct: np.ndarray
    comm_open: np.ndarray
    comm_close: np.ndarray
    comm_total: np.ndarray
    comm_total_cum: np.ndarray
    open_trades_total: np.ndarray
    open_lots_total: np.ndarray
    ot_long: Optional[np.ndarray]
    ot_short: Optional[np.ndarray]
    ol_long: Optional[np.ndarray]
    ol_short: Optional[np.ndarray]


def _parse_dt64_ns(x) -> np.datetime64:
    try:
        if isinstance(x, np.datetime64):
            return x.astype("datetime64[ns]")
        if isinstance(x, datetime):
            return np.datetime64(x, "ns")
        return np.datetime64(x, "ns")
    except Exception:
        if HAS_PANDAS:
            # type: ignore
            return np.datetime64(pd.to_datetime(x), "ns")  # pragma: no cover
        raise


def _prepare_arrays(equity_over_time: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray | None]]:
    dts, eq, bal, flo, ot, ol = [], [], [], [], [], []
    for r in equity_over_time:
        dt = r.get("date", None)
        e = r.get("equity", None)
        b = r.get("balance", None)
        if dt is None or e is None or b is None:
            continue
        try:
            e = float(e)
            b = float(b)
        except Exception:
            continue
        dts.append(_parse_dt64_ns(dt))
        eq.append(e)
        bal.append(b)
        flo.append(r.get("floating", None))
        ot.append(r.get("open_trades", None))
        ol.append(r.get("open_lots", None))

    if not dts:
        return np.array([], dtype="datetime64[ns]"), {}

    dates = np.array(dts, dtype="datetime64[ns]")
    order = np.argsort(dates)
    dates = dates[order]

    eq_np = np.asarray(eq, dtype=np.float64)[order]
    bal_np = np.asarray(bal, dtype=np.float64)[order]

    def _opt(arr_list, dtype):
        if all(v is None for v in arr_list):
            return None
        vals = [np.nan if v is None else float(v) for v in arr_list]
        return np.asarray(vals, dtype=dtype)[order]

    flo_np = _opt(flo, np.float64)
    ot_np = _opt(ot, np.float64)
    ol_np = _opt(ol, np.float64)

    # limpia filas donde equity/balance sean NaN
    mask = ~(np.isnan(eq_np) | np.isnan(bal_np))
    dates = dates[mask]
    eq_np = eq_np[mask]
    bal_np = bal_np[mask]
    if flo_np is not None:
        flo_np = flo_np[mask]
    if ot_np is not None:
        ot_np = ot_np[mask]
    if ol_np is not None:
        ol_np = ol_np[mask]

    cols = {
        "equity": eq_np,
        "balance": bal_np,
        "floating": flo_np,
        "open_trades": ot_np,
        "open_lots": ol_np,
    }
    return dates, cols


def _side_series(
    dates_np: np.ndarray,
    idx_subset_np: np.ndarray,
    trades: List[Dict],
    use_gpu: bool,
) -> Dict[str, np.ndarray | None]:
    """Reconstruye series acumuladas long/short en los índices pedidos (subset)."""
    xp = np
    r_gpu = None
    mask_gpu = None
    if use_gpu and HAS_CUPY:
        assert to_xp is not None  # para mypy
        # elegir xp=cp en llamadas to_xp
        try:
            import cupy as cp  # type: ignore
        except Exception:  # pragma: no cover
            pass
        xp = cp  # type: ignore

    out = {
        "open_trades_long": None,
        "open_trades_short": None,
        "open_lots_long": None,
        "open_lots_short": None,
    }
    if not trades or dates_np.size == 0 or idx_subset_np.size == 0:
        return out

    # Compactar por ticket (CPU)
    tmap: Dict = {}
    for r in trades:
        k = r.get("ticket")
        if k is None:
            continue
        typ = str(r.get("type", "")).lower()
        lot = float(r.get("lot_size", r.get("lot", 0.0)) or 0.0)

        if r.get("status") == "open":
            tmap[k] = {
                "type": typ,
                "lot": lot,
                "open": _parse_dt64_ns(r.get("open_time")),
                "close": None,
            }
        elif r.get("status") == "closed":
            x = tmap.get(
                k,
                {
                    "type": typ,
                    "lot": lot,
                    "open": (
                        _parse_dt64_ns(r.get("open_time", None))
                        if r.get("open_time", None) is not None
                        else None
                    ),
                    "close": None,
                },
            )
            x["close"] = _parse_dt64_ns(r.get("close_time"))
            tmap[k] = x

    if not tmap:
        return out

    # Fechas a enteros ns para searchsorted eficiente
    d_int_np = dates_np.astype("datetime64[ns]").astype("int64")
    n = d_int_np.size

    ev_idx, dcl, dll, dcs, dls = [], [], [], [], []
    for info in tmap.values():
        if info.get("open") is None:
            continue
        i0 = int(
            np.searchsorted(
                d_int_np, np.datetime64(info["open"], "ns").astype("int64"), "left"
            )
        )
        if i0 >= n:
            continue
        i1 = (
            int(
                np.searchsorted(
                    d_int_np,
                    np.datetime64(info["close"], "ns").astype("int64"),
                    "left",
                )
            )
            if info.get("close") is not None
            else n
        )
        lot = float(info.get("lot", 0.0) or 0.0)
        if info.get("type") == "long":
            ev_idx += [i0, i1]
            dcl += [1, -1]
            dll += [lot, -lot]
            dcs += [0, 0]
            dls += [0, 0]
        elif info.get("type") == "short":
            ev_idx += [i0, i1]
            dcl += [0, 0]
            dll += [0, 0]
            dcs += [1, -1]
            dls += [lot, -lot]

    if not ev_idx:
        return out

    # Ordena eventos y acumula
    ev_idx_np = np.asarray(ev_idx, dtype=np.int64)
    order = np.argsort(ev_idx_np)
    ev_idx_np = ev_idx_np[order]

    dcl_ps = np.asarray(dcl, dtype=np.float32 if GPU_USE_FP32 else np.float64)[order]
    dll_ps = np.asarray(dll, dtype=np.float32 if GPU_USE_FP32 else np.float64)[order]
    dcs_ps = np.asarray(dcs, dtype=np.float32 if GPU_USE_FP32 else np.float64)[order]
    dls_ps = np.asarray(dls, dtype=np.float32 if GPU_USE_FP32 else np.float64)[order]

    if use_gpu and HAS_CUPY:
        import cupy as cp  # type: ignore

        dcl_xp = cp.cumsum(to_xp(dcl_ps, True))
        dll_xp = cp.cumsum(to_xp(dll_ps, True))
        dcs_xp = cp.cumsum(to_xp(dcs_ps, True))
        dls_xp = cp.cumsum(to_xp(dls_ps, True))

        r_np = np.searchsorted(ev_idx_np, idx_subset_np, side="right") - 1
        mask_np = r_np >= 0
        r_clip_np = np.clip(r_np, 0, None)
        r_gpu = cp.asarray(
            r_clip_np, dtype=cp.int32 if (len(ev_idx_np) < 2**31) else cp.int64
        )
        mask_gpu = cp.asarray(mask_np)

        def pick(ps_xp):
            base = ps_xp[r_gpu]
            return cp.where(mask_gpu, base, 0.0)

        return {
            "open_trades_long": to_numpy(pick(dcl_xp)),
            "open_trades_short": to_numpy(pick(dcs_xp)),
            "open_lots_long": to_numpy(xp.around(pick(dll_xp), 2)),
            "open_lots_short": to_numpy(xp.around(pick(dls_xp), 2)),
        }
    else:
        # CPU
        dcl_ps = np.cumsum(dcl_ps)
        dll_ps = np.cumsum(dll_ps)
        dcs_ps = np.cumsum(dcs_ps)
        dls_ps = np.cumsum(dls_ps)

        r_np = np.searchsorted(ev_idx_np, idx_subset_np, side="right") - 1
        mask_np = r_np >= 0
        r_clip_np = np.clip(r_np, 0, None)

        def pick(ps_np):
            base = ps_np[r_clip_np]
            return np.where(mask_np, base, 0.0)

        return {
            "open_trades_long": pick(dcl_ps),
            "open_trades_short": pick(dcs_ps),
            "open_lots_long": np.around(pick(dll_ps), 2),
            "open_lots_short": np.around(pick(dls_ps), 2),
        }


class DataPreparer:
    """Pipeline de preparación de datos para el dashboard.

    GPU-first: sube a GPU si hay suficientes puntos y se solicita, y solo baja a
    NumPy justo antes del plot.
    """

    def __init__(self, *, use_gpu: bool = True, max_points: Optional[int] = None, downsample: bool = True) -> None:
        self.use_gpu_requested = bool(use_gpu and HAS_CUPY)
        self.downsample = bool(downsample)
        self.max_points = None if max_points is None else int(max_points)

    def prepare(self, result_backtest: Dict) -> PreparedData:
        if "equity_over_time" not in result_backtest:
            raise ValueError("result_backtest debe incluir la clave 'equity_over_time'.")

        dates_np, cols_np = _prepare_arrays(result_backtest["equity_over_time"])  # host
        n = dates_np.size
        if n == 0:
            raise ValueError("No hay datos en 'equity_over_time' para graficar.")

        use_gpu_now = bool(self.use_gpu_requested and (n >= GPU_MIN_POINTS))
        xp = None
        if use_gpu_now:
            import cupy as cp  # type: ignore

            xp = cp  # type: ignore
            gpu_dtype = cp.float32 if GPU_USE_FP32 else cp.float64
        else:
            xp = np
            gpu_dtype = None

        equity = to_xp(cols_np["equity"], use_gpu_now, dtype=gpu_dtype)
        balance = to_xp(cols_np["balance"], use_gpu_now, dtype=gpu_dtype)

        # Recalcular floating para asegurar consistencia y redondear a 2 decimales
        # floating = equity - balance
        try:
            # xp puede ser np o cp; ambas tienen around
            floating = xp.around(equity - balance, 2)  # type: ignore[attr-defined]
        except Exception:
            floating = np.around(to_numpy(equity) - to_numpy(balance), 2)

        open_trades = (
            to_xp(cols_np["open_trades"], use_gpu_now, dtype=gpu_dtype)
            if cols_np.get("open_trades") is not None
            else (xp.full_like(equity, xp.nan, dtype=equity.dtype))
        )
        open_lots = (
            to_xp(cols_np["open_lots"], use_gpu_now, dtype=gpu_dtype)
            if cols_np.get("open_lots") is not None
            else (xp.full_like(equity, xp.nan, dtype=equity.dtype))
        )

        # Downsample: índices sobre NumPy (para fechas)
        if (not self.downsample) or (self.max_points is not None and self.max_points <= 0):
            idx_np = np.arange(n, dtype=np.int64)
        else:
            max_pts = self.max_points if self.max_points is not None else 10_000
            idx_np = choose_idx(n, max_pts)

        if use_gpu_now:
            import cupy as cp  # type: ignore

            idx_dtype = cp.int32 if (n < 2**31) else cp.int64
            idx_xp = to_xp(idx_np, True, dtype=idx_dtype)
        else:
            idx_xp = idx_np

        # Series derivadas: Floating vs Initial Balance (%)
        stats = result_backtest.get("statistics", {}) or {}
        init_bal = float(
            stats.get(
                "initial_balance",
                float(to_numpy(equity)[0] if n > 0 else 0.0),
            )
            or 0.0
        )

        if use_gpu_now:
            import cupy as cp  # type: ignore

            denom = cp.asarray(init_bal, dtype=floating.dtype)
            denom = cp.where(denom > 0.0, denom, cp.asarray(1.0, dtype=floating.dtype))
            floating_init_full = cp.around((floating / denom) * cp.asarray(100.0, dtype=floating.dtype), 2)
        else:
            denom = init_bal if init_bal > 0.0 else 1.0
            floating_init_full = np.around((to_numpy(floating) / denom) * 100.0, 2)

        # Descomposición por lados a nivel del subset
        side = {
            "open_trades_long": None,
            "open_trades_short": None,
            "open_lots_long": None,
            "open_lots_short": None,
        }
        if "trades" in result_backtest:
            side = _side_series(dates_np, idx_np, result_backtest["trades"], use_gpu_now)

            # si no hay open_trades/open_lots, usa suma de lados
            if all_nan(open_trades, use_gpu_now) and side["open_trades_long"] is not None:
                open_trades = (  # type: ignore
                    side["open_trades_long"] + side["open_trades_short"]  # type: ignore
                )
            if all_nan(open_lots, use_gpu_now) and side["open_lots_long"] is not None:
                open_lots = xp.around(  # type: ignore
                    side["open_lots_long"] + side["open_lots_short"], 2  # type: ignore
                )

        # Subset y copia final a host (una vez por serie)
        dates = dates_np[idx_np]

        if use_gpu_now:
            import cupy as cp  # type: ignore

            equity_s = to_numpy(cp.take(equity, idx_xp))
            balance_s = to_numpy(cp.take(balance, idx_xp))
            floating_s = to_numpy(cp.take(floating, idx_xp))
            floating_init_s = to_numpy(cp.take(floating_init_full, idx_xp))
            ot_s = to_numpy(cp.take(open_trades, idx_xp))
            ol_s = to_numpy(cp.take(open_lots, idx_xp))
        else:
            equity_s = to_numpy(equity[idx_xp])
            balance_s = to_numpy(balance[idx_xp])
            floating_s = to_numpy(floating[idx_xp])
            floating_init_s = to_numpy(floating_init_full[idx_xp])
            ot_s = to_numpy(open_trades[idx_xp])
            ol_s = to_numpy(open_lots[idx_xp])

        otL_s = olL_s = otS_s = olS_s = None
        if side["open_trades_long"] is not None:
            otL_s = to_numpy(side["open_trades_long"])  # type: ignore
            otS_s = to_numpy(side["open_trades_short"])  # type: ignore
            olL_s = to_numpy(side["open_lots_long"])  # type: ignore
            olS_s = to_numpy(side["open_lots_short"])  # type: ignore

        # ---------------- Deposit Load (%) ----------------
        # Definición (aprox): carga del depósito = suma(|lot| * tick) / initial_balance * 100.
        # tick: $ por punto y por lote. Mide impacto por 1 punto de movimiento agregado, normalizado por el depósito.
        trades = result_backtest.get("trades", [])
        dl_full = _deposit_load_series(dates_np, trades, init_bal)
        dl_s = to_numpy(dl_full[idx_np]) if dl_full.size else np.zeros_like(equity_s)

        # Comisiones por tiempo (barra): open, close, total y acumulado
        comm_open_full, comm_close_full = _commission_series(dates_np, trades)
        # Totales y acumulados sobre TODA la serie (antes de downsample)
        comm_total_full = (comm_open_full + comm_close_full) if comm_open_full.size else np.zeros_like(equity_s)
        comm_total_cum_full = np.cumsum(comm_total_full) if comm_total_full.size else np.zeros_like(equity_s)

        # Subset y redondeo (mostrar valores consistentes con el total acumulado real)
        if comm_open_full.size:
            comm_open_s = np.around(to_numpy(comm_open_full[idx_np]), 2)
            comm_close_s = np.around(to_numpy(comm_close_full[idx_np]), 2)
            comm_total_s = np.around(to_numpy(comm_total_full[idx_np]), 2)
            comm_total_cum_s = np.around(to_numpy(comm_total_cum_full[idx_np]), 2)
        else:
            comm_open_s = np.zeros_like(equity_s)
            comm_close_s = np.zeros_like(equity_s)
            comm_total_s = np.zeros_like(equity_s)
            comm_total_cum_s = np.zeros_like(equity_s)

        return PreparedData(
            dates=dates,
            equity=equity_s,
            balance=balance_s,
            floating=floating_s,
            floating_init_pct=floating_init_s,
            initial_balance=float(init_bal),
            deposit_load_pct=dl_s,
            comm_open=comm_open_s,
            comm_close=comm_close_s,
            comm_total=comm_total_s,
            comm_total_cum=comm_total_cum_s,
            open_trades_total=ot_s,
            open_lots_total=ol_s,
            ot_long=otL_s,
            ot_short=otS_s,
            ol_long=olL_s,
            ol_short=olS_s,
        )


def _deposit_load_series(
    dates_np: np.ndarray,
    trades: List[Dict],
    initial_balance: float,
) -> np.ndarray:
    """Deposit Load (%) aproximado en el tiempo.

    Definición usada: sum(|lot| * tick) / initial_balance * 100.
    - |lot|: tamaño de lote
    - tick: $/punto por 1 lote
    """
    n = dates_np.size
    if n == 0:
        return np.array([], dtype=float)

    diff = np.zeros(n + 1, dtype=float)
    # reconstruir por ticket con open/close
    by_ticket: Dict[Any, Dict[str, Any]] = {}
    for ev in trades or []:
        t = ev.get("ticket")
        if t is None:
            continue
        rec = by_ticket.get(t, {})
        st = str(ev.get("status", "")).lower()
        if st == "open":
            rec.update(
                open_time=ev.get("open_time"),
                lot_size=float(ev.get("lot_size", ev.get("lot", 0.0)) or 0.0),
                point=float(ev.get("point", 0.0) or 0.0),
                tick=float(ev.get("tick", 0.0) or 0.0),
            )
        elif st == "closed":
            rec.setdefault("open_time", ev.get("open_time"))
            if "lot_size" not in rec:
                rec["lot_size"] = float(ev.get("lot_size", ev.get("lot", 0.0)) or 0.0)
            if "point" not in rec:
                rec["point"] = float(ev.get("point", 0.0) or 0.0)
            if "tick" not in rec:
                rec["tick"] = float(ev.get("tick", 0.0) or 0.0)
            rec["close_time"] = ev.get("close_time")
        by_ticket[t] = rec

    d_int = dates_np.astype("datetime64[ns]").astype("int64")
    for rec in by_ticket.values():
        lot = abs(float(rec.get("lot_size", 0.0) or 0.0))
        point = float(rec.get("point", 0.0) or 0.0)
        tick = float(rec.get("tick", 0.0) or 0.0)
        # carga por orden: $ por punto = lot * tick (si falta tick, cae a 0)
        volume = lot * tick if (lot > 0.0 and tick > 0.0) else 0.0
        if volume <= 0.0:
            continue
        try:
            i0 = int(np.searchsorted(d_int, np.datetime64(rec.get("open_time"), "ns").astype("int64"), side="left"))
        except Exception:
            continue
        i0 = max(0, min(n, i0))
        if i0 >= n:
            continue
        if rec.get("close_time") is not None:
            try:
                i1 = int(np.searchsorted(d_int, np.datetime64(rec.get("close_time"), "ns").astype("int64"), side="left"))
            except Exception:
                i1 = n
        else:
            i1 = n
        i1 = max(i0, min(n, i1))
        diff[i0] += volume
        diff[i1] -= volume

    expo = np.cumsum(diff[:-1])
    expo[~np.isfinite(expo)] = 0.0
    den = float(initial_balance if initial_balance and initial_balance > 0.0 else 1.0)
    return np.around((expo / den) * 100.0, 2)


def _commission_series(
    dates_np: np.ndarray,
    trades: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve series por fecha de comisiones de apertura y de cierre.

    Asigna cada comisión a la barra cuya fecha es >= a su time stamp más cercano por la izquierda.
    """
    n = dates_np.size
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    open_series = np.zeros(n, dtype=float)
    close_series = np.zeros(n, dtype=float)

    d_int = dates_np.astype("datetime64[ns]").astype("int64")

    for ev in trades or []:
        st = str(ev.get("status", "")).lower()
        if st == "open":
            ts = ev.get("open_time")
            val = float(ev.get("commission", ev.get("commission_open", 0.0)) or 0.0)
        elif st == "closed":
            ts = ev.get("close_time")
            val = float(ev.get("commission", ev.get("commission_close", 0.0)) or 0.0)
        else:
            continue

        try:
            ti = int(np.datetime64(ts, "ns").astype("int64"))
        except Exception:
            continue
        idx = int(np.searchsorted(d_int, ti, side="left"))
        if idx < 0:
            continue
        if idx >= n:
            idx = n - 1
        if st == "open":
            open_series[idx] += val
        elif st == "closed":
            close_series[idx] += val

    # Redondear a 2 decimales
    return np.around(open_series, 2), np.around(close_series, 2)







