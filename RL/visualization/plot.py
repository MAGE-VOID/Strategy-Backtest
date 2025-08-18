# visualization/plot.py
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
from typing import Any  # para anotar sin depender de tipos de cupy

# ----- GPU opcional (CuPy en Windows con CUDA) -----
try:
    import cupy as cp

    HAS_CUPY = True
    # acelerar host<->device: usar memoria "pinned"
    try:
        _PINNED_POOL = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)
    except Exception:
        pass
except Exception:
    cp = None
    HAS_CUPY = False

# (opcional) pandas solo como último recurso para parsear fechas atípicas
try:
    import pandas as pd  # noqa: F401

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# Render en navegador
pio.renderers.default = "browser"


class BacktestPlotter:
    """
    Dashboard de backtest en dark mode (único).

    GPU-first: todos los cálculos pesados se hacen en GPU (CuPy) si está disponible
    y hay suficientes puntos. Solo se copia a NumPy justo antes del plot.

        plotter = BacktestPlotter()                             # usa GPU si CuPy está disponible
        plotter = BacktestPlotter(max_points=200_000)           # más puntos
        plotter = BacktestPlotter(downsample=False)             # TODOS los puntos
        # plotter = BacktestPlotter(use_gpu=False)              # fuerza CPU
        fig = plotter.show(result_backtest)
    """

    TEMPLATE = "plotly_dark"
    MAX_POINTS = 10_000  # más puntos por defecto
    GPU_MIN_POINTS = 10_000  # evita overhead si hay pocos puntos

    # --- Performance knobs ---
    GPU_BLOCK = 256  # hilos por bloque para kernels
    GPU_USE_FP32 = True  # usa float32 en VRAM para ganar ancho de banda

    COLORS = {
        "equity": "#00cc96",
        "balance": "#EF553B",
        "drawdown": "#FFA15A",
        "floating": "#636EFA",
        "open_trades": "#FF851B",
        "open_lots": "#AB63FA",
        "ot_long": "#2ECC71",
        "ot_short": "#E74C3C",
        "ol_long": "#1ABC9C",
        "ol_short": "#C0392B",
    }

    TITLES = (
        "Equity (MTM) y Balance (Step)",
        "Drawdown de Equity (%)",
        "Floating P/L",
        "Open Trades",
        "Open Lots",
    )

    # ---- cache de kernels (evita recompilar) ----
    _kern_cache: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        *,
        use_gpu: bool = True,
        max_points: int | None = None,
        downsample: bool = True,
    ) -> None:
        self.use_gpu_requested = bool(use_gpu and HAS_CUPY)
        self.downsample = bool(downsample)
        # Si max_points es None, usa la constante; si <=0, no se hace downsample (todos los puntos)
        self.max_points = int(max_points) if max_points is not None else self.MAX_POINTS

    # ------------- API pública -------------

    def show(self, result_backtest: dict) -> go.Figure:
        fig = self.build(result_backtest)
        fig.show(config={"responsive": True, "scrollZoom": True})
        return fig

    def build(self, result_backtest: dict) -> go.Figure:
        if "equity_over_time" not in result_backtest:
            raise ValueError(
                "result_backtest debe incluir la clave 'equity_over_time'."
            )

        pio.templates.default = self.TEMPLATE

        # --- Ingesta -> NumPy (host). Sin pandas en el camino crítico.
        dates_np, cols_np = self._prepare_arrays(result_backtest["equity_over_time"])
        n = dates_np.size
        if n == 0:
            raise ValueError("No hay datos en 'equity_over_time' para graficar.")

        # ¿Usamos GPU para este tamaño?
        use_gpu_now = bool(self.use_gpu_requested and (n >= self.GPU_MIN_POINTS))
        xp = cp if use_gpu_now else np

        # dtype objetivo en GPU
        gpu_dtype = (
            cp.float32
            if (use_gpu_now and self.GPU_USE_FP32)
            else (cp.float64 if use_gpu_now else None)
        )

        # --- Sube a GPU una sola vez (si aplica)
        equity = self._to_xp(cols_np["equity"], use_gpu_now, dtype=gpu_dtype)
        balance = self._to_xp(cols_np["balance"], use_gpu_now, dtype=gpu_dtype)

        # Si no viene "floating", se calcula en GPU = equity - balance
        if cols_np.get("floating") is not None:
            floating = self._to_xp(cols_np["floating"], use_gpu_now, dtype=gpu_dtype)
        else:
            floating = equity - balance

        open_trades = (
            self._to_xp(cols_np["open_trades"], use_gpu_now, dtype=gpu_dtype)
            if cols_np.get("open_trades") is not None
            else xp.full_like(equity, xp.nan, dtype=equity.dtype)
        )
        open_lots = (
            self._to_xp(cols_np["open_lots"], use_gpu_now, dtype=gpu_dtype)
            if cols_np.get("open_lots") is not None
            else xp.full_like(equity, xp.nan, dtype=equity.dtype)
        )

        # --- Downsample / índices
        if (not self.downsample) or (
            self.max_points is not None and self.max_points <= 0
        ):
            idx_np = np.arange(n, dtype=np.int64)
        else:
            idx_np = self._choose_idx(n, self.max_points)  # para dates (NumPy)

        if use_gpu_now:
            # usa int32 si alcanza (más rápido)
            idx_dtype = cp.int32 if (n < 2**31) else cp.int64
            idx_xp = self._to_xp(idx_np, True, dtype=idx_dtype)
        else:
            idx_xp = idx_np

        # --- Series derivadas en GPU (cummax O(n) con kernels)
        dd_equity = self._dd_series_fast(equity, xp)  # drawdown (%)

        # --- Lados Long/Short si hay trades
        side = {
            "open_trades_long": None,
            "open_trades_short": None,
            "open_lots_long": None,
            "open_lots_short": None,
        }

        if "trades" in result_backtest:
            side = self._side_series(
                dates_np, idx_np, result_backtest["trades"], use_gpu_now
            )

            # Si open_trades/open_lots no llegaron, usa la suma de lados
            if (
                self._all_nan(open_trades, use_gpu_now)
                and side["open_trades_long"] is not None
            ):
                open_trades = side["open_trades_long"] + side["open_trades_short"]
            if (
                self._all_nan(open_lots, use_gpu_now)
                and side["open_lots_long"] is not None
            ):
                open_lots = xp.around(
                    side["open_lots_long"] + side["open_lots_short"], 2
                )

        # --- Subset por índices (en GPU) y copia final a host (una vez por serie)
        dates = dates_np[idx_np]  # NumPy datetime64 para el eje X

        if use_gpu_now:
            # cp.take suele ser más eficiente que indexado avanzado
            equity_s = self._to_numpy(cp.take(equity, idx_xp))
            balance_s = self._to_numpy(cp.take(balance, idx_xp))
            floating_s = self._to_numpy(cp.take(floating, idx_xp))
            dd_s = self._to_numpy(cp.take(dd_equity, idx_xp))
            ot_s = self._to_numpy(cp.take(open_trades, idx_xp))
            ol_s = self._to_numpy(cp.take(open_lots, idx_xp))
        else:
            equity_s = self._to_numpy(equity[idx_xp])
            balance_s = self._to_numpy(balance[idx_xp])
            floating_s = self._to_numpy(floating[idx_xp])
            dd_s = self._to_numpy(dd_equity[idx_xp])
            ot_s = self._to_numpy(open_trades[idx_xp])
            ol_s = self._to_numpy(open_lots[idx_xp])

        otL_s = olL_s = otS_s = olS_s = None
        if side["open_trades_long"] is not None:
            if use_gpu_now:
                otL_s = self._to_numpy(cp.take(side["open_trades_long"], idx_xp))
                otS_s = self._to_numpy(cp.take(side["open_trades_short"], idx_xp))
                olL_s = self._to_numpy(cp.take(side["open_lots_long"], idx_xp))
                olS_s = self._to_numpy(cp.take(side["open_lots_short"], idx_xp))
            else:
                otL_s = self._to_numpy(side["open_trades_long"][idx_xp])
                otS_s = self._to_numpy(side["open_trades_short"][idx_xp])
                olL_s = self._to_numpy(side["open_lots_long"][idx_xp])
                olS_s = self._to_numpy(side["open_lots_short"][idx_xp])

        # --- Plotly
        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.46, 0.14, 0.14, 0.13, 0.13],
            subplot_titles=self.TITLES,
        )

        # 1) Equity / Balance
        self._line(fig, dates, equity_s, "Equity", self.COLORS["equity"], row=1)
        self._line(
            fig, dates, balance_s, "Balance", self.COLORS["balance"], row=1, step=True
        )

        # 2) Drawdown
        self._line(fig, dates, dd_s, "DD Equity (%)", self.COLORS["drawdown"], row=2)

        # 3) Floating P/L
        self._line(
            fig, dates, floating_s, "Floating P/L", self.COLORS["floating"], row=3
        )

        # 4) Open Trades
        self._line(
            fig, dates, ot_s, "Open Trades (Total)", self.COLORS["open_trades"], row=4
        )
        if otL_s is not None:
            self._line(
                fig,
                dates,
                otL_s,
                "Open Trades (Long)",
                self.COLORS["ot_long"],
                row=4,
                dash="dot",
                visible=False,
            )
            self._line(
                fig,
                dates,
                otS_s,
                "Open Trades (Short)",
                self.COLORS["ot_short"],
                row=4,
                dash="dot",
                visible=False,
            )

        # 5) Open Lots
        self._line(
            fig, dates, ol_s, "Open Lots (Total)", self.COLORS["open_lots"], row=5
        )
        if olL_s is not None:
            self._line(
                fig,
                dates,
                olL_s,
                "Open Lots (Long)",
                self.COLORS["ol_long"],
                row=5,
                dash="dot",
                visible=False,
            )
            self._line(
                fig,
                dates,
                olS_s,
                "Open Lots (Short)",
                self.COLORS["ol_short"],
                row=5,
                dash="dot",
                visible=False,
            )

        fig.update_layout(
            hovermode="x unified",
            legend_title="Metric",
            title="Cuenta única con múltiples estrategias (aisladas por symbol+magic)",
            xaxis_title="Date",
            dragmode="zoom",
            autosize=True,
        )
        fig.update_yaxes(title_text="Value", row=1, col=1, fixedrange=True)
        fig.update_yaxes(title_text="DD [%]", row=2, col=1, fixedrange=True)
        fig.update_yaxes(
            title_text="Floating P/L", row=3, col=1, zeroline=True, fixedrange=True
        )
        fig.update_yaxes(
            title_text="Open Trades", row=4, col=1, rangemode="tozero", fixedrange=True
        )
        fig.update_yaxes(
            title_text="Open Lots", row=5, col=1, rangemode="tozero", fixedrange=True
        )

        return fig

    # ================== Helpers internos (GPU-first) ==================

    @staticmethod
    def _choose_idx(n: int, max_points: int) -> np.ndarray:
        if n <= max_points:
            return np.arange(n, dtype=np.int64)
        ev_idx = np.linspace(0, n - 1, max_points).round().astype(np.int64)
        ev_idx[-1] = n - 1
        return np.unique(ev_idx)

    @staticmethod
    def _parse_dt64_ns(x) -> np.datetime64:
        try:
            if isinstance(x, np.datetime64):
                return x.astype("datetime64[ns]")
            if isinstance(x, datetime):
                return np.datetime64(x, "ns")
            return np.datetime64(x, "ns")
        except Exception:
            if HAS_PANDAS:
                return np.datetime64(pd.to_datetime(x), "ns")  # type: ignore
            raise

    def _prepare_arrays(
        self, equity_over_time: list[dict]
    ) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
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
            dts.append(self._parse_dt64_ns(dt))
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

    @staticmethod
    def _to_xp(arr: np.ndarray, use_gpu: bool, dtype=None):
        if not use_gpu:
            return arr.astype(dtype) if dtype is not None else arr
        a = cp.asarray(arr)
        return a.astype(dtype) if dtype is not None else a

    @staticmethod
    def _to_numpy(arr):
        if HAS_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    @staticmethod
    def _all_nan(arr, use_gpu: bool) -> bool:
        if use_gpu and HAS_CUPY and isinstance(arr, cp.ndarray):
            return bool(cp.isnan(arr).all().item())
        return bool(np.isnan(arr).all())

    # --------- Cummax + DD super rápidos en GPU ---------

    def _get_cummax_kernels(self, dtype: str) -> dict[str, Any]:
        """
        Compila y cachea kernels para cummax O(n) + offsets.
        dtype: 'f32' o 'f64'
        """
        if dtype in self._kern_cache:
            return self._kern_cache[dtype]

        assert HAS_CUPY, "CuPy requerido"
        if dtype == "f32":
            T = "float"
            MINUS_INF = "-3.402823466e+38f"  # -FLT_MAX
            SUFFIX = "f32"
        else:
            T = "double"
            MINUS_INF = "-1.7976931348623157e+308"  # -DBL_MAX
            SUFFIX = "f64"

        block_scan_src = rf"""
        extern "C" __global__
        void block_cummax_{SUFFIX}(const {T}* __restrict__ x,
                                   const {T}* __restrict__ __x_dummy,
                                   {T}* __restrict__ y,
                                   {T}* __restrict__ block_last,
                                   const int n)
        {{
            extern __shared__ {T} s[];
            const int tid   = threadIdx.x;
            const int start = blockIdx.x * blockDim.x;
            const int gid   = start + tid;

            {T} val = ({T}){MINUS_INF};
            if (gid < n) val = x[gid];
            s[tid] = val;
            __syncthreads();

            // inclusive scan (max) en el bloque
            for (int offset = 1; offset < blockDim.x; offset <<= 1) {{
                {T} tmp = s[tid];
                if (tid >= offset) {{
                    {T} other = s[tid - offset];
                    tmp = (other > tmp) ? other : tmp;
                }}
                __syncthreads();
                s[tid] = tmp;
                __syncthreads();
            }}

            if (gid < n) y[gid] = s[tid];

            // escribir el último válido del bloque
            int last_valid = n - start - 1;
            if (last_valid < 0) last_valid = 0;
            int last = (blockDim.x - 1 < last_valid) ? (blockDim.x - 1) : last_valid;
            if (tid == last) {{
                block_last[blockIdx.x] = s[last];
            }}
        }}
        """

        prefix_src = rf"""
        extern "C" __global__
        void block_exclusive_prefix_max_{SUFFIX}(const {T}* __restrict__ in_last,
                                                 {T}* __restrict__ out_prefix,
                                                 const int nblocks)
        {{
            if (blockIdx.x == 0 && threadIdx.x == 0) {{
                {T} run = ({T}){MINUS_INF};
                for (int i = 0; i < nblocks; ++i) {{
                    out_prefix[i] = run;          // exclusivo
                    {T} v = in_last[i];
                    run = (v > run) ? v : run;
                }}
            }}
        }}
        """

        apply_src = rf"""
        extern "C" __global__
        void apply_block_offsets_{SUFFIX}({T}* __restrict__ y,
                                          const {T}* __restrict__ block_prefix,
                                          const int n)
        {{
            const int bid = blockIdx.x;
            const int tid = threadIdx.x;
            const int gid = bid * blockDim.x + tid;
            if (gid < n) {{
                {T} off = block_prefix[bid];
                {T} val = y[gid];
                y[gid]  = (off > val) ? off : val;
            }}
        }}
        """

        kernels = {
            "block_scan": cp.RawKernel(block_scan_src, f"block_cummax_{SUFFIX}"),
            "block_prefix": cp.RawKernel(
                prefix_src, f"block_exclusive_prefix_max_{SUFFIX}"
            ),
            "apply_offsets": cp.RawKernel(apply_src, f"apply_block_offsets_{SUFFIX}"),
        }
        self._kern_cache[dtype] = kernels
        return kernels

    def _cummax_gpu_fast(self, v: Any) -> Any:
        """
        Cummax O(n) en GPU usando kernels (sin cp.maximum.accumulate).
        Devuelve un nuevo array con el máximo prefijo.
        """
        n = int(v.size)
        if n == 0:
            return v.copy()

        dtype = "f32" if v.dtype == cp.float32 else "f64"
        ks = self._get_cummax_kernels(dtype)

        threads = self.GPU_BLOCK
        blocks = (n + threads - 1) // threads
        smem = threads * v.dtype.itemsize

        y = cp.empty_like(v)
        block_last = cp.empty(blocks, dtype=v.dtype)
        block_prefix = cp.empty(blocks, dtype=v.dtype)

        ks["block_scan"](
            (blocks,), (threads,), (v, v, y, block_last, n), shared_mem=smem
        )
        ks["block_prefix"]((1,), (1,), (block_last, block_prefix, blocks))
        ks["apply_offsets"]((blocks,), (threads,), (y, block_prefix, n))

        return y

    @staticmethod
    def _dd_from_cummax(v, m, xp):
        denom = xp.where(m == 0.0, 1.0, m)
        ratio = (m - v) / denom
        ratio = xp.where(m > 0.0, ratio, 0.0)
        return -ratio * (100.0 if xp is np else xp.asarray(100.0, dtype=m.dtype))

    def _dd_series_fast(self, vals, xp):
        if vals.size == 0:
            return vals
        if (xp is np) or (cp is None) or (xp is not cp):
            v = np.asarray(vals, dtype=np.float64)
            m = np.maximum.accumulate(v)
            return self._dd_from_cummax(v, m, np)
        v = vals.astype(vals.dtype, copy=False)
        try:
            m = self._cummax_gpu_fast(v)
        except Exception:
            # Fallback: método O(n log n) si algo falla
            m = v.copy()
            step = 1
            while step < m.size:
                cp.maximum(m[step:], m[:-step], out=m[step:])
                step <<= 1
        return self._dd_from_cummax(v, m, cp)

    # ----------------- Lados Long/Short -----------------

    def _side_series(
        self,
        dates_np: np.ndarray,
        idx_subset_np: np.ndarray,
        trades: list[dict],
        use_gpu: bool,
    ) -> dict[str, np.ndarray | None]:
        xp = cp if (use_gpu and HAS_CUPY) else np
        out = {
            "open_trades_long": None,
            "open_trades_short": None,
            "open_lots_long": None,
            "open_lots_short": None,
        }
        if not trades or dates_np.size == 0 or idx_subset_np.size == 0:
            return out

        # Compactar por ticket (CPU)
        tmap: dict = {}
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
                    "open": self._parse_dt64_ns(r.get("open_time")),
                    "close": None,
                }
            elif r.get("status") == "closed":
                x = tmap.get(
                    k,
                    {
                        "type": typ,
                        "lot": lot,
                        "open": (
                            self._parse_dt64_ns(r.get("open_time", None))
                            if r.get("open_time", None) is not None
                            else None
                        ),
                        "close": None,
                    },
                )
                x["close"] = self._parse_dt64_ns(r.get("close_time"))
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

        # Ordena eventos y acumula en GPU/CPU
        ev_idx_np = np.asarray(ev_idx, dtype=np.int64)
        order = np.argsort(ev_idx_np)
        ev_idx_np = ev_idx_np[order]

        dcl_xp = xp.cumsum(
            self._to_xp(
                np.asarray(dcl, dtype=np.float32 if self.GPU_USE_FP32 else np.float64)[
                    order
                ],
                use_gpu,
            )
        )
        dll_xp = xp.cumsum(
            self._to_xp(
                np.asarray(dll, dtype=np.float32 if self.GPU_USE_FP32 else np.float64)[
                    order
                ],
                use_gpu,
            )
        )
        dcs_xp = xp.cumsum(
            self._to_xp(
                np.asarray(dcs, dtype=np.float32 if self.GPU_USE_FP32 else np.float64)[
                    order
                ],
                use_gpu,
            )
        )
        dls_xp = xp.cumsum(
            self._to_xp(
                np.asarray(dls, dtype=np.float32 if self.GPU_USE_FP32 else np.float64)[
                    order
                ],
                use_gpu,
            )
        )

        # Para cada idx del subset, tomar el acumulado más reciente
        r_np = np.searchsorted(ev_idx_np, idx_subset_np, side="right") - 1
        mask_np = r_np >= 0
        r_clip_np = np.clip(r_np, 0, None)

        if use_gpu and HAS_CUPY:
            r_gpu = cp.asarray(
                r_clip_np, dtype=cp.int32 if (len(ev_idx_np) < 2**31) else cp.int64
            )
            mask_gpu = cp.asarray(mask_np)

            def pick(ps_xp):
                base = ps_xp[r_gpu]
                return cp.where(mask_gpu, base, 0.0)

        else:

            def pick(ps_np):
                base = ps_np[r_clip_np]
                return np.where(mask_np, base, 0.0)

        return {
            "open_trades_long": pick(dcl_xp),
            "open_trades_short": pick(dcs_xp),
            "open_lots_long": xp.around(pick(dll_xp), 2),
            "open_lots_short": xp.around(pick(dls_xp), 2),
        }

    # --------- utilidades de plotting ---------

    @staticmethod
    def _step_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convierte una serie (x,y) en forma 'hv' duplicando puntos para usar Scattergl."""
        n = len(x)
        if n <= 1:
            return x, y
        x2 = np.empty(2 * n - 1, dtype=x.dtype)
        y2 = np.empty(2 * n - 1, dtype=y.dtype)
        # primer punto
        x2[0] = x[0]
        y2[0] = y[0]
        # horizontales hasta el siguiente x con el y previo
        x2[1::2] = x[1:]
        y2[1::2] = y[:-1]
        # verticales en el mismo x con el nuevo y
        x2[2::2] = x[1:]
        y2[2::2] = y[1:]
        return x2, y2

    @staticmethod
    def _line(
        fig: go.Figure,
        x,
        y,
        name: str,
        color: str,
        row: int,
        *,
        step: bool = False,
        dash: str | None = None,
        visible: bool | None = True,
    ) -> None:
        # Siempre Scattergl (WebGL). Si step=True, generamos la forma 'hv' por datos.
        if step:
            x, y = BacktestPlotter._step_xy(np.asarray(x), np.asarray(y))
        trace_cls = go.Scattergl
        kw = dict(
            x=x,
            y=y,
            mode="lines",
            name=name,
            hovertemplate="%{y}<extra>" + name + "</extra>",
            line=dict(width=1, color=color, dash=dash),
        )
        fig.add_trace(
            trace_cls(**kw, visible=("legendonly" if visible is False else True)),
            row=row,
            col=1,
        )
