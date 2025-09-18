from __future__ import annotations

"""
gpu_utils.py

Utilidades de GPU/CPU compartidas para acelerar cálculos intensivos.
Se intenta usar CuPy si está disponible; si no, se cae a NumPy sin romper.
"""

from typing import Any, Dict

import numpy as np

try:
    import cupy as cp  # type: ignore

    HAS_CUPY = True
    try:
        _PINNED_POOL = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)
    except Exception:
        # No es crítico si falla la memoria pinned
        pass
except Exception:  # pragma: no cover - entorno sin CuPy
    cp = None  # type: ignore
    HAS_CUPY = False


# Parámetros de rendimiento por defecto
GPU_MIN_POINTS = 10_000  # evitar overhead de GPU para series pequeñas
GPU_BLOCK = 256          # hilos por bloque para kernels
GPU_USE_FP32 = True      # usar float32 en GPU para mayor ancho de banda


def choose_idx(n: int, max_points: int) -> np.ndarray:
    """Elige índices equiespaciados hasta `max_points` (incluye el último)."""
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    ev_idx = np.linspace(0, n - 1, max_points).round().astype(np.int64)
    ev_idx[-1] = n - 1
    return np.unique(ev_idx)


def to_xp(arr: np.ndarray, use_gpu: bool, dtype=None):
    if not use_gpu:
        return arr.astype(dtype) if dtype is not None else arr
    assert HAS_CUPY and cp is not None
    a = cp.asarray(arr)
    return a.astype(dtype) if dtype is not None else a


def to_numpy(arr):
    if HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def all_nan(arr, use_gpu: bool) -> bool:
    if use_gpu and HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        return bool(cp.isnan(arr).all().item())
    return bool(np.isnan(arr).all())


# ----------------- Cummax + DD rápidos -----------------

_kern_cache: Dict[str, Dict[str, Any]] = {}


def _get_cummax_kernels(dtype: str) -> Dict[str, Any]:
    """Compila y cachea kernels para cummax O(n) + offsets.
    dtype: 'f32' o 'f64'
    """
    assert HAS_CUPY and cp is not None, "CuPy requerido"
    if dtype in _kern_cache:
        return _kern_cache[dtype]

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
    _kern_cache[dtype] = kernels
    return kernels


def _cummax_gpu_fast(v: Any) -> Any:
    """Cummax O(n) en GPU usando kernels (sin cp.maximum.accumulate)."""
    assert HAS_CUPY and cp is not None
    n = int(v.size)
    if n == 0:
        return v.copy()

    dtype = "f32" if v.dtype == cp.float32 else "f64"
    ks = _get_cummax_kernels(dtype)

    threads = GPU_BLOCK
    blocks = (n + threads - 1) // threads
    smem = threads * v.dtype.itemsize

    y = cp.empty_like(v)
    block_last = cp.empty(blocks, dtype=v.dtype)
    block_prefix = cp.empty(blocks, dtype=v.dtype)

    ks["block_scan"]((blocks,), (threads,), (v, v, y, block_last, n), shared_mem=smem)
    ks["block_prefix"]((1,), (1,), (block_last, block_prefix, blocks))
    ks["apply_offsets"]((blocks,), (threads,), (y, block_prefix, n))

    return y


def _dd_from_cummax(v, m, xp):
    denom = xp.where(m == 0.0, 1.0, m)
    ratio = (m - v) / denom
    ratio = xp.where(m > 0.0, ratio, 0.0)
    return -ratio * (100.0 if xp is np else xp.asarray(100.0, dtype=m.dtype))


def dd_series_fast(vals, xp):
    """Calcula drawdown (%) desde máximos previos de la serie `vals`."""
    if vals.size == 0:
        return vals
    if (xp is np) or (cp is None) or (xp is not cp):
        v = np.asarray(vals, dtype=np.float64)
        m = np.maximum.accumulate(v)
        return _dd_from_cummax(v, m, np)
    v = vals.astype(vals.dtype, copy=False)
    try:
        m = _cummax_gpu_fast(v)
    except Exception:
        # Fallback: método O(n log n) si algo falla
        m = v.copy()
        step = 1
        while step < m.size:
            cp.maximum(m[step:], m[:-step], out=m[step:])
            step <<= 1
    return _dd_from_cummax(v, m, cp)

