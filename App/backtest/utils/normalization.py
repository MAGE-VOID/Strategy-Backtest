"""
Utilities for consistent price and lot normalization across the backtest.

Avoids code duplication between managers by centralizing the logic here.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Mapping, Any


def normalize_lot(lot: float) -> float:
    """Round lot size to 2 decimals and clamp to minimum 0.01."""
    q = float(Decimal(lot).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return max(0.01, q)


def normalize_price(
    symbol: str,
    price: Optional[float],
    symbol_points_mapping: Mapping[str, Mapping[str, Any]],
) -> Optional[float]:
    """
    Snap price to the symbol tick grid and round to symbol digits when available.

    - If price is None, returns None.
    - Uses mapping keys: "point" (tick size) and optional "digits".
    """
    if price is None:
        return None

    meta = symbol_points_mapping.get(symbol, {})
    point = meta.get("point") or 1e-6
    digits = meta.get("digits")

    snapped = round(price / point) * point
    if digits is not None:
        snapped = float(round(snapped, int(digits)))
    return float(snapped)

