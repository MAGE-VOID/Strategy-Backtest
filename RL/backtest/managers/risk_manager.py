# backtest/managers/risk_manager.py
from __future__ import annotations
from typing import Optional, Set, Tuple
import numpy as np


class RiskManager:
    """
    Cierre por TP/SL con precios correctos:
      - BUY: TP/SL en BID
      - SELL: TP/SL en ASK = BID + spread
    Resolución intrabar realista (qué toca primero) con modelo OHLC tipo "tick":
      - Vela alcista: O→L→H→C (Low primero)
      - Vela bajista: O→H→L→C (High primero)
    Gaps:
      - Si OPEN pasa por SL (peor): cerrar a OPEN (peor disponible).
      - Si OPEN pasa por TP (mejor): cerrar en TP (sin slippage positiva).
    Limpieza de estado por (symbol, magic).
    """

    def __init__(self, entry_manager, position_manager) -> None:
        self.em = entry_manager
        self.pm = position_manager

    def check_tp_sl(
        self,
        opens,
        lows,
        highs,
        closes,
        date,
        only_opened_before: Optional[object] = None,
        only_opened_on: Optional[object] = None,
    ) -> None:
        """
        Evalúa TP/SL de un subconjunto de posiciones según filtros temporales.
          - only_opened_before: procesa pos con open_dt < date
          - only_opened_on:     procesa pos con open_dt == date
        """
        closed_pairs: Set[Tuple[str, int]] = set()

        def _should_process(pos) -> bool:
            odt = pos.get("open_dt")
            if only_opened_before is not None:
                return (odt is None) or (odt < only_opened_before)
            if only_opened_on is not None:
                return odt == only_opened_on
            return True

        for ticket, pos in list(self.pm.positions.items()):
            if not _should_process(pos):
                continue

            idx = pos["sym_idx"]
            if idx is None:
                continue

            open_bid = opens[idx]
            low_bid = lows[idx]
            high_bid = highs[idx]
            close_bid = closes[idx]
            if not (
                np.isfinite(open_bid)
                and np.isfinite(low_bid)
                and np.isfinite(high_bid)
                and np.isfinite(close_bid)
            ):
                continue

            symbol = pos["symbol"]
            tp, sl = pos.get("tp"), pos.get("sl")
            point = pos["point"]
            spread_move = self.em.spread_points * point
            bullish = close_bid >= open_bid

            # ----------------------- LONG (BUY) ------------------------ #
            if pos["position"] == "long":
                # Gap en apertura
                if tp is not None and open_bid >= tp:
                    exit_price = tp  # sin positiva
                elif sl is not None and open_bid <= sl:
                    exit_price = open_bid  # peor disponible
                else:
                    hit_tp = (tp is not None) and (high_bid >= tp)
                    hit_sl = (sl is not None) and (low_bid <= sl)
                    if not (hit_tp or hit_sl):
                        continue

                    if hit_tp and hit_sl:
                        # Alcista: Low primero ⇒ SL primero; Bajista: High primero ⇒ TP primero
                        exit_price = sl if bullish else tp
                    elif hit_tp:
                        exit_price = tp
                    else:
                        exit_price = sl

            # ----------------------- SHORT (SELL) ---------------------- #
            else:
                # Convertimos BID→ASK para comparar niveles en ventas
                open_ask = open_bid + spread_move
                low_ask = low_bid + spread_move
                high_ask = high_bid + spread_move

                # Gap en apertura
                if tp is not None and open_ask <= tp:
                    exit_price = tp  # sin positiva
                elif sl is not None and open_ask >= sl:
                    exit_price = open_ask  # peor disponible
                else:
                    hit_tp = (tp is not None) and (low_ask <= tp)
                    hit_sl = (sl is not None) and (high_ask >= sl)
                    if not (hit_tp or hit_sl):
                        continue

                    if hit_tp and hit_sl:
                        # Alcista (bid bull): Low primero ⇒ TP primero; Bajista: High primero ⇒ SL primero
                        exit_price = tp if bullish else sl
                    elif hit_tp:
                        exit_price = tp
                    else:
                        exit_price = sl

            # Normalizamos precio de salida al grid del símbolo
            exit_price = self.em.normalize_price(symbol, exit_price)
            self.pm.close_position(ticket, exit_price, date)
            closed_pairs.add((pos["symbol"], pos.get("magic")))

        # Limpieza de estado por (symbol, magic)
        for sym, mag in closed_pairs:
            still_open = any(
                p
                for p in self.pm.positions.values()
                if p["symbol"] == sym and p.get("magic") == mag
            )
            if not still_open:
                self.em.clear_symbol_data(sym, mag)
