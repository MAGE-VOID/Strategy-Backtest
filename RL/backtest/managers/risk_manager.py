# backtest/managers/risk_manager.py
from __future__ import annotations
from typing import Set


class RiskManager:
    """
    Cierra posiciones cuando Bid/Ask alcanzan TP o SL.
    """

    def __init__(self, entry_manager, position_manager) -> None:
        self.em = entry_manager
        self.pm = position_manager

    # ------------------------------------------------------------------ #
    def check_tp_sl(self, lows, highs, symbols, date) -> None:
        """
        Recorre todas las posiciones abiertas y las cierra al TP/SL.
        Los niveles TP/SL ya incluyen la lógica Bid/Ask, por lo que no
        hay que volver a sumar o restar el spread.
        """
        closed_symbols: Set[str] = set()

        for ticket, pos in list(self.pm.positions.items()):
            idx = pos["sym_idx"]
            low_bid, high_bid = lows[idx], highs[idx]
            tp, sl = pos.get("tp"), pos.get("sl")

            point = pos["point"]
            spread_move = self.em.spread_points * point

            # ------------------------- BUY ----------------------------- #
            if pos["position"] == "long":
                hit_tp = tp is not None and high_bid >= tp
                hit_sl = sl is not None and low_bid <= sl
                if not (hit_tp or hit_sl):
                    continue

                if hit_tp and hit_sl:
                    entry_ask = pos["entry_price"]
                    entry_bid = entry_ask - spread_move
                    level = tp if (tp - entry_bid) <= (entry_bid - sl) else sl
                elif hit_tp:
                    level = tp
                else:
                    level = sl

                exit_price = level  # Bid

            # ------------------------ SELL ----------------------------- #
            else:
                ask_low = low_bid + spread_move
                ask_high = high_bid + spread_move

                hit_tp = tp is not None and ask_low <= tp
                hit_sl = sl is not None and ask_high >= sl
                if not (hit_tp or hit_sl):
                    continue

                if hit_tp and hit_sl:
                    entry_bid = pos["entry_price"]
                    entry_ask = entry_bid + spread_move
                    level = tp if (entry_ask - tp) <= (sl - entry_ask) else sl
                elif hit_tp:
                    level = tp
                else:
                    level = sl

                exit_price = level  # Ask

            # Cerrar posición
            self.pm.close_position(ticket, exit_price, date)
            closed_symbols.add(pos["symbol"])

        # Limpieza de estado para símbolos sin posiciones abiertas
        for sym in closed_symbols:
            still_open = any(
                p for p in self.pm.positions.values() if p["symbol"] == sym
            )
            if not still_open:
                self.em.clear_symbol_data(sym)
