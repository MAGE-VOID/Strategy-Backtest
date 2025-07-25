# backtest/managers/risk_manager.py

class RiskManager:
    def __init__(self, entry_manager, position_manager):
        self.em = entry_manager
        self.pm = position_manager

    def check_tp_sl(self, lows, highs, symbols, date):
        """
        Recorre todas las posiciones y cierra las que alcancen TP o SL.
        Luego borra el estado de grid de los símbolos que hayan quedado sin posiciones.
        """
        closed_symbols = set()
        # 1) Cerrar por TP/SL
        for ticket, pos in list(self.pm.positions.items()):
            sym = pos["symbol"]
            idx = pos["sym_idx"]
            low, high = lows[idx], highs[idx]
            tp, sl = pos.get("tp"), pos.get("sl")

            hit_tp = pos["position"] == "long"  and tp is not None and high >= tp \
                  or pos["position"] == "short" and tp is not None and low  <= tp
            hit_sl = pos["position"] == "long"  and sl is not None and low  <= sl \
                  or pos["position"] == "short" and sl is not None and high >= sl

            if hit_tp:
                self.pm.close_position(ticket, tp, date)
                closed_symbols.add(sym)
            elif hit_sl:
                self.pm.close_position(ticket, sl, date)
                closed_symbols.add(sym)

        # 2) Limpiar estado de grid de los símbolos que ya no tienen posiciones
        for sym in closed_symbols:
            if not self.em.get_positions(sym):
                self.em.clear_symbol_data(sym)
