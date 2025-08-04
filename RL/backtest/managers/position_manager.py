# backtest/managers/position_manager.py

from datetime import datetime


class PositionManager:
    def __init__(self, balance, symbol_points_mapping=None):
        self.positions = {}
        self.balance = balance
        self.ticket_counter = 0
        self.global_counter = 0
        self.results = []

        # Para calcular metadata
        self.default_magic = None
        self.symbol_points_mapping = symbol_points_mapping or {}
        self.sym2idx = {}  # se inyecta desde el engine

    def open_position(
        self,
        symbol,
        position_type,
        price,
        lot_size,
        sl=None,
        tp=None,
        open_date=None,
    ):
        open_time = (
            open_date.strftime("%Y-%m-%d %H:%M:%S")
            if open_date
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # metadata calculada internamente
        sym_idx = self.sym2idx.get(symbol)
        meta = self.symbol_points_mapping.get(symbol, {})
        point = meta.get("point")
        tick = meta.get("tick_value")
        direction = 1 if position_type == "long" else -1
        magic = self.default_magic

        # guardo en memoria
        pos = {
            "symbol": symbol,
            "position": position_type,
            "entry_price": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "open_time": open_time,
            "status": "open",
            # metadata
            "sym_idx": sym_idx,
            "point": point,
            "tick": tick,
            "dir": direction,
            "magic": magic,
        }
        self.positions[self.ticket_counter] = pos

        # y también en el log de resultados
        result = {
            "count": self.global_counter + 1,
            "symbol": symbol,
            "ticket": self.ticket_counter,
            "type": position_type,
            "entry": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "open_time": open_time,
            "status": "open",
            # metadata
            "sym_idx": sym_idx,
            "point": point,
            "tick": tick,
            "dir": direction,
            "magic": magic,
        }
        self.results.append(result)

        self.ticket_counter += 1
        self.global_counter += 1

    def close_position(self, ticket, current_price, close_date):
        position = self.positions.pop(ticket, None)
        if position is None:
            return

        symbol = position["symbol"]
        meta = self.symbol_points_mapping[symbol]
        point = meta["point"]
        tick_value = meta["tick_value"]

        # cálculo de profit
        if position["position"] == "long":
            diff = current_price - position["entry_price"]
        else:
            diff = position["entry_price"] - current_price

        profit = (diff / point) * tick_value * position["lot_size"]
        self.balance += profit

        # registro del cierre, con la misma metadata
        result = {
            "count": self.global_counter + 1,
            "symbol": symbol,
            "ticket": ticket,
            "type": position["position"],
            "entry": position["entry_price"],
            "close_time": close_date.strftime("%Y-%m-%d %H:%M:%S"),
            "exit": current_price,
            "profit": profit,
            "balance": self.balance,
            "status": "closed",
            # metadata
            "sym_idx": position.get("sym_idx"),
            "point": position.get("point"),
            "tick": position.get("tick"),
            "dir": position.get("dir"),
            "magic": position.get("magic"),
        }
        self.results.append(result)
        self.global_counter += 1

    def update_position_tp_sl(self, ticket, tp=None, sl=None):
        pos = self.positions.get(ticket)
        if not pos:
            return
        if tp is not None:
            pos["tp"] = tp
        if sl is not None:
            pos["sl"] = sl

        # reflejarlo también en el log de apertura
        for rec in reversed(self.results):
            if rec.get("ticket") == ticket and rec.get("status") == "open":
                if tp is not None:
                    rec["tp"] = tp
                if sl is not None:
                    rec["sl"] = sl
                break

    def update_symbol_tp_sl(self, symbol, tp=None, sl=None):
        for ticket, pos in self.positions.items():
            if pos["symbol"] == symbol:
                self.update_position_tp_sl(ticket, tp, sl)

    def get_results(self):
        return self.results

    def get_balance(self):
        return self.balance
