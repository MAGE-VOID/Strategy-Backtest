# backtest/managers/entry_manager.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any
from backtest.managers.position_manager import PositionManager


class EntryManager:
    """
    Gestiona la apertura de posiciones y la lógica auxiliar para estrategias grid.
    """

    def __init__(
        self,
        initial_balance: float,
        strategies_params: Optional[Dict[str, Dict[str, Any]]] = None,
        symbol_points_mapping: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.strategies_params = strategies_params or {}
        self.symbol_points_mapping = symbol_points_mapping or {}

        self.position_manager = PositionManager(initial_balance, symbol_points_mapping)

        # Estructuras auxiliares para grid
        self.first_position_tp: Dict[str, float] = {}
        self.first_position_sl: Dict[str, float] = {}
        self.last_position_price: Dict[str, float] = {}
        self.grid_positions: Dict[str, int] = {}

        # Spread fijo (en puntos). Si usas spread variable cámbialo.
        self.spread_points: int = 2

    # ------------------------------------------------------------------ #
    # Utilidades internas                                                #
    # ------------------------------------------------------------------ #
    def get_symbol_points(self, symbol: str) -> float:
        return self.symbol_points_mapping[symbol]["point"]

    def calculate_tp_sl(
        self,
        bid_price: float,
        point: float,
        tp_distance: int,
        sl_distance: Optional[int],
        is_buy: bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcula TP y SL **en el precio que realmente dispara la orden**.
        BUY  → disparador = Bid ; SELL → disparador = Ask.
        """
        spread_move = self.spread_points * point

        if is_buy:
            trigger = bid_price  # Bid
            tp = trigger + tp_distance * point
            sl = trigger - sl_distance * point if sl_distance is not None else None
        else:
            trigger = bid_price + spread_move  # Ask
            tp = trigger - tp_distance * point
            sl = trigger + sl_distance * point if sl_distance is not None else None

        return tp, sl

    def _has_open(self, symbol: str, position_type: str, magic: int) -> bool:
        return any(
            p
            for p in self.position_manager.positions.values()
            if p["symbol"] == symbol
            and p["position"] == position_type
            and p["magic"] == magic
        )

    # ------------------------------------------------------------------ #
    # Apertura de posiciones                                             #
    # ------------------------------------------------------------------ #
    def _open_position(
        self,
        symbol: str,
        current_bid: float,
        date,
        is_buy: bool,
        params: Dict[str, Any],
        lot_size: Optional[float] = None,
    ) -> None:
        if lot_size is None:
            lot_size = params["initial_lot_size"]

        point = self.get_symbol_points(symbol)
        spread_move = self.spread_points * point

        # Precio de ejecución (Ask para BUY, Bid para SELL)
        entry_price = current_bid + spread_move if is_buy else current_bid

        tp, sl = self.calculate_tp_sl(
            bid_price=current_bid,
            point=point,
            tp_distance=params["tp_distance"],
            sl_distance=params["sl_distance"],
            is_buy=is_buy,
        )

        position_type = "long" if is_buy else "short"

        self.position_manager.open_position(
            symbol=symbol,
            position_type=position_type,
            price=entry_price,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            open_date=date,
            magic=params["magic"],
        )

    # ------------------------------------------------------------------ #
    # Interface usado por BacktestEngine                                 #
    # ------------------------------------------------------------------ #
    def apply_strategy(
        self,
        strategy_name: str,
        symbol: str,
        signal_buy: bool,
        signal_sell: bool,
        current_price: float,
        index: int,
        date,
    ) -> None:
        params = self.strategies_params[strategy_name]
        magic = params["magic"]

        if strategy_name == "simple_buy":
            if signal_buy and not self._has_open(symbol, "long", magic):
                self._open_position(symbol, current_price, date, True, params)

        elif strategy_name == "simple_sell":
            if signal_sell and not self._has_open(symbol, "short", magic):
                self._open_position(symbol, current_price, date, False, params)

        elif strategy_name == "grid_buy" and signal_buy:
            self.grid_buy(symbol, current_price, date, params)

    # ------------------------------------------------------------------ #
    # Estrategia Grid BUY                                                #
    # ------------------------------------------------------------------ #
    def grid_buy(self, symbol: str, current_price: float, date, params: Dict[str, Any]):
        point = self.get_symbol_points(symbol)
        magic = params["magic"]

        longs = [
            p
            for p in self.position_manager.positions.values()
            if p["symbol"] == symbol and p["position"] == "long" and p["magic"] == magic
        ]

        if not longs:
            self._open_position(symbol, current_price, date, True, params)
            self.last_position_price[symbol] = current_price
            self.grid_positions[symbol] = 1
        else:
            last = self.last_position_price.get(symbol)
            if last is None:
                return
            if current_price <= last - params["grid_distance"] * point:
                self._add_grid_position(symbol, current_price, params)

    def _add_grid_position(
        self, symbol: str, current_price: float, params: Dict[str, Any]
    ) -> None:
        count = self.grid_positions.get(symbol, 0)
        lot_size = params["initial_lot_size"] * (params["lot_multiplier"] ** count)
        self._open_position(
            symbol, current_price, None, True, params, lot_size=lot_size
        )
        self.last_position_price[symbol] = current_price
        self.grid_positions[symbol] = count + 1
        self.update_tp_sl(symbol, params)

    def update_tp_sl(self, symbol: str, params: Dict[str, Any]) -> None:
        magic = params["magic"]
        longs = [
            p
            for p in self.position_manager.positions.values()
            if p["symbol"] == symbol and p["position"] == "long" and p["magic"] == magic
        ]
        if not longs:
            return

        prices = np.array([p["entry_price"] for p in longs])  # Ask
        lots = np.array([p["lot_size"] for p in longs])
        total_lots = lots.sum()
        if total_lots == 0:
            return

        avg_ask = np.dot(prices, lots) / total_lots
        point = self.get_symbol_points(symbol)
        spread_move = self.spread_points * point
        bid_equiv = avg_ask - spread_move

        new_tp = bid_equiv + params["tp_distance"] * point
        new_sl = (
            bid_equiv - params["sl_distance"] * point
            if params["sl_distance"] is not None
            else None
        )

        self.position_manager.update_symbol_tp_sl(symbol, tp=new_tp, sl=new_sl)

    # ------------------------------------------------------------------ #
    # Limpieza de estado                                                 #
    # ------------------------------------------------------------------ #
    def clear_symbol_data(self, symbol: str) -> None:
        for d in (
            self.first_position_tp,
            self.first_position_sl,
            self.last_position_price,
            self.grid_positions,
        ):
            d.pop(symbol, None)
