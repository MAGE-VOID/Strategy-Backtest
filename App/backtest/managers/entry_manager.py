# backtest/managers/entry_manager.py
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import importlib
import pkgutil
from decimal import Decimal, ROUND_HALF_UP

from backtest.managers.position_manager import PositionManager
from backtest.managers.strategies.registry import StrategyRegistry
import backtest.managers.strategies as strategies_pkg  # solo para descubrir módulos


class EntryManager:
    """
    Orquesta la ejecución de estrategias de ENTRADA.
    - Descubre e importa automáticamente todos los módulos en managers/strategies/*
      (excepto base/registry), sin usar __init__.py para efectos secundarios.
    - Instancia estrategias registradas en StrategyRegistry.
    - Expone utilidades comunes (open_position, calculate_tp_sl, etc.)
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

        # Spread fijo en puntos (puedes pasarlo a config si quieres)
        self.spread_points: int = 2

        # 1) Carga dinámica de plugins (no depende de __init__.py)
        self._load_strategy_plugins()

        # 2) Instanciar estrategias declaradas en config
        self._strategies: Dict[str, Any] = {}
        unknown = []
        for key, params in self.strategies_params.items():
            try:
                self._strategies[key] = StrategyRegistry.create(
                    key, params, context=self
                )
            except KeyError:
                unknown.append(key)

        if unknown:
            raise KeyError(
                f"Estrategias no registradas: {unknown}. "
                f"Disponibles: {StrategyRegistry.available()}"
            )

        # equity (lo rellena BacktestEngine)
        self.equity_over_time = []

    # --------- Descubrimiento/registro sin __init__.py --------- #
    def _load_strategy_plugins(self) -> None:
        """
        Importa todos los módulos en backtest.managers.strategies.*
        (excepto base y registry), para que se ejecuten los decoradores @register_strategy.
        """
        prefix = strategies_pkg.__name__ + "."
        for modinfo in pkgutil.iter_modules(strategies_pkg.__path__, prefix):
            mod_name = modinfo.name.rsplit(".", 1)[-1]
            if mod_name.startswith("_") or mod_name in ("base", "registry"):
                continue
            importlib.import_module(modinfo.name)

    # ----------------- Utilidades de normalización ----------------- #
    def normalize_price(self, symbol: str, price: Optional[float]) -> Optional[float]:
        if price is None:
            return None
        meta = self.symbol_points_mapping.get(symbol, {})
        point = meta.get("point") or 1e-6
        digits = meta.get("digits")
        # Ajuste a la grilla de ticks
        snapped = round(price / point) * point
        # Redondeo a dígitos (si disponible)
        if digits is not None:
            snapped = float(round(snapped, int(digits)))
        return float(snapped)

    def normalize_lot(self, lot: float) -> float:
        # 2 decimales (estándar FX), mínimo 0.01
        q = float(Decimal(lot).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return max(0.01, q)

    # ----------------- Utilidades de mercado/orden ----------------- #
    def get_symbol_points(self, symbol: str) -> float:
        return self.symbol_points_mapping[symbol]["point"]

    def calculate_tp_sl(
        self,
        symbol: str,
        bid_price: float,
        point: float,
        tp_distance: int,
        sl_distance: Optional[int],
        is_buy: bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        TP/SL en el precio DISPARADOR correcto:
        - BUY: TP/SL disparan en BID
        - SELL: TP/SL disparan en ASK (= BID + spread)
        Todos los niveles se normalizan a dígitos y grilla de ticks.
        """
        spread_move = self.spread_points * point
        if is_buy:
            tp = bid_price + tp_distance * point if tp_distance is not None else None
            sl = bid_price - sl_distance * point if sl_distance is not None else None
        else:
            ask_price = bid_price + spread_move
            tp = ask_price - tp_distance * point if tp_distance is not None else None
            sl = ask_price + sl_distance * point if sl_distance is not None else None

        tp = self.normalize_price(symbol, tp)
        sl = self.normalize_price(symbol, sl)
        return tp, sl

    def has_open(self, symbol: str, position_type: str, magic: int) -> bool:
        return any(
            p
            for p in self.position_manager.positions.values()
            if p["symbol"] == symbol
            and p["position"] == position_type
            and p["magic"] == magic
        )

    def open_position(
        self,
        symbol: str,
        current_bid: float,
        date,
        is_buy: bool,
        params: Dict[str, Any],
        lot_size: Optional[float] = None,
    ) -> None:
        lot = self.normalize_lot(
            lot_size if lot_size is not None else params["initial_lot_size"]
        )
        point = self.get_symbol_points(symbol)
        spread_move = self.spread_points * point

        # Precio de entrada (BUY→Ask, SELL→Bid), normalizado
        raw_entry = current_bid + spread_move if is_buy else current_bid
        entry_price = self.normalize_price(symbol, raw_entry)

        tp, sl = self.calculate_tp_sl(
            symbol=symbol,
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
            lot_size=lot,
            sl=sl,
            tp=tp,
            open_date=date,
            magic=params["magic"],
        )

    # ----------------- API usado por BacktestEngine ----------------- #
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
        strat = self._strategies.get(strategy_name)
        if strat is None:
            return
        strat.on_candle(symbol, signal_buy, signal_sell, current_price, date)

    # ----------------- Limpieza (invocada por RiskManager) --------- #
    def clear_symbol_data(self, symbol: str, magic: int) -> None:
        for strat in self._strategies.values():
            strat.clear_symbol_data(symbol, magic)
