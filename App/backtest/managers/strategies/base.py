# backtest/managers/strategies/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEntryStrategy(ABC):
    """
    Contrato base para estrategias de ENTRADA.
    Cada instancia administra su propio estado por (symbol, magic).
    """
    def __init__(self, name: str, params: Dict[str, Any], context) -> None:
        self.name = name
        self.params = params or {}
        self.ctx = context  # EntryManager (para ejecutar órdenes y utilidades)

        if "magic" not in self.params:
            raise ValueError(f"[{self.name}] falta 'magic' en params.")

    @property
    def magic(self) -> int:
        return int(self.params["magic"])

    # Hooks que puede usar la RiskManager para limpiar estado cuando ya no hay posiciones
    def clear_symbol_data(self, symbol: str, magic: int) -> None:
        """Opcional: sobrescribir si la estrategia mantiene estado por (symbol, magic)."""
        return

    @abstractmethod
    def on_candle(
        self,
        symbol: str,
        signal_buy: bool,
        signal_sell: bool,
        current_bid: float,
        date,
    ) -> None:
        """Se llama una vez por vela/símbolo en el OPEN."""
        ...
