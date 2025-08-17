# backtest/managers/strategies/registry.py
from __future__ import annotations
from typing import Dict, Type

class StrategyRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, key: str, strategy_cls: Type) -> None:
        key = key.strip().lower()
        if key in cls._registry:
            raise ValueError(f"Estrategia '{key}' ya registrada.")
        cls._registry[key] = strategy_cls

    @classmethod
    def get(cls, key: str):
        key = key.strip().lower()
        if key not in cls._registry:
            raise KeyError(
                f"Estrategia '{key}' no encontrada en el registro. "
                f"Disponibles: {list(cls._registry.keys())}"
            )
        return cls._registry[key]

    @classmethod
    def create(cls, key: str, params: dict, context):
        return cls.get(key)(name=key, params=params, context=context)

    @classmethod
    def available(cls):
        return sorted(cls._registry.keys())


def register_strategy(key: str):
    def _decorator(strategy_cls):
        StrategyRegistry.register(key, strategy_cls)
        return strategy_cls
    return _decorator
