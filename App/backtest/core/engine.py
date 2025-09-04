# backtest/core/engine.py

import numpy as np
import pandas as pd
from backtest.stats import Statistics
from backtest.managers.entry_manager import EntryManager
from backtest.managers.risk_manager import RiskManager
from backtest.core.data import DataPrep
from backtest.core.prices import PriceMatrixBuilder
from backtest.core.signals import SignalBuilder
from backtest.core.simulator import Simulator


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.debug_mode = config.debug_mode
        self.strategies = list(config.strategies_params.keys())
        self.equity_over_time = []

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        # Delegar responsabilidades a componentes core sin cambiar la lógica
        df = DataPrep.prepare_dataframe(input_data)
        dates, symbols = DataPrep.extract_index_symbols(df)
        price_mats = PriceMatrixBuilder.build(df, dates, symbols)
        symbol_points = DataPrep.map_symbol_points(df, symbols)
        em, pm, risk = self._setup_managers(symbol_points, symbols)
        signal_gens, local_idx_map = SignalBuilder.build(
            df, dates, symbols, self.config.strategy_signal_class
        )

        sim = Simulator.run(
            dates,
            symbols,
            price_mats,
            em,
            pm,
            risk,
            signal_gens,
            local_idx_map,
            self.debug_mode,
            strategies_order=self.strategies,
        )
        stats = self._finalize(em, sim)
        return stats

    def _setup_managers(self, symbol_points, symbols):
        em = EntryManager(
            self.config.initial_balance,
            strategies_params=self.config.strategies_params,
            symbol_points_mapping=symbol_points,
        )
        pm = em.position_manager
        pm.sym2idx = {sym: idx for idx, sym in enumerate(symbols)}
        risk = RiskManager(em, pm)
        return em, pm, risk

    def _finalize(self, em, sim):
        stats = Statistics(
            em.position_manager.results,
            em.equity_over_time,
            self.config.initial_balance,
        ).calculate_statistics()
        return {
            "trades": sim["results"],
            "equity_over_time": em.equity_over_time,
            "statistics": stats,
        }

