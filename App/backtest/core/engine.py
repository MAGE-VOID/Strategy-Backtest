# backtest/core/engine.py

import pandas as pd
from backtest.analysis.statistics import Statistics
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

    def run_backtest(self, input_data: pd.DataFrame) -> dict:
        # Delegar responsabilidades a componentes core sin cambiar la lógica
        df = DataPrep.prepare_dataframe(input_data)
        dates, symbols = DataPrep.extract_index_symbols(df)
        price_mats = PriceMatrixBuilder.build(df, dates, symbols)
        symbol_points = DataPrep.map_symbol_points(df, symbols)
        em, pm, risk = self._setup_managers(symbol_points, symbols)
        # Build signal generators per strategy: cada estrategia debe definir
        # explícitamente su propia clase de señales.
        strategy_signal_classes = {}
        for strat in self.strategies:
            params = self.config.strategies_params.get(strat, {})
            signal_cls = params.get("strategy_signal_class", None)
            if signal_cls is None:
                raise ValueError(
                    f"Estrategia '{strat}' sin 'strategy_signal_class' definido en strategies_params."
                )
            strategy_signal_classes[strat] = signal_cls

        # Construir generadores de señales por estrategia/símbolo (vela por vela).
        signal_gens, local_idx_map = SignalBuilder.build_multi(
            df, dates, symbols, strategy_signal_classes
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
        result = self._finalize(em, sim)

        # Salidas opcionales controladas por configuración
        stats_mode = getattr(self.config, "get_statistics", "off")
        if stats_mode == "print":
            try:
                from backtest.utils.formatters import format_statistics
                import pandas as pd

                stats_df = format_statistics(result["statistics"])
                print("\n--- Backtest Statistics ---")
                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", None,
                    "display.max_colwidth", None,
                ):
                    print(stats_df.to_string(index=True, max_rows=None, max_cols=None, line_width=None))
            except Exception as e:
                print(f"[BacktestEngine] Error al imprimir estadísticas: {e}")
        elif stats_mode == "json":
            try:
                from backtest.utils.formatters import statistics_to_json

                result_json = statistics_to_json(result["statistics"])
                # Unica clave pública para JSON rápido
                result["json"] = result_json
            except Exception as e:
                print(f"[BacktestEngine] Error al serializar estadísticas JSON: {e}")

        if getattr(self.config, "plot_graph", False):
            try:
                from visualization.plot import BacktestPlotter

                BacktestPlotter().show(result)
            except Exception as e:
                print(f"[BacktestEngine] Error al graficar: {e}")

        return result

    def _setup_managers(self, symbol_points, symbols):
        em = EntryManager(
            self.config.initial_balance,
            strategies_params=self.config.strategies_params,
            symbol_points_mapping=symbol_points,
            commission_per_lot_side=self.config.commission_per_lot_side,
        )
        pm = em.position_manager
        pm.sym2idx = {sym: idx for idx, sym in enumerate(symbols)}
        # Alinear spread del EntryManager con el configurado
        em.spread_points = int(self.config.spread_points)
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
