# backtest/config.py
from typing import Optional


class BacktestConfig:

    def __init__(
        self,
        initial_balance=1000,
        debug_mode="none",
        get_statistics: str = "off",  # opciones: 'off' | 'json' | 'print'
        plot_graph: bool = False,
        strategies_params=None,
        spread_points: int = 2,
        commission_per_lot_side: float = 3.5,
        # Compat: parámetro antiguo (deprecated). Si True y get_statistics es 'off', se mapea a 'print'.
        print_statistics: Optional[bool] = None,
    ):
        self.initial_balance = initial_balance
        self.debug_mode = debug_mode  # Opciones: "none", "final", "realtime"
        # Flags de salida/visualización
        mode = str(get_statistics or "off").lower()
        if mode not in ("off", "json", "print"):
            mode = "off"
        if print_statistics is True and mode == "off":
            mode = "print"
        self.get_statistics = mode
        self.plot_graph = bool(plot_graph)
        self.strategies_params = strategies_params or {}
        # Spread en puntos para ASK = BID + spread_points*point
        self.spread_points = int(spread_points)
        # Comisión por lado por lote (p. ej., $3.5 open y $3.5 close => $7 round-turn/lot)
        self.commission_per_lot_side = float(commission_per_lot_side)
