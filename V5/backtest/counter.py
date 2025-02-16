# backtest/counter.py
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from typing import Optional

# Diccionario global para almacenar las barras de progreso activas (por worker)
_active_bars = {}


class ProgressCounter:
    def __init__(
        self,
        total: int,
        desc: str = "",
        position: int = 0,  # No se usa en Rich, se conserva por compatibilidad
        leave: bool = True,
        ncols: Optional[int] = 80,
    ):
        """
        Inicializa una barra de progreso usando Rich.

        :param total: Número total de pasos.
        :param desc: Descripción de la barra.
        :param position: Posición (no utilizada con Rich, pero se deja por compatibilidad).
        :param leave: Si se deja la barra en pantalla al finalizar.
                      Con Rich se usa el parámetro 'transient' (False = limpia al finalizar).
        :param ncols: Ancho de la barra.
        """
        self.total = total
        self.desc = desc
        self.leave = leave
        self.counter = 0

        # El parámetro 'transient' borra la barra al finalizar si es True.
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=ncols),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=not leave,  # Si leave es False, se limpia la barra
        )
        self.task_id = self.progress.add_task(desc, total=total)
        self.progress.start()

    def update(self, n: int = 1) -> None:
        """
        Incrementa el contador y actualiza la barra.

        :param n: Incremento (por defecto 1).
        """
        self.counter += n
        self.progress.update(self.task_id, advance=n)

    def close(self) -> None:
        """Cierra la barra de progreso (Rich se encarga de limpiar la salida si 'transient' es True)."""
        self.progress.stop()

    def __enter__(self) -> "ProgressCounter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_progress_counter(
    mode: str, total_steps: int, worker_id: int = 1
) -> ProgressCounter:
    """
    Crea y devuelve una barra de progreso.
    Si ya existe una barra para el worker indicado, se cierra y se reemplaza.
    """
    global _active_bars

    if mode == "single":
        return ProgressCounter(total=total_steps, desc="Backtest", leave=True)
    elif mode == "optimization":
        if worker_id in _active_bars:
            _active_bars[worker_id].close()
            del _active_bars[worker_id]
        bar = ProgressCounter(
            total=total_steps, desc=f"Nucleo {worker_id}", leave=False
        )
        _active_bars[worker_id] = bar
        return bar
    else:
        raise ValueError("Modo inválido. Use 'single' o 'optimization'.")
