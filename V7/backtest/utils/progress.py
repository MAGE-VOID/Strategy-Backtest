# backtest/utils/barprogress.py
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class BarProgress:
    def __init__(self, total: int, description: str = "[cyan]Running backtest"):
        """
        Inicializa la barra de progreso con el total de pasos y comienza a mostrarla.
        """
        self.progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("• {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total)

    def update(self, current: int):
        """
        Actualiza la barra de progreso con el valor actual.
        """
        self.progress.update(self.task_id, completed=current)

    def stop(self):
        """
        Detiene la barra de progreso.
        """
        self.progress.stop()
