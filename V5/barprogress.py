# barprogress.py
from datetime import datetime
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

progress = None
task_id = None


def init_progress_bar(total: int):
    """
    Inicializa la barra de progreso con el total de pasos.
    La barra mostrará:
      - Descripción de la tarea.
      - Barra visual de progreso.
      - Porcentaje completado.
      - Cantidad de pasos completados y totales.
      - Tiempo transcurrido.
      - Tiempo estimado restante.
    """
    global progress, task_id
    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("• {task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    progress.start()
    task_id = progress.add_task("[cyan]Running backtest", total=total)


def update_progress_bar(current: int):
    """
    Actualiza la barra de progreso con el valor actual.
    Cuando se alcanza el total, asigna el tiempo final para que
    TimeElapsedColumn se detenga y se muestra el tiempo final transcurrido.
    """
    global progress, task_id
    if progress and task_id is not None:
        progress.update(task_id, completed=current)


def stop_progress_bar():
    """
    Detiene y reinicia las variables de la barra de progreso.
    """
    global progress, task_id
    if progress:
        progress.stop()
        progress = None
        task_id = None
