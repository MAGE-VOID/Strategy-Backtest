# backtest/utils/barprogress_multi.py
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import time


class MultiProgress:
    def __init__(self, total_tasks: int):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=1,
        )
        self.total_tasks = total_tasks
        self.overall_task_id = self.progress.add_task(
            "Overall Progress", total=total_tasks
        )
        # Mapping de nucleo_id -> id de tarea en rich
        self.worker_tasks = {}

    def add_worker(self, worker_id: int, description: str, total: int):
        # Se agrega o reinicia la barra para el nucleo indicado
        task_id = self.progress.add_task(description, total=total, visible=True)
        self.worker_tasks[worker_id] = task_id

    def update_worker(self, worker_id: int, completed: int):
        if worker_id in self.worker_tasks:
            self.progress.update(self.worker_tasks[worker_id], completed=completed)

    def update_overall(self, completed_workers: int):
        self.progress.update(self.overall_task_id, completed=completed_workers)

    def start(self):
        self.progress.start()

    def stop(self):
        self.progress.stop()
