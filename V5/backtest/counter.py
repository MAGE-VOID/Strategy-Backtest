from tqdm import tqdm
from typing import Optional
from multiprocessing import RLock

tqdm.set_lock(RLock())


class ProgressCounter:
    def __init__(
        self,
        total: int,
        desc: str = "",
        position: int = 0,
        leave: bool = True,
        ncols: Optional[int] = 80,
    ):
        """
        Inicializa una barra de progreso usando tqdm.

        :param total: Número total de pasos.
        :param desc: Descripción de la barra.
        :param position: Posición de la barra en el terminal.
                         En barras anidadas o multiprocesamiento, la posición se controla manualmente.
        :param leave: Si se deja la barra al finalizar.
        :param ncols: Número fijo de columnas para la barra.
        """
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave
        self.counter = 0
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit="step",
            ncols=ncols,
            ascii=True,
            position=position,
            leave=leave,
        )

    def update(self, n: int = 1) -> None:
        """
        Incrementa el contador y actualiza la barra.

        :param n: Incremento (por defecto 1).
        """
        self.counter += n
        self.pbar.update(n)

    def close(self) -> None:
        """Cierra la barra de progreso sin limpiar la salida (delega en tqdm)."""
        self.pbar.close()

    def __enter__(self) -> "ProgressCounter":
        """Permite usar la barra en un bloque 'with'."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Se asegura de cerrar la barra al salir del bloque 'with'."""
        self.close()


def create_progress_counter(
    mode: str, total_steps: int, worker_id: int = 1
) -> ProgressCounter:
    """
    Crea y devuelve una instancia de ProgressCounter configurada según el modo.

    :param mode: Modo de ejecución ("single" o "optimization").
    :param total_steps: Número total de pasos.
    :param worker_id: Identificador del núcleo (solo para "optimization").
    :return: Instancia de ProgressCounter.
    :raises ValueError: Si el modo no es "single" ni "optimization".
    """
    if mode == "single":
        return ProgressCounter(total=total_steps, desc="Backtest", leave=True)
    elif mode == "optimization":
        # Se asigna la posición (empezando en 0) para que cada barra aparezca en su línea,
        # y se utiliza leave=False para que la barra se borre al finalizar.
        return ProgressCounter(
            total=total_steps,
            desc=f"Nucleo {worker_id}",
            position=worker_id - 1,
            leave=False,
        )
    else:
        raise ValueError("Modo inválido. Use 'single' o 'optimization'.")
