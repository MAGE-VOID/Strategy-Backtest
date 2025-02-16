from time import sleep
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import RLock
from tqdm import tqdm
from typing import Optional

# Configuramos un lock global para evitar conflictos en la salida (útil en multiprocesamiento/hilos)
tqdm.set_lock(RLock())

# ------------------------------
# Definición de ProgressCounter
# ------------------------------
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

# ------------------------------------------
# Función para crear una barra configurada
# ------------------------------------------
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
        # Cada barra se muestra en una línea distinta (posición = worker_id - 1) y se borra al finalizar.
        return ProgressCounter(
            total=total_steps,
            desc=f"Nucleo {worker_id}",
            position=worker_id - 1,
            leave=False,
        )
    else:
        raise ValueError("Modo inválido. Use 'single' o 'optimization'.")

# -----------------------------------------------------
# Función similar al ejemplo con trange para testing
# -----------------------------------------------------
def progresser(n: int) -> str:
    """
    Simula un proceso de trabajo en modo optimization.
    Cada proceso (núcleo) muestra su propia barra en una línea distinta.
    
    Aquí, el intervalo de espera se ajusta para que cada worker termine en un tiempo distinto.
    Por ejemplo, worker 0 tendrá un intervalo de 0.001 s, worker 1 de 0.002 s, etc.
    """
    # Asignamos un intervalo creciente según el ID del worker.
    interval = 0.005 * (n + 1)
    total = 5000
    text = f"#{n}, est. {interval * total:<04.2f}s"
    # Creamos la barra en modo "optimization", asignándole una posición única.
    counter = create_progress_counter("optimization", total_steps=total, worker_id=n + 1)
    for _ in range(total):
        sleep(interval)
        counter.update(1)
    counter.close()
    if n == 6:
        tqdm.write("n == 6 completed.")
        tqdm.write("`tqdm.write()` is thread-safe in py3!")
    return f"Worker {n} finished."

# -------------------------------------------
# Función para probar el modo single (un solo contador)
# -------------------------------------------
def test_single_mode():
    total = 100
    with create_progress_counter("single", total_steps=total) as counter:
        for _ in range(total):
            sleep(0.01)
            counter.update(1)
    tqdm.write("Single mode test completed.")

# -----------------------
# Código de ejecución
# -----------------------
if __name__ == '__main__':
    # Ejecutamos la prueba en modo single
    test_single_mode()

    # Ejecutamos la prueba en modo optimization con varios "núcleos"
    workers = list(range(3))  # Simula 9 procesos o núcleos (IDs: 0 a 8)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(progresser, workers))
    for res in results:
        tqdm.write(res)
