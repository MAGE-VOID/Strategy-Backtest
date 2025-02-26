# backtest/algorithm/search_algorithms.py
import queue
from typing import Any, Dict, List, Optional

class DynamicSearchAlgorithm:
    """
    Ejemplo de clase base para un algoritmo de búsqueda dinámico.
    Puede ser usado para un grid search, random search, o algo más avanzado.
    """

    def __init__(self, all_combinations: List[Dict[str, Any]]):
        """
        all_combinations: lista inicial de parámetros a probar.
        En un grid search clásico, aquí tendrías todas las combinaciones;
        en un random search, podrían ser muestras aleatorias iniciales.
        """
        self._pending = queue.Queue()
        for combo in all_combinations:
            self._pending.put(combo)

        # Aquí podrías guardar resultados y lógica de refinamiento
        self.results = []

    def get_next_params(self) -> Optional[Dict[str, Any]]:
        """
        Devuelve el siguiente conjunto de parámetros a optimizar (o None si ya no hay más).
        """
        if self._pending.empty():
            return None
        return self._pending.get_nowait()

    def report_result(self, params: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """
        Recibe la estadística (stats) para los 'params' que acaban de terminar el backtest.
        Aquí se podrían generar nuevas combinaciones en base a stats,
        o simplemente registrar el resultado y no modificar la lista pendiente.
        """
        self.results.append((params, stats))
        # EJEMPLO (grid search puro): no se hace nada adicional.
        # EJEMPLO (random/bayesian): podrías crear nuevas combinaciones y meterlas a _pending
        # en función de 'stats'.
