# backtest/algorithm/metric_config.py


def select_best_result(results, metric_name="Win Rate [%]", mode="max"):
    """
    Selecciona el mejor resultado basado en la métrica indicada.

    Parámetros:
      - results: Lista de resultados, donde cada resultado es un diccionario que debe contener
                 al menos la clave "statistics" (un diccionario con las métricas).
      - metric_name: Nombre de la métrica que se utilizará para evaluar cada resultado.
                     Por defecto es "Win Rate [%]".
      - mode: 'max' para seleccionar el resultado con el valor máximo de la métrica,
              'min' para seleccionar el resultado con el valor mínimo.

    Retorna:
      - El resultado (diccionario) que tenga el mejor valor de la métrica según el modo.
    """
    if not results:
        raise ValueError("La lista de resultados está vacía.")

    best_result = None

    if mode == "max":
        best_metric = -float("inf")
        for res in results:
            stats = res.get("statistics", {})
            metric_value = stats.get(metric_name, 0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_result = res
    elif mode == "min":
        best_metric = float("inf")
        for res in results:
            stats = res.get("statistics", {})
            metric_value = stats.get(metric_name, 0)
            if metric_value < best_metric:
                best_metric = metric_value
                best_result = res
    else:
        raise ValueError("El modo debe ser 'max' o 'min'.")

    return best_result
