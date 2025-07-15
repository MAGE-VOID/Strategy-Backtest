# backtest/algorithm/parameter_combinations.py
import numpy as np
from itertools import product
from typing import Dict, List, Any


def generate_combinations(optimization_params):
    param_values = []
    for i, (param_key, param_info) in enumerate(optimization_params.items(), start=1):
        param_type = param_info.get("type", "float")
        if param_type in ["int", "float"]:
            start = param_info.get("start")
            stop = param_info.get("stop")
            step = param_info.get("step")
            if start is None or stop is None or step is None:
                raise ValueError(
                    f"Parámetros de rango no definidos correctamente para {param_key}"
                )
            generated_values = np.arange(
                start, stop, step, dtype=float if param_type == "float" else int
            )
            print(f"Lista {i} para {param_key} (tipo {param_type}): {generated_values}")
            param_values.append(generated_values)
        elif param_type == "bool":
            generated_values = [True, False]
            print(f"Lista {i} para {param_key} (tipo bool): {generated_values}")
            param_values.append(generated_values)
        else:
            raise ValueError(
                f"Tipo de parámetro desconocido para {param_key}: {param_type}"
            )
    return list(product(*param_values))
