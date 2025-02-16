# backtest/optimizer.py
import copy
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest.engine import BacktestEngine


def generate_param_combinations(opt_params: dict):
    """
    Genera todas las combinaciones de parámetros a partir de un diccionario de optimización.
    Los valores pueden definirse como una tupla (inicio, paso, fin) o como una lista de valores.
    """
    keys = list(opt_params.keys())
    values_list = []
    for key in keys:
        val = opt_params[key]
        if isinstance(val, tuple) and len(val) == 3:
            start, step, end = val
            if (
                isinstance(start, int)
                and isinstance(step, int)
                and isinstance(end, int)
            ):
                values = list(range(start, end + 1, step))
            else:
                values = list(np.arange(start, end + step / 2, step))
            values_list.append(values)
        elif isinstance(val, list):
            values_list.append(val)
        else:
            values_list.append([val])
    combinations = []
    for combo in itertools.product(*values_list):
        combination = {key: value for key, value in zip(keys, combo)}
        combinations.append(combination)
    return combinations


def run_trial(trial_params, base_config, input_data):
    """
    Ejecuta un backtest con una combinación específica de parámetros.
    Se crea una copia de la configuración base y se reemplaza 'optimization_params'.
    NOTA: 'input_data' es el DataFrame obtenido en el proceso principal.
    """
    config_trial = copy.deepcopy(base_config)
    config_trial.optimization_params = trial_params
    engine = BacktestEngine(config_trial)
    result = engine.run_backtest(input_data)
    result["optimization_params"] = trial_params
    return result


def run_optimization(base_config, input_data, max_workers=None):
    """
    Ejecuta la optimización de parámetros en paralelo.
    Se generan todas las combinaciones y se ejecuta un backtest para cada una en paralelo.
    Se retorna el mejor resultado (según 'net_profit') y la lista completa de resultados.
    """
    opt_params = getattr(base_config, "optimization_params", None)
    if not opt_params:
        opt_params = base_config.strategy_signal_class.default_optimization_params

    param_combinations = generate_param_combinations(opt_params)
    print(f"Total combinaciones a probar: {len(param_combinations)}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_trial, params, base_config, input_data)
            for params in param_combinations
        ]
        # Se recorre as_completed sin tqdm aquí
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                net_profit = result["statistics"].get("net_profit", 0)
                print(
                    f"Combinación: {result['optimization_params']} -> Net Profit: {net_profit}"
                )
            except Exception as e:
                print(f"Error en una tarea de optimización: {e}")

    best_result = (
        max(results, key=lambda r: r["statistics"].get("net_profit", float("-inf")))
        if results
        else None
    )
    return best_result, results
