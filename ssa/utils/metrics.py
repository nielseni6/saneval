from typing import Dict


def flatten_and_sanitize_metrics(metrics: Dict) -> Dict:
    """
    Flattens a nested dictionary of metrics and sanitizes the values for metrics logging.
    """
    flattened_metrics = {
        f"{key}.{subkey}": subval
        for key, val in metrics.items()
        if isinstance(val, dict)
        for subkey, subval in val.items()
    }
    flattened_metrics.update(
        {key: val for key, val in metrics.items() if not isinstance(val, dict)}
    )

    sanitized_metrics = {
        k: float(v) if isinstance(v, (int, float)) else float("nan")
        for k, v in flattened_metrics.items()
        if isinstance(v, (int, float))
    }
    return sanitized_metrics
