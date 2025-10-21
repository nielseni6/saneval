import json
from statistics import mean, median
from typing import List, Optional

import mlflow.artifacts
import pandas as pd

from ssa.utils.base import create_temp_download_directory, get_temp_file
from ssa.utils.logging import log
from ssa.utils.moe_sample_size import needed_samples, standard_dev

# Constants for statistical calculations
CONFIDENCE_INTERVAL_MULTIPLIER = 1.96  # 95% confidence interval (z-score)
STRICT_THRESHOLD = 1.0  # Threshold for strict pass/fail


def percentile(sorted_nums: List[float], perc: float) -> Optional[float]:
    if not sorted_nums:
        return None
    n = len(sorted_nums)
    perc_index = min(int(n * perc), n - 1)
    return sorted_nums[perc_index]


class Aggregator:
    """Very simple aggregator"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.data: List[float] = []

    def add_datum(self, datum: float) -> None:
        self.data.append(datum)

    def add_data(self, data: List[float]) -> None:
        self.data.extend(data)

    def get_mean(self) -> float:
        return mean(self.data)

    def get_sum(self) -> float:
        return sum(self.data)

    def get_count(self) -> int:
        return len(self.data)

    def get_aggregates(self, keys: Optional[List[str]] = None, with_strict: bool = False, prefix: Optional[str] = None) -> dict:

        log.debug(f"get_aggregates for {self.name!r}, count={len(self.data)}.")

        numbers = self.data
        sorted_nums = sorted(numbers)

        current_prefix = prefix if prefix is not None else self.name

        # if there were no datums report only the count
        if not numbers:
            return {f"{current_prefix}.count": 0}

        # Get needed samples data and write to log
        moe, n2 = needed_samples(numbers)

        if moe == -1:
            log.warning(f"Too few samples to compute CI (need ≥2, got {len(numbers)}).")

        if n2 > 0:
            log.warning(
                f"Error margin is high: {moe:.4f}, consider increasing the minimum number of prompts to {n2}. Current number of prompts: {len(numbers)}."
            )

        if n2 == 0:
            log.info(f"Error margin is acceptable: {moe:.4f}.")

        # compute once upfront
        m = mean(numbers)
        std = standard_dev(numbers)
        err = CONFIDENCE_INTERVAL_MULTIPLIER * (std / (len(numbers) ** 0.5))

        res = {
            f"{current_prefix}.{k}": v
            for k, v in {
                "p10": percentile(sorted_nums, 0.1),
                "mean": m,  # use cached mean
                "median": median(numbers),
                "p90": percentile(sorted_nums, 0.9),
                "count": len(numbers),
                "stdev": std,  # use cached std
                "ci95_lower": m - err,  # use cached mean and error margin
                "ci95_upper": m + err,
            }.items()
            if (not keys or k in keys)
        }

        if with_strict:
            # strict is percentage that were exactly at the threshold
            res[f"{current_prefix}.strict"] = float(numbers.count(STRICT_THRESHOLD)) / len(numbers)

        return res


def _combine_results_json(experiment_name: str, status_info: pd.DataFrame) -> str:
    """Combine results from finished tasks into a single JSON file."""
    results = []

    for _, task_row in status_info.iterrows():
        if str(task_row["status"]) == "FINISHED":
            task_results = _load_task_results(experiment_name, str(task_row["run_id"]))
            results.extend(task_results)

    output_file_path = get_temp_file("aggregated_run_results", ".json")

    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    return str(output_file_path)


def _load_task_results(experiment_name: str, run_id: str) -> List:
    """Load results from a specific task run."""
    temp_download_path = create_temp_download_directory(experiment_name, run_id)

    try:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="run_result.json", dst_path=temp_download_path
        )

        result_file_path = temp_download_path / "run_result.json"

        with open(result_file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not isinstance(results, list):
            log.error(
                f"Results for run {run_id} are not in expected list format. "
                "Skipping this run."
            )
            return []

        return results

    except FileNotFoundError as e:
        log.error(f"Results file not found for run {run_id}: {e}")
        return []
    except json.JSONDecodeError as e:
        log.error(f"Could not decode JSON for run {run_id}: {e}")
        return []
    except Exception as e:
        log.error(f"Error loading results for run {run_id}: {e}")
        return []
