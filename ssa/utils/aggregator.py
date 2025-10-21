from statistics import mean, median
from typing import List, Optional

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
