"""
Margin of Error (MOE) and sample size calculation utilities.
Statistical utilities for determining sample sizes needed for confidence intervals.
"""

import math
from typing import List, Tuple


def standard_dev(numbers: List[float]) -> float:
    """
    Calculate standard deviation of a list of numbers.

    Args:
        numbers: List of numeric values

    Returns:
        Standard deviation
    """
    if not numbers or len(numbers) < 2:
        return 0.0

    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)
    return math.sqrt(variance)


def needed_samples(
    numbers: List[float], target_moe: float = 0.05, confidence_level: float = 0.95
) -> Tuple[float, int]:
    """
    Calculate margin of error and additional samples needed.

    Args:
        numbers: List of existing sample values
        target_moe: Target margin of error (default 0.05 = 5%)
        confidence_level: Confidence level (default 0.95 = 95%)

    Returns:
        Tuple of (current_moe, additional_samples_needed)
        - current_moe: Current margin of error, or -1 if too few samples
        - additional_samples_needed: Number of additional samples needed to reach target MOE,
          or 0 if target is already met, or -1 if calculation not possible
    """
    if not numbers or len(numbers) < 2:
        return -1, -1

    n = len(numbers)
    std_dev = standard_dev(numbers)

    # Z-score for confidence level (1.96 for 95% confidence)
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 2.576 for 99%

    # Current margin of error
    current_moe = z_score * (std_dev / math.sqrt(n))

    # If current MOE is already acceptable
    if current_moe <= target_moe:
        return current_moe, 0

    # Calculate needed sample size for target MOE
    # MOE = z * (std_dev / sqrt(n))
    # n = (z * std_dev / MOE)^2
    if target_moe > 0:
        needed_n = math.ceil((z_score * std_dev / target_moe) ** 2)
        additional_needed = max(0, needed_n - n)
    else:
        additional_needed = 0

    return current_moe, additional_needed
