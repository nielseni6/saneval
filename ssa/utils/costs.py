"""Cost tracking utilities for model API calls.

This module provides simple cost tracking for monitoring API usage across
different model providers (Gemini, Bedrock, etc.). Costs are tracked globally
and can be reset or retrieved at any time.

Example usage:
    from ssa.utils.costs import report_cost, get_total_cost, reset_cost_tracking

    reset_cost_tracking()
    # ... make model calls that report costs
    report_cost(0.00123, "bedrock_llama")
    report_cost(0.00045, "gemini_flash")
    print(f"Total cost: ${get_total_cost():.6f}")
"""

from typing import Dict

from ssa.utils.logging import get_log

log = get_log(__file__)

# Global cost tracking
_total_cost = 0.0
_cost_breakdown = {}


def report_cost(cost: float, name: str = "") -> None:
    """
    Report an incurred cost from a model API call.

    Args:
        cost: The cost in dollars (e.g., 0.00123 for $0.00123)
        name: Optional name/identifier for this cost (e.g., "bedrock_llama_maverick")
    """
    global _total_cost, _cost_breakdown

    _total_cost += cost

    # Track breakdown by name if provided
    if name:
        if name not in _cost_breakdown:
            _cost_breakdown[name] = 0.0
        _cost_breakdown[name] += cost

    # Log the cost
    if name:
        log.info(f"Cost: ${cost:.6f} for {name}")
    else:
        log.info(f"Cost: ${cost:.6f}")


def get_total_cost() -> float:
    """
    Get the total cost tracked in this session.

    Returns:
        Total cost in dollars
    """
    return _total_cost


def get_cost_breakdown() -> Dict[str, float]:
    """
    Get a breakdown of costs by name.

    Returns:
        Dictionary mapping cost names to their total costs
    """
    return dict(_cost_breakdown)


def reset_cost_tracking() -> None:
    """
    Reset all cost tracking to zero.

    This clears both the total cost and the cost breakdown.
    """
    global _total_cost, _cost_breakdown
    _total_cost = 0.0
    _cost_breakdown = {}
    log.debug("Cost tracking reset")


class CostTracker:
    """
    Context manager for tracking costs within a specific scope.

    Example:
        with CostTracker() as tracker:
            # ... make model calls
            pass
        print(f"Scope cost: ${tracker.total:.6f}")
    """

    def __init__(self):
        self.total = 0.0
        self.breakdown = {}
        self._initial_total = 0.0
        self._initial_breakdown = {}

    def __enter__(self):
        """Record initial cost state."""
        self._initial_total = get_total_cost()
        self._initial_breakdown = get_cost_breakdown().copy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Calculate costs incurred during this context."""
        # Calculate total cost in this scope
        self.total = get_total_cost() - self._initial_total

        # Calculate breakdown for this scope
        current_breakdown = get_cost_breakdown()
        for name, cost in current_breakdown.items():
            initial_cost = self._initial_breakdown.get(name, 0.0)
            scope_cost = cost - initial_cost
            if scope_cost > 0:
                self.breakdown[name] = scope_cost

        log.debug(f"CostTracker: Scope total = ${self.total:.6f}")
        return False  # Don't suppress exceptions
