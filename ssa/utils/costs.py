"""
Cost tracking utilities.
Stub implementation - needs to be replaced with actual cost tracking.
"""
from typing import Dict


class CostTracker:
    """Track costs for LLM/VLM API calls."""

    def __init__(self):
        self._total_cost = 0.0
        self._costs_by_model: Dict[str, float] = {}

    def add_cost(self, cost: float, model: str = "unknown"):
        """Add a cost to the tracker."""
        self._total_cost += cost
        if model not in self._costs_by_model:
            self._costs_by_model[model] = 0.0
        self._costs_by_model[model] += cost

    def total_cost(self) -> float:
        """Get total cost."""
        return self._total_cost

    def get_costs_by_model(self) -> Dict[str, float]:
        """Get costs broken down by model."""
        return self._costs_by_model.copy()

    def reset(self):
        """Reset all costs."""
        self._total_cost = 0.0
        self._costs_by_model = {}


# Global cost tracker instance
global_cost_tracker = CostTracker()
