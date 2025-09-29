"""
Standardized interfaces for scoring infrastructure.
Provides consistent return types and interfaces for all scorers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScoreResult:
    """Standardized score result structure for individual metrics."""

    score: Union[float, int, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    errors: Optional[List[str]] = None


@dataclass
class BenchmarkScoreResult:
    """Result from scoring a single prompt with a specific scorer."""

    prompt_id: str
    scorer_name: str
    results: Dict[str, ScoreResult]  # metric_name -> result
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)  # For backward compatibility


class ScorerInterface(ABC):
    """Base interface all scorers must implement for standardized scoring."""

    @abstractmethod
    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """
        Score a single prompt-response pair.

        Args:
            prompt: The text prompt
            response: The response (image, text, etc.)
            context: Additional context including prompt_id, metadata, etc.

        Returns:
            BenchmarkScoreResult with standardized format
        """
        pass

    @property
    @abstractmethod
    def scorer_name(self) -> str:
        """Unique identifier for this scorer."""
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """List of metrics this scorer can produce."""
        pass

    def warmup(self) -> None:
        """
        Optional warmup method for scorers to prepare models, etc.
        Default implementation does nothing.
        """
        pass


class AggregatedResults:
    """Container for aggregated scoring results across all prompts."""

    def __init__(self):
        self.results: List[BenchmarkScoreResult] = []
        self.aggregated_metrics: Dict[str, Any] = {}
        self.error_counts: Dict[str, int] = {}

    def add_result(self, result: BenchmarkScoreResult) -> None:
        """Add a single result."""
        self.results.append(result)

        if not result.success:
            error_key = f"{result.scorer_name}_errors"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

    def get_metrics_for_scorer(self, scorer_name: str) -> Dict[str, List[Any]]:
        """Get all metric values for a specific scorer."""
        metrics = {}
        for result in self.results:
            if result.scorer_name == scorer_name and result.success:
                for metric_name, score_result in result.results.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(score_result.score)
        return metrics

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get final aggregated metrics including error counts."""
        return {**self.aggregated_metrics, **self.error_counts}
