"""
Numeracy Benchmark Scorer Implementation.
"""

from typing import Any, Dict, List, Optional

try:
    import mlflow
except ImportError:
    mlflow = None  # Handle environments without MLflow

from ssa.benchmark_scorers.spatial_numeracy_benchmark_base import \
    SpatialNumeracyBenchmarkBase
from ssa.prompts import Corpus, Prompt
from ssa.scorers.model_scorer import ModelScorer
from ssa.scoring import NUMERACY
from ssa.utils.aggregator import Aggregator


class NumeracyBenchmarkScorer(SpatialNumeracyBenchmarkBase):
    """
    BenchmarkScorer implementation for Numeracy scoring.
    """

    def __init__(
        self,
        model_scorer: ModelScorer,
        corpus: Optional[Corpus] = None,
        corpus_prompts: Optional[List[Prompt]] = None,
        scorer_key: str = NUMERACY,
    ):
        super().__init__(
            model_scorer=model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key=scorer_key,
        )

    def _initialize_aggregators_internal(
        self,
        corpus: Optional[Corpus] = None,
        corpus_prompts: Optional[List[Prompt]] = None,
    ) -> Dict[str, Any]:
        """Initializes aggregators for Numeracy scoring."""
        return {
            "numeracy_agg": Aggregator(f"{self.scorer_key}/numeracy_score"),
        }

    def get_score_key(self) -> str:
        """Get the score key for numeracy scoring."""
        return "numeracy_score"

    def get_aggregator_key(self) -> str:
        """Get the aggregator key for numeracy scoring."""
        return "numeracy_agg"

    def get_running_metrics_key(self) -> str:
        """Get the running metrics key for numeracy scoring."""
        return "numeracy"
