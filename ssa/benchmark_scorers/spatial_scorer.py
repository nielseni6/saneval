"""
Spatial Benchmark Scorer Implementation.
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
from ssa.scoring import SPATIAL
from ssa.utils.aggregator import Aggregator


class SpatialBenchmarkScorer(SpatialNumeracyBenchmarkBase):
    """
    BenchmarkScorer implementation for Spatial scoring.
    """

    def __init__(
        self,
        model_scorer: ModelScorer,
        corpus: Optional[Corpus] = None,
        corpus_prompts: Optional[List[Prompt]] = None,
        scorer_key: str = SPATIAL,
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
        """Initializes aggregators for Spatial scoring."""
        return {
            "spatial_agg": Aggregator(f"{self.scorer_key}/spatial_score"),
        }

    def get_score_key(self) -> str:
        """Get the score key for spatial scoring."""
        return "spatial_score"

    def get_aggregator_key(self) -> str:
        """Get the aggregator key for spatial scoring."""
        return "spatial_agg"

    def get_running_metrics_key(self) -> str:
        """Get the running metrics key for spatial scoring."""
        return "spatial"
