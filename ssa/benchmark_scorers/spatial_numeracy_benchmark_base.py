"""
Spatial Numeracy Benchmark Scorer Base Implementation.

This module provides the base class for spatial and numeracy benchmark scorers,
containing all shared functionality that was previously duplicated between the two.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast

try:
    import mlflow
except ImportError:
    mlflow = None  # Handle environments without MLflow

from ssa.interfaces import BenchmarkScorer, ImageType
from ssa.prompts import Corpus, Prompt
from ssa.scorers.model_scorer import ModelScorer
from ssa.utils.logging import log
from ssa.utils.metrics import flatten_and_sanitize_metrics


class SpatialNumeracyBenchmarkBase(BenchmarkScorer, ABC):
    """
    Base class for spatial and numeracy benchmark scorers containing shared functionality.

    This class contains all the common logic that was previously duplicated between
    SpatialBenchmarkScorer and NumeracyBenchmarkScorer, including initialization,
    error handling, and aggregation patterns.
    """

    def __init__(
        self,
        model_scorer: ModelScorer,
        corpus: Optional[Corpus] = None,
        corpus_prompts: Optional[List[Prompt]] = None,
        scorer_key: str = "",
    ):
        """
        Initialize the base benchmark scorer.

        Args:
            model_scorer: The model scorer to use for evaluation
            corpus: Optional corpus for scoring
            corpus_prompts: Optional list of prompts for scoring
            scorer_key: The key identifier for this scorer type
        """
        super().__init__(
            scorer_key=scorer_key,
            model_scorer=model_scorer,
            corpus_prompts=corpus_prompts,
            corpus=corpus,
        )

    @abstractmethod
    def _initialize_aggregators_internal(
        self,
        corpus: Optional[Corpus] = None,
        corpus_prompts: Optional[List[Prompt]] = None,
    ) -> Dict[str, Any]:
        """
        Initialize aggregators specific to the scorer type.

        Args:
            corpus: Optional corpus for scoring
            corpus_prompts: Optional list of prompts for scoring

        Returns:
            Dictionary of aggregators for the scorer type
        """
        pass

    @abstractmethod
    def get_score_key(self) -> str:
        """
        Get the score key for this scorer type.

        Returns:
            The score key (e.g., "spatial_score", "numeracy_score")
        """
        pass

    @abstractmethod
    def get_aggregator_key(self) -> str:
        """
        Get the aggregator key for this scorer type.

        Returns:
            The aggregator key (e.g., "spatial_agg", "numeracy_agg")
        """
        pass

    @abstractmethod
    def get_running_metrics_key(self) -> str:
        """
        Get the running metrics key for this scorer type.

        Returns:
            The running metrics key (e.g., "spatial", "numeracy")
        """
        pass

    def score_prompt(
        self,
        prompt: Prompt,
        image: Optional[ImageType] = None,
        image_path: Optional[str] = None,
        prompt_result: Optional[Dict[str, Any]] = None,
        running_metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle scoring for a single prompt.

        Args:
            prompt: The prompt to score
            image: Optional image for scoring
            image_path: Optional path to image
            prompt_result: Optional existing prompt result dictionary
            running_metrics: Optional running metrics dictionary
            **kwargs: Additional keyword arguments

        Returns:
            Updated prompt result dictionary
        """
        if prompt_result is None:
            prompt_result = {}

        if image is None:
            log.error(
                f"Image is None for {self.scorer_key} scoring of prompt {prompt.id}. Cannot score."
            )
            prompt_result[self.scorer_key] = {
                self.get_score_key(): None,
                "non_conformity": None,
                "correctness": None,
                "error": "Image not provided",
            }
            return prompt_result

        model = cast(Any, self.model_scorer.model)
        ig_version = kwargs.get("ig_version")

        if ig_version is not None and hasattr(model, "ig_version"):
            model.ig_version = ig_version
        elif ig_version is not None:
            log.warning(
                f"Model {type(model).__name__} for {self.scorer_key} does not have ig_version attribute. Scoring may proceed without it."
            )

        try:
            # Model.evaluate returns score, non_conformity, correctness
            evaluation_result = model.evaluate(image, prompt.text)
            if not isinstance(evaluation_result, tuple) or len(evaluation_result) != 3:
                raise TypeError(
                    f"{self.scorer_key} model.evaluate did not return a tuple of 3 values."
                )
            score, non_conformity, correctness = evaluation_result

            log.debug(f"Bench {self.scorer_key} {self.get_score_key()}={score}")
            log.debug(f"Bench {self.scorer_key} non-conformity: {non_conformity}")
            log.debug(f"Bench {self.scorer_key} correctness: {correctness}")

            self.aggregators[self.get_aggregator_key()].add_datum(score)

            if running_metrics is not None:
                running_metrics[
                    f"running/{self.scorer_key.lower()}_{self.get_running_metrics_key()}"
                ] = self.aggregators[self.get_aggregator_key()].get_mean()

            prompt_result[self.scorer_key] = {
                self.get_score_key(): score,
                "non_conformity": non_conformity,
                "correctness": correctness,
            }
        except Exception as e:
            log.error(f"{self.scorer_key} scoring error: {e}", exc_info=True)
            prompt_result[self.scorer_key] = {
                self.get_score_key(): None,
                "error": str(e),
            }
        return prompt_result

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate metrics for the full run.

        Args:
            final_agg_dict: Dictionary to store aggregated metrics

        Returns:
            Updated aggregated metrics dictionary
        """
        if not self.aggregators:
            return final_agg_dict

        aggregator_key = self.get_aggregator_key()
        if (
            aggregator_key in self.aggregators
            and self.aggregators[aggregator_key].get_count() > 0
        ):
            final_agg_dict.update(self.aggregators[aggregator_key].get_aggregates())

        # Log aggregated metrics to MLflow
        if mlflow is not None:
            try:
                sanitized_metrics = flatten_and_sanitize_metrics(final_agg_dict)
                mlflow.log_metrics(sanitized_metrics)
            except Exception as e:
                # In test environments or when MLflow/AWS isn't configured,
                # we should continue without logging rather than failing
                log.warning(f"Could not log metrics to MLflow: {e}")

        return final_agg_dict
