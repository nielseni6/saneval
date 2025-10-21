"""
Standardized scorer adapters for specific scoring methods.
"""

import time
from typing import Any, Dict, List, Optional

from ssa.scoring.interfaces import BenchmarkScoreResult, ScoreResult, ScorerInterface
from ssa.utils.logging import log


class OdAttrBindingScorerAdapter(ScorerInterface):
    """Adapter for Object Detection-Based Attribute Binding scorer."""

    def __init__(
        self,
        od_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = od_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["binding_accuracy", "detection_count", "attribute_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using OD Attribute Binding with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image_path = context.get("image_path")
        prompt_obj = context.get("prompt_obj")

        try:
            if image_path is None:
                raise ValueError(
                    "Image path not provided for OD Attribute Binding scoring"
                )

            # Get criteria questions if available
            criteria_questions = None
            if prompt_obj and hasattr(prompt_obj, "criteria") and prompt_obj.criteria:
                criteria_questions = prompt_obj.criteria

            # Perform object detection and attribute binding analysis
            binding_score = self.model_scorer.model.evaluate(
                image_path, prompt, criteria_questions
            )

            results = {
                "binding_accuracy": ScoreResult(
                    score=binding_score,
                    metadata={"prompt": prompt, "image_path": image_path},
                ),
                "detection_count": ScoreResult(
                    score=len(
                        getattr(self.model_scorer.model, "attribute_objects", [])
                    ),
                    metadata={
                        "detected_objects": getattr(
                            self.model_scorer.model, "attribute_objects", []
                        )
                    },
                ),
                "attribute_score": ScoreResult(
                    score=binding_score,
                    metadata={"score": binding_score},
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "score": binding_score,
                    "categories": context.get("categories", []),
                },
            )

        except Exception as e:
            log.error(
                f"OD Attribute Binding scoring error for {prompt_id}: {e}",
                exc_info=True,
            )
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "od-attr-binding"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying OD model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying OD Attribute Binding scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class SpatialScorerAdapter(ScorerInterface):
    """Adapter for Spatial reasoning scorer."""

    def __init__(
        self,
        spatial_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
        benchmark_scorer=None,
    ):
        self.model_scorer = spatial_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["spatial_score"]

        # Use dependency injection for benchmark scorer, with default fallback
        if benchmark_scorer is not None:
            self.benchmark_scorer = benchmark_scorer
        else:
            # Default: create benchmark scorer instance for proper aggregation
            from ssa.benchmark_scorers.spatial_scorer import SpatialBenchmarkScorer

            self.benchmark_scorer = SpatialBenchmarkScorer(
                model_scorer=spatial_model_scorer,
                corpus=corpus,
                corpus_prompts=corpus_prompts,
                scorer_key="spatial",
            )

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Spatial scorer with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for Spatial scoring")

            # Evaluate with Spatial model - returns (spatial_score, non_conformity, correctness)
            spatial_score, non_conformity, correctness = (
                self.model_scorer.model.evaluate(image, prompt)
            )

            # Also call the benchmark scorer to populate its aggregators
            from ssa.prompts import Prompt

            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                prompt_obj = Prompt(text=prompt, id=prompt_id)

            self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result={},
                running_metrics={},
            )

            results = {
                "spatial_score": ScoreResult(
                    score=spatial_score,
                    metadata={
                        "prompt": prompt,
                        "non_conformity": non_conformity,
                        "correctness": correctness,
                    },
                )
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "spatial_score": spatial_score,
                    "non_conformity": non_conformity,
                    "correctness": correctness,
                },
            )

        except Exception as e:
            log.error(f"Spatial scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "spatial"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Spatial model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Spatial benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


class NumeracyScorerAdapter(ScorerInterface):
    """Adapter for Numeracy counting scorer."""

    def __init__(
        self,
        numeracy_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
        benchmark_scorer=None,
    ):
        self.model_scorer = numeracy_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["numeracy_score"]

        # Use dependency injection for benchmark scorer, with default fallback
        if benchmark_scorer is not None:
            self.benchmark_scorer = benchmark_scorer
        else:
            # Default: create benchmark scorer instance for proper aggregation
            from ssa.benchmark_scorers.numeracy_scorer import NumeracyBenchmarkScorer

            self.benchmark_scorer = NumeracyBenchmarkScorer(
                model_scorer=numeracy_model_scorer,
                corpus=corpus,
                corpus_prompts=corpus_prompts,
                scorer_key="numeracy",
            )

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Numeracy scorer with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for Numeracy scoring")

            # Evaluate with Numeracy model - returns (numeracy_score, non_conformity, correctness)
            numeracy_score, non_conformity, correctness = (
                self.model_scorer.model.evaluate(image, prompt)
            )

            # Also call the benchmark scorer to populate its aggregators
            from ssa.prompts import Prompt

            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                prompt_obj = Prompt(text=prompt, id=prompt_id)

            self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result={},
                running_metrics={},
            )

            results = {
                "numeracy_score": ScoreResult(
                    score=numeracy_score,
                    metadata={
                        "prompt": prompt,
                        "non_conformity": non_conformity,
                        "correctness": correctness,
                    },
                )
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "numeracy_score": numeracy_score,
                    "non_conformity": non_conformity,
                    "correctness": correctness,
                },
            )

        except Exception as e:
            log.error(f"Numeracy scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "numeracy"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Numeracy model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Numeracy benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


def create_scorer_adapter(
    scorer_key: str,
    model_scorer,
    config: Optional[Dict[str, Any]] = None,
    corpus=None,
    corpus_prompts=None,
) -> ScorerInterface:
    """
    Factory function to create the appropriate scorer adapter.

    Args:
        scorer_key: The scoring method key (e.g., 'spatial', 'numeracy', 'od-attr-binding')
        model_scorer: The model scorer instance
        config: Optional configuration
        corpus: Optional corpus for adapters that need it
        corpus_prompts: Optional corpus prompts for adapters that need it

    Returns:
        ScorerInterface instance
    """
    adapter_map = {
        "spatial": SpatialScorerAdapter,
        "numeracy": NumeracyScorerAdapter,
        "od-attr-binding": OdAttrBindingScorerAdapter,
    }

    adapter_class = adapter_map.get(scorer_key)
    if adapter_class is None:
        raise ValueError(f"No adapter available for scorer: {scorer_key}")

    return adapter_class(
        model_scorer, config, corpus=corpus, corpus_prompts=corpus_prompts
    )
