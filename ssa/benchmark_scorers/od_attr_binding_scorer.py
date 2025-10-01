"""
OD_ATTR_BINDING Benchmark Scorer Implementation.
"""

from typing import Any, Dict, List, Optional, cast  # Added List

# MLflow import removed - logging now handled centrally by benchmark runner.
# Centralized logging prevents metric overwriting issues and ensures consistency across scorers.
from ssa.interfaces import BenchmarkScorer
from ssa.prompts import Corpus, Prompt  # Added Corpus, Prompt
from ssa.scorers.model_scorer import ModelScorer
from ssa.scoring import OD_ATTR_BINDING
from ssa.utils.aggregator import Aggregator
from ssa.utils.logging import log

# Related utility imports also removed as part of centralized logging approach

OdAttrBindingModelType = (
    Any  # Placeholder type for the model used by OD_ATTR_BINDING scorer
)


class OdAttrBindingBenchmarkScorer(BenchmarkScorer):
    """
    BenchmarkScorer implementation for OD_ATTR_BINDING (Object Detection Attribute Binding) scoring.
    """

    def __init__(
        self,
        model_scorer: ModelScorer,
        corpus: Optional[Corpus] = None,  # Added corpus
        corpus_prompts: Optional[List[Prompt]] = None,  # Added corpus_prompts
        scorer_key: str = OD_ATTR_BINDING,  # Added scorer_key
    ):
        super().__init__(  # Pass all args to super
            scorer_key=scorer_key,
            model_scorer=model_scorer,
            corpus_prompts=corpus_prompts,
            corpus=corpus,
        )

    def _initialize_aggregators_internal(
        self,
        corpus: Optional[Corpus] = None,  # Added corpus
        corpus_prompts: Optional[List[Prompt]] = None,  # Added corpus_prompts
    ) -> Dict[str, Any]:
        """Initializes aggregators for OD_ATTR_BINDING scoring."""
        # Define the detailed metrics that ODBAB provides
        metric_names = ["binding_accuracy", "detection_count", "attribute_score"]

        aggs = {
            "overall_aggs": {},
            "category_aggs": {},
        }

        # Create overall aggregators for each metric
        for metric in metric_names:
            aggs["overall_aggs"][metric] = Aggregator(f"{self.scorer_key}/{metric}")

        # Get all categories from the corpus prompts
        prompts = corpus_prompts or (
            corpus.prompts if corpus and corpus.prompts else []
        )
        all_categories = set()
        for p in prompts:
            # Check both categories field and tag from extras
            categories = getattr(p, "categories", [])
            if not categories and hasattr(p, "extras") and "tag" in p.extras:
                # If no categories but has tag, use tag as category
                categories = [p.extras["tag"]]
            for cat in categories:
                all_categories.add(cat)

        # If no categories found, add a default one
        if not all_categories:
            all_categories = {"unknown_category"}

        # Create category-specific aggregators for each metric
        for category in all_categories:
            aggs["category_aggs"][category] = {}
            for metric in metric_names:
                aggs["category_aggs"][category][metric] = Aggregator(
                    f"{self.scorer_key}/{category}/{metric}"
                )

        return aggs

    def score_prompt(
        self,
        prompt: Prompt,
        image: Optional[Any] = None,  # Added image parameter
        image_path: Optional[str] = None,  # Added image_path
        prompt_result: Optional[Dict[str, Any]] = None,
        running_metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:  # Return type changed to Optional[Dict[str, Any]]
        """Handles OD_ATTR_BINDING scoring for a single prompt."""
        if prompt_result is None:
            prompt_result = {}
        if image_path is None:  # Added check for image
            log.error(
                f"Image is None for {self.scorer_key} scoring of prompt {prompt.id}. Cannot score."
            )
            prompt_result[self.scorer_key] = {
                "score": None,
                "error": "Image not provided",
            }
            return prompt_result

        od_attr_model = cast(
            OdAttrBindingModelType, self.model_scorer.model
        )  # Cast model
        ig_version = kwargs.get("ig_version")

        if ig_version is not None:
            if hasattr(od_attr_model, "ig_version"):
                od_attr_model.ig_version = ig_version
            else:
                log.warning(
                    f"Model {type(od_attr_model).__name__} for {self.scorer_key} does not have ig_version attribute. Scoring may proceed without it."
                )
        try:
            # Handle criteria format - convert list to string if needed
            criteria = prompt.criteria
            if isinstance(criteria, list):
                # Convert list to string - join multiple criteria or take first one
                criteria = criteria[0] if criteria else None

            # Get detailed evaluation result from model
            od_attr_result = od_attr_model.evaluate(image_path, prompt.text, criteria)

            # Handle both single score and detailed result formats
            if isinstance(od_attr_result, dict):
                # Detailed result with multiple metrics
                binding_accuracy = od_attr_result.get(
                    "binding_accuracy", od_attr_result.get("score", 0)
                )
                detection_count = od_attr_result.get("detection_count", 0)
                attribute_score = od_attr_result.get(
                    "attribute_score", binding_accuracy
                )
            else:
                # Single score result - use for all metrics
                binding_accuracy = od_attr_result
                # Try to get detection count from model attribute objects
                detection_count = len(getattr(od_attr_model, "attribute_objects", []))
                attribute_score = od_attr_result

            detailed_metrics = {
                "binding_accuracy": binding_accuracy,
                "detection_count": detection_count,
                "attribute_score": attribute_score,
            }

            log.info(f"{self.scorer_key} Metrics: {detailed_metrics}")

            # Add to overall aggregators
            for metric_name, metric_value in detailed_metrics.items():
                if metric_name in self.aggregators["overall_aggs"]:
                    self.aggregators["overall_aggs"][metric_name].add_datum(
                        metric_value
                    )

            # Add to category-specific aggregators
            categories = getattr(prompt, "categories", [])
            if not categories and hasattr(prompt, "extras") and "tag" in prompt.extras:
                # If no categories but has tag, use tag as category
                categories = [prompt.extras["tag"]]
            if not categories:
                categories = ["unknown_category"]

            for category in categories:
                if category in self.aggregators["category_aggs"]:
                    for metric_name, metric_value in detailed_metrics.items():
                        if metric_name in self.aggregators["category_aggs"][category]:
                            self.aggregators["category_aggs"][category][
                                metric_name
                            ].add_datum(metric_value)
                else:
                    log.warning(f"Category '{category}' not found in aggregators")

            # Store result with backward compatibility
            prompt_result[self.scorer_key] = {
                "score": binding_accuracy,
                **detailed_metrics,
            }

            # Update running metrics
            if running_metrics is not None and "overall_aggs" in self.aggregators:
                if "binding_accuracy" in self.aggregators["overall_aggs"]:
                    running_metrics[f"running/{self.scorer_key.lower()}_score"] = (
                        self.aggregators["overall_aggs"]["binding_accuracy"].get_mean()
                    )
        except Exception as e:
            log.error(f"{self.scorer_key} scoring error: {e}", exc_info=True)
            prompt_result[self.scorer_key] = {"score": None, "error": repr(e)}
        return prompt_result  # Return updated prompt_result

    def aggregate_metrics(
        self, final_agg_dict: Dict[str, Any]
    ) -> Dict[str, Any]:  # Return type changed to Dict[str, Any]
        """Aggregates OD_ATTR_BINDING metrics for the full run."""
        # Add overall metric aggregations
        for metric_name, agg in self.aggregators["overall_aggs"].items():
            if agg.get_count() > 0:
                final_agg_dict.update(agg.get_aggregates())

        # Add category-specific metric aggregations
        for category, metric_aggs in self.aggregators["category_aggs"].items():
            for metric_name, agg in metric_aggs.items():
                if agg.get_count() > 0:
                    final_agg_dict.update(agg.get_aggregates())

        # Note: MLflow logging is now handled centrally by the benchmark runner
        # to prevent summary metrics from being overwritten when logged from multiple sources

        return final_agg_dict  # Return updated final_agg_dict
