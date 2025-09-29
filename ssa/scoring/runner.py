from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from ssa.scoring.interfaces import (
    AggregatedResults,
    BenchmarkScoreResult,
    ScorerInterface,
)
from ssa.utils.aggregator import Aggregator
from ssa.utils.logging import log


@dataclass
class PromptContext:
    """Context information for scoring a prompt."""

    prompt_id: str
    prompt_text: str
    prompt_obj: Any  # The original prompt object
    image: Optional[Any] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StandardizedBenchmarkRunner:
    """
    Refactored benchmark runner using immutable data flow and standardized interfaces.
    """

    def __init__(self, scorers: List[ScorerInterface]):
        """
        Initialize with a list of standardized scorers.

        Args:
            scorers: List of scorers implementing ScorerInterface
        """
        self.scorers = {scorer.scorer_name: scorer for scorer in scorers}
        self.results = AggregatedResults()

    def run_benchmark(
        self, prompts: List[PromptContext], progress_callback: Optional[Callable] = None
    ) -> Iterator[BenchmarkScoreResult]:
        """
        Run benchmark without mutation - yield results as they complete.

        Args:
            prompts: List of PromptContext objects
            progress_callback: Optional callback for progress updates

        Yields:
            BenchmarkScoreResult for each scorer/prompt combination
        """
        total_prompts = len(prompts)

        for idx, prompt_context in enumerate(prompts):
            log.info(
                f"Processing prompt {idx + 1}/{total_prompts}: {prompt_context.prompt_id}"
            )

            # Yield results from all scorers for this prompt
            yield from self._score_single_prompt(prompt_context)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(idx + 1, total_prompts, prompt_context.prompt_id)

    def _score_single_prompt(
        self, prompt_context: PromptContext
    ) -> Iterator[BenchmarkScoreResult]:
        """Score a single prompt with all configured scorers."""
        context = {
            "prompt_id": prompt_context.prompt_id,
            "prompt_obj": prompt_context.prompt_obj,
            "image": prompt_context.image,
            "image_path": prompt_context.image_path,
            **prompt_context.metadata,
        }

        for scorer_name, scorer in self.scorers.items():
            try:
                result = scorer.score_prompt(
                    prompt=prompt_context.prompt_text,
                    response=prompt_context.image,
                    context=context,
                )

                log.debug(
                    f"Scored {prompt_context.prompt_id} with {scorer_name} "
                    f"in {result.execution_time:.3f}s, success={result.success}"
                )

                self.results.add_result(result)
                yield result

            except Exception as e:
                log.error(
                    f"Error scoring {prompt_context.prompt_id} with {scorer_name}: {e}",
                    exc_info=True,
                )

                error_result = BenchmarkScoreResult(
                    prompt_id=prompt_context.prompt_id,
                    scorer_name=scorer_name,
                    results={},
                    execution_time=0.0,
                    success=False,
                    error_message=str(e),
                )

                self.results.add_result(error_result)
                yield error_result

    def get_aggregated_results(self) -> AggregatedResults:
        """Get the aggregated results object."""
        return self.results

    def compute_final_aggregates(self) -> Dict[str, Any]:
        """
        Compute final aggregated metrics across all scorers.
        Returns aggregated metrics compatible with legacy format.
        Includes category-specific aggregation for supported scorers.
        """
        final_aggregates = {}

        # Aggregate metrics for each scorer
        for scorer_name in self.scorers.keys():
            # Overall aggregation (existing behavior)
            scorer_metrics = self.results.get_metrics_for_scorer(scorer_name)

            for metric_name, values in scorer_metrics.items():
                if not values:
                    continue

                # Create aggregator for this metric
                agg_key = f"{scorer_name}/{metric_name}"
                aggregator = Aggregator(agg_key)

                # Add all values
                for value in values:
                    if isinstance(value, (int, float)):
                        aggregator.add_datum(value)

                # Get aggregated statistics
                if aggregator.get_count() > 0:
                    final_aggregates.update(aggregator.get_aggregates())

            # Category-specific aggregation
            category_aggregators = {}

            for result in self.results.results:
                if result.scorer_name != scorer_name or not result.success:
                    continue

                # Extract categories from the result's raw_data
                categories = []
                if hasattr(result, "raw_data") and isinstance(result.raw_data, dict):
                    categories = result.raw_data.get("categories", [])

                # If no categories found, skip category-specific aggregation
                if not categories:
                    continue

                # Add metrics to category-specific aggregators
                for category in categories:
                    if category not in category_aggregators:
                        category_aggregators[category] = {}

                    for metric_name, score_result in result.results.items():
                        if metric_name not in category_aggregators[category]:
                            category_aggregators[category][metric_name] = Aggregator(
                                f"{scorer_name}/{category}/{metric_name}"
                            )

                        if isinstance(score_result.score, (int, float)):
                            category_aggregators[category][metric_name].add_datum(
                                score_result.score
                            )

            # Add category aggregation results to final aggregates
            for category, metric_aggs in category_aggregators.items():
                for metric_name, aggregator in metric_aggs.items():
                    if aggregator.get_count() > 0:
                        final_aggregates.update(aggregator.get_aggregates())

        # Add error counts
        final_aggregates.update(self.results.get_aggregated_metrics())

        return final_aggregates


def create_prompt_context_from_legacy(
    prompt_obj: Any, image: Optional[Any] = None, image_path: Optional[str] = None
) -> PromptContext:
    """
    Helper function to create PromptContext from legacy prompt objects.
    """
    return PromptContext(
        prompt_id=getattr(prompt_obj, "id", "unknown"),
        prompt_text=getattr(prompt_obj, "text", ""),
        prompt_obj=prompt_obj,
        image=image,
        image_path=image_path,
        metadata={
            "categories": getattr(prompt_obj, "categories", []),
            "extensions": getattr(prompt_obj, "extensions", []),
            "extras": getattr(prompt_obj, "extras", {}),
            "criteria": getattr(prompt_obj, "criteria", None),
            "num_extensions": getattr(prompt_obj, "num_extensions", None),
        },
    )
