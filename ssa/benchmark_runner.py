"""
Core benchmarking logic for SSA benchmarking.
Contains core functions for running, configuring, and logging benchmarks.

Functions:
    - resolve_benchmark_config: Validates and constructs scorer objects for benchmarking.
    - get_config: Builds a configuration dictionary for the benchmark run.
    - simple_eval: Runs the main evaluation loop for a model/corpus/scorer set.
    - benchmark_model: Orchestrates a benchmark run and logs results.

Usage:
    Import these functions in your CLI or orchestration script to run benchmarks.
"""

import json
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ssa.exceptions import InvalidConfigurationError, MetricAggregationError
from ssa.scorers.model_scorer import ModelScorer
from ssa.utils.base import unique_id
from ssa.utils.logging import log
from ssa.utils.reasoning_trace import ReasoningTraceCollector, default_json_handler
from ssa.utils.system import default_context, get_subconfigs

if TYPE_CHECKING:
    from ssa.prompts import Prompt
    from ssa.scoring.interfaces import ScorerInterface
    from ssa.scoring.runner import StandardizedBenchmarkRunner


def resolve_benchmark_config(
    scoring_keys: List[str], bench_config_overrides: List[str]
) -> Dict[str, ModelScorer]:
    """
    Validates and constructs scorer objects for the given scoring keys and config overrides.
    Raises ValueError if an override does not match an enabled scoring key.
    Returns a dict of {scoring_key: Scorer}.
    """
    # This function returns a dict of {scoring_key: ssa.scorers.model_scorer.ModelScorer}
    # This is used by the new BenchmarkScorer classes, so it should remain as is for now.
    # However, the new BenchmarkScorer classes expect ssa.scorers.model_scorer.ModelScorer instances.
    for config_override in bench_config_overrides:
        if not any(config_override.startswith(f"{key}.") for key in scoring_keys):
            raise InvalidConfigurationError(
                f"Invalid --bench-config override '{config_override}': "
                f"no matching scorer in enabled scoring methods {scoring_keys}"
            )

    res = {
        scoring_key: ModelScorer(  # This is ssa.scorers.model_scorer.ModelScorer
            scoring_key,
            config_overrides=get_subconfigs(bench_config_overrides, scoring_key),
        )
        for scoring_key in scoring_keys
    }
    for scorer in res.values():
        scorer.warmup()
    return res


def get_config(
    images_dir: str,
    scorers: Dict[str, ModelScorer],
    scoring_config: Optional[Dict[str, Any]],
    execution_config: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Builds a configuration dictionary for the benchmark run, including images dir and scorer configs.
    Returns (name, config_dict).
    """
    config = {
        **default_context(),
        **(scoring_config or {}),
        **(execution_config or {}),
        "scoring": list(scorers.keys()),
        "images_dir": images_dir,
    }
    for (
        scoring_key,
        scorer_model_instance,
    ) in (
        scorers.items()
    ):  # scorer_model_instance is ssa.scorers.model_scorer.ModelScorer
        config.update(
            {
                f"scoring.{scoring_key}.{key}": val
                for key, val in scorer_model_instance.config().items()
            }
        )
    log.info(f"config:\n{pformat(config)}")
    name = unique_id(name="image-eval", random_chars=6)
    log.info(f"name={name}")
    return name, config


# Note: _generate_image_and_update_base_aggregators function removed
# as image generation has been replaced with direct image loading


def simple_eval(
    images_dir: str,
    model_scorers: Dict[str, ModelScorer],
    rescoring: bool = False,
    execution_config: Optional[Dict[str, Any]] = None,
    trace_collector: Optional[ReasoningTraceCollector] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main evaluation loop for a benchmark run using standardized scoring infrastructure.
    Loads images from directory, extracts prompts from filenames, scores them, and aggregates results.
    Returns (aggregates, run_results_list, active_benchmark_scorers_dict).
    `model_scorers` is a dict of {scoring_key: ssa.scorers.model_scorer.ModelScorer} from `resolve_benchmark_config`.
    """
    # Delegate to standardized implementation
    return simple_eval_standardized(
        images_dir, model_scorers, rescoring, execution_config, trace_collector
    )


def benchmark_model(
    images_dir: str,
    output_dir: str,
    scorers: Dict[str, ModelScorer],
    rescoring: bool = False,
    scoring_config: Optional[Dict[str, Any]] = None,
    execution_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Orchestrates a benchmark run and logs results.
    Loads images from directory, extracts prompts from filenames, evaluates with scorer setup, and logs artifacts.
    Returns the output directory path.
    """
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    name, config = get_config(images_dir, scorers, scoring_config, execution_config)
    log.info(f"Saving results to: {output_dir}")

    # Create trace collector for this benchmark run
    trace_collector = ReasoningTraceCollector()

    # Inject trace collector and output_dir into all scorers
    for scorer in scorers.values():
        scorer.set_trace_collector(trace_collector)
        # Set output directory for debug outputs (e.g., bounding box visualizations)
        if hasattr(scorer.model, "config"):
            scorer.model.config.output_dir = str(output_dir)

    aggregates, run_results, active_benchmark_scorers = simple_eval(
        images_dir,
        scorers,
        rescoring=rescoring,
        execution_config=execution_config,
        trace_collector=trace_collector,
    )

    log.info(f"run_results:\n{pformat(run_results)}")
    log.info(f"aggregates:\n{pformat(aggregates)}")

    # Aggregate metrics from benchmark scorers (add additional metrics if needed)
    for scorer_key, benchmark_scorer in active_benchmark_scorers.items():
        try:
            scorer_agg_metrics = benchmark_scorer.aggregate_metrics(final_agg_dict={})
            aggregates.update(scorer_agg_metrics)
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            # Log specific errors that can occur during metric aggregation:
            # - AttributeError: scorer missing aggregate_metrics method or required attributes
            # - KeyError: accessing missing keys in metric dictionaries
            # - ValueError: invalid values during metric calculations
            # - TypeError: type mismatches in metric operations
            log.error(
                f"Error aggregating metrics for scorer '{scorer_key}': {e}",
                exc_info=True,
            )
        except Exception as e:
            # Catch any unexpected errors but log them separately
            log.error(
                f"Unexpected error aggregating metrics for scorer '{scorer_key}': {e}",
                exc_info=True,
            )

    log.info(f"aggregates:\n{pformat(aggregates)}")

    # Save aggregated results
    final_agg_path = output_path / "aggregates.json"
    with open(final_agg_path, "w") as f:
        json.dump(aggregates, f, indent=4, sort_keys=True)

    # Save run results
    final_run_result_path = output_path / "run_results.json"
    cleaned_run_results = remove_circular_refs(run_results)
    with open(final_run_result_path, "w") as f:
        json.dump(cleaned_run_results, f, indent=4, sort_keys=True)

    # Save reasoning traces
    reasoning_traces = trace_collector.get_traces()
    if reasoning_traces:
        final_reasoning_trace_path = output_path / "reasoning_trace.json"
        with open(final_reasoning_trace_path, "w") as f:
            json.dump(
                reasoning_traces,
                f,
                indent=4,
                sort_keys=True,
                default=default_json_handler,
            )

    log.info("Results saved to:")
    log.info(f"  - {final_agg_path}")
    log.info(f"  - {final_run_result_path}")
    if reasoning_traces:
        log.info(f"  - {final_reasoning_trace_path}")

    return str(output_dir)


def remove_circular_refs(
    obj: Union[Dict[str, Any], List[Any], Any],
    seen: Optional[Set[int]] = None,
    max_depth: int = 100,
) -> Union[Dict[str, Any], List[Any], str, None, int, float, bool]:
    """
    Iteratively remove circular references and non-serializable objects from a data structure.
    Returns a version safe for JSON serialization.

    Args:
        obj: The object to process
        seen: Set of object IDs already seen (for circular reference detection)
        max_depth: Maximum depth to traverse (default: 100)

    Returns:
        A JSON-serializable version of the object
    """
    primitive_types = (int, str, bool, float, type(None))

    if seen is None:
        seen = set()

    # Use iterative approach with a stack to avoid recursion limits
    # Stack items: (object, parent_result_id, key/index, depth)
    # parent_result_id is the id to look up in results dict, or None for root
    stack = [(obj, None, None, 0)]
    results = {}

    while stack:
        current_obj, parent_result_id, parent_key, depth = stack.pop()

        # Check depth limit
        if depth > max_depth:
            if parent_result_id is not None:
                parent_result = results[parent_result_id]
                parent_result[parent_key] = "<max_depth_exceeded>"
            continue

        obj_id = id(current_obj)

        # Handle primitive types
        if isinstance(current_obj, primitive_types):
            if parent_result_id is not None:
                parent_result = results[parent_result_id]
                parent_result[parent_key] = current_obj
            else:
                return current_obj
            continue

        # Check for circular references
        if obj_id in seen:
            if parent_result_id is not None:
                parent_result = results[parent_result_id]
                parent_result[parent_key] = None
            continue

        seen.add(obj_id)

        # Handle dictionaries
        if isinstance(current_obj, dict):
            result_dict = {}
            results[obj_id] = result_dict

            if parent_result_id is not None:
                parent_result = results[parent_result_id]
                parent_result[parent_key] = result_dict

            for k, v in current_obj.items():
                if not callable(v) and not k.startswith("__"):
                    stack.append((v, obj_id, k, depth + 1))

        # Handle lists
        elif isinstance(current_obj, list):
            result_list = [None] * len(current_obj)
            results[obj_id] = result_list

            if parent_result_id is not None:
                parent_result = results[parent_result_id]
                parent_result[parent_key] = result_list

            for i, item in enumerate(current_obj):
                stack.append((item, obj_id, i, depth + 1))

        # Handle other types (convert to string)
        else:
            str_repr = str(current_obj)
            if parent_result_id is not None:
                parent_result = results[parent_result_id]
                parent_result[parent_key] = str_repr
            else:
                return str_repr

    # Return the root result
    return results.get(id(obj), obj)


class ProcessingContext:
    """Context object to reduce argument count in _process_prompt_result."""

    def __init__(self, runner, total_prompts, job_t0):
        self.runner = runner
        self.total_prompts = total_prompts
        self.job_t0 = job_t0


def _process_prompt_result(
    prompt: "Prompt",
    generated_image: Any,
    error_message: Optional[str],
    pidx: int,
    context: ProcessingContext,
    failed_gen: int,
    failed_scoring_prompts: int,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Process a single prompt result (success or failure) and return updated counters and result dict.

    This function consolidates the common logic for processing prompt results
    whether they come from parallel or sequential execution.

    Args:
        prompt: The prompt that was processed
        generated_image: The generated image (None if failed)
        error_message: Error message if generation failed
        pidx: Index of the prompt in the batch
        context: ProcessingContext with shared data
        failed_gen: Current count of failed generations
        failed_scoring_prompts: Current count of failed scoring prompts
    """
    from ssa.scoring.runner import create_prompt_context_from_legacy

    log.info(
        f"Processing result {pidx + 1}/{context.total_prompts} id={prompt.id} : {prompt.text}"
    )
    current_prompt_result: Dict[str, Any] = {
        "prompt": prompt.to_dict(),
        "prompt_id": prompt.id,
        "prompt_text": prompt.text,
    }

    if error_message:
        # Handle generation failure
        failed_gen += 1
        current_prompt_result["failed_imagegen"] = True
        current_prompt_result["error_message"] = error_message
        return current_prompt_result, failed_gen, failed_scoring_prompts

    # Handle successful generation or skip_generation mode
    skip_generation = prompt.extras.get("skip_generation", False)

    if generated_image:
        current_prompt_result["image_id"] = generated_image.info["image_id"]

        # Add source image info if present
        if prompt.image:
            current_prompt_result["source_image_uri"] = prompt.image

        # Score with standardized runner
        image_path = generated_image.info["path"]
        prompt_context = create_prompt_context_from_legacy(
            prompt, generated_image, image_path
        )

        scoring_failed = False
        for score_result in context.runner._score_single_prompt(prompt_context):
            if not score_result.success:
                scoring_failed = True
                log.error(
                    f"Scoring failed for {score_result.scorer_name} on prompt {prompt.id}"
                )
            else:
                # Store results in the legacy format for backward compatibility
                current_prompt_result[score_result.scorer_name] = score_result.raw_data

        if scoring_failed:
            failed_scoring_prompts += 1
    elif skip_generation:
        # Score without generated image (scorer will load base image internally)
        prompt_context = create_prompt_context_from_legacy(prompt, None, None)

        scoring_failed = False
        for score_result in context.runner._score_single_prompt(prompt_context):
            if not score_result.success:
                scoring_failed = True
                log.error(
                    f"Scoring failed for {score_result.scorer_name} on prompt {prompt.id}"
                )
            else:
                # Store results in the legacy format for backward compatibility
                current_prompt_result[score_result.scorer_name] = score_result.raw_data

        if scoring_failed:
            failed_scoring_prompts += 1

    return current_prompt_result, failed_gen, failed_scoring_prompts


def _load_and_parse_images(images_dir: str) -> List[Tuple[Any, "Prompt"]]:
    """
    Load images from directory and extract prompts from filenames.

    Args:
        images_dir: Directory containing images with prompt-encoded filenames

    Returns:
        List of (image_file_path, Prompt) tuples
    """
    from pathlib import Path

    from ssa.config import SUPPORTED_IMAGE_FORMATS
    from ssa.prompts import Prompt, generate_prompt_hash

    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory does not exist: '{images_dir}'")

    # Parse image files and extract prompts from filenames
    # Format: {prompt}_{img_num}.ext
    image_prompt_pairs = []
    for img_file in images_path.glob("*"):
        if img_file.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            # Parse filename: split by underscore, everything before last underscore is the prompt
            filename_no_ext = img_file.stem
            parts = filename_no_ext.rsplit("_", 1)

            if len(parts) == 2:
                prompt_text = parts[0]
                img_num = parts[1]
            else:
                # If no underscore, use entire filename as prompt
                prompt_text = filename_no_ext
                img_num = "0"

            # Create a Prompt object
            prompt_id = generate_prompt_hash(prompt_text)
            prompt = Prompt(
                id=f"{prompt_id}_{img_num}", text=prompt_text, source=img_file.name
            )

            image_prompt_pairs.append((img_file, prompt))

    log.info(f"Found {len(image_prompt_pairs)} images in {images_dir}")
    if image_prompt_pairs:
        log.info(
            f"Example: '{image_prompt_pairs[0][1].text}' from '{image_prompt_pairs[0][0].name}'"
        )

    return image_prompt_pairs


def _create_standardized_scorers(
    model_scorers: Dict[str, "ModelScorer"], prompts_list: List["Prompt"]
) -> List["ScorerInterface"]:
    """
    Create standardized scorer adapters from model scorers.

    Args:
        model_scorers: Dict mapping scorer keys to ModelScorer instances
        prompts_list: List of prompts for scorer initialization

    Returns:
        List of standardized scorer adapters
    """
    from ssa.scoring import SCORING_METHODS
    from ssa.scoring.adapters import create_scorer_adapter

    standardized_scorers = []

    for scorer_key, model_scorer in model_scorers.items():
        if scorer_key in SCORING_METHODS:
            try:
                adapter = create_scorer_adapter(
                    scorer_key,
                    model_scorer,
                    corpus=None,
                    corpus_prompts=prompts_list,
                )
                standardized_scorers.append(adapter)
            except ValueError as e:
                log.error(f"Failed to create adapter for scorer '{scorer_key}': {e}")
        else:
            log.warning(f"No adapter available for scorer '{scorer_key}'")

    return standardized_scorers


def _process_image_prompts(
    image_prompt_pairs: List[Tuple[Any, "Prompt"]],
    runner: "StandardizedBenchmarkRunner",
    trace_collector: Optional["ReasoningTraceCollector"] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Process each image-prompt pair and collect scoring results.

    Args:
        image_prompt_pairs: List of (image_path, prompt) tuples
        runner: Standardized benchmark runner for scoring
        trace_collector: Optional trace collector for recording

    Returns:
        Tuple of (run_results, failed_gen_count, failed_scoring_count)
    """
    from PIL import Image as PILImage

    from ssa.interfaces import ImageInfo
    from ssa.scoring.runner import create_prompt_context_from_legacy

    run_results: List[Dict[str, Any]] = []
    failed_gen = 0
    failed_scoring_prompts = 0
    total_prompts = len(image_prompt_pairs)

    for pidx, (img_file, prompt) in enumerate(image_prompt_pairs):
        log.info(f"Image {pidx + 1}/{total_prompts} id={prompt.id} : {prompt.text}")

        # Start trace collection for this prompt
        if trace_collector:
            trace_collector.start_prompt(prompt.id, prompt.text)

        # Load image for this prompt
        loaded_image = None
        image_path = str(img_file)
        error_message = None

        try:
            loaded_image = PILImage.open(image_path)
            # Create standardized image info object
            generated_image = ImageInfo(
                path=image_path, image_id=prompt.id, pil_image=loaded_image
            )
            log.info(f"Loaded image from {image_path}")
        except Exception as e:
            log.error(f"Failed to load image '{image_path}': {e}")
            generated_image = None
            error_message = f"Failed to load image: {e}"

        # Create prompt context for scoring
        current_prompt_result: Dict[str, Any] = {
            "prompt": prompt.to_dict(),
            "prompt_id": prompt.id,
            "prompt_text": prompt.text,
        }

        if error_message:
            # Handle loading failure
            failed_gen += 1
            current_prompt_result["failed_imagegen"] = True
            current_prompt_result["error_message"] = error_message
        elif generated_image:
            current_prompt_result["image_id"] = generated_image.info["image_id"]

            # Score with standardized runner
            prompt_context = create_prompt_context_from_legacy(
                prompt, generated_image, image_path
            )

            scoring_failed = False
            for score_result in runner._score_single_prompt(prompt_context):
                if not score_result.success:
                    scoring_failed = True
                    log.error(
                        f"Scoring failed for scorer '{score_result.scorer_name}' on prompt '{prompt.id}'"
                    )
                else:
                    # Store results in the legacy format for backward compatibility
                    current_prompt_result[score_result.scorer_name] = (
                        score_result.raw_data
                    )

            if scoring_failed:
                failed_scoring_prompts += 1

        run_results.append(current_prompt_result)

        # Finish trace collection for this prompt
        if trace_collector:
            trace_collector.finish_prompt()

    log.info(f"Completed evaluation of {total_prompts} prompts")
    log.info(f"Failed to load: {failed_gen}, Failed scoring: {failed_scoring_prompts}")

    return run_results, failed_gen, failed_scoring_prompts


def simple_eval_standardized(
    images_dir: str,
    model_scorers: Dict[str, ModelScorer],
    rescoring: bool = False,
    execution_config: Optional[Dict[str, Any]] = None,
    trace_collector: Optional[ReasoningTraceCollector] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Standardized evaluation loop using new scoring infrastructure with direct image loading.
    Extracts prompts from image filenames in format: {prompt}_{img_num}.ext
    Returns same format as simple_eval for compatibility.

    This function orchestrates the evaluation by:
    1. Loading and parsing images from the directory
    2. Creating standardized scorer adapters
    3. Processing each image-prompt pair with scoring
    4. Aggregating final metrics

    Args:
        images_dir: Directory containing images with prompt-encoded filenames
        model_scorers: Dict mapping scorer keys to ModelScorer instances
        rescoring: Whether this is a rescoring run (currently unused)
        execution_config: Optional execution configuration
        trace_collector: Optional trace collector for recording

    Returns:
        Tuple of (aggregated_metrics, run_results, active_scorers_dict)
    """
    from ssa.scoring.runner import StandardizedBenchmarkRunner

    # Step 1: Load and parse images from directory
    image_prompt_pairs = _load_and_parse_images(images_dir)

    # Step 2: Extract prompts list and create standardized scorers
    prompts_list = [pair[1] for pair in image_prompt_pairs]
    standardized_scorers = _create_standardized_scorers(model_scorers, prompts_list)

    # Create standardized runner
    runner = StandardizedBenchmarkRunner(standardized_scorers)

    # Warm up standardized scorers
    for scorer in standardized_scorers:
        try:
            log.info(f"Warming up {scorer.scorer_name} scorer...")
            scorer.warmup()
            log.info(f"Successfully warmed up {scorer.scorer_name} scorer.")
        except Exception as e:
            log.error(
                f"Error during warmup for scorer '{scorer.scorer_name}': {e}",
                exc_info=True,
            )

    # Initialize trace collector
    if trace_collector:
        trace_collector.clear()

    # Step 3: Process each image-prompt pair
    run_results, failed_gen, failed_scoring_prompts = _process_image_prompts(
        image_prompt_pairs, runner, trace_collector
    )

    # Step 4: Aggregate results
    aggregates = {
        "failed_gen": failed_gen,
        "failed_scoring_prompts": failed_scoring_prompts,
    }

    # Get standardized aggregated metrics
    standardized_aggregates = runner.compute_final_aggregates()
    aggregates.update(standardized_aggregates)

    return (
        aggregates,
        run_results,
        {scorer.scorer_name: scorer for scorer in standardized_scorers},
    )
