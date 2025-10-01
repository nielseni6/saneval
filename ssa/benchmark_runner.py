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
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional

## TODO: replace IgModel with image loading from a local path
# from ssa.ig import IgModel
from ssa.prompts import Corpus
# from ssa.retry import RetryableGenerationError, create_retryer, is_retryable_error
from ssa.scorers.model_scorer import (
    ModelScorer,
)
# from ssa.utils.aggregator import Aggregator
from ssa.utils.base import (
    prepare_artifact_path,
    reset_run_dir,
    unique_id,
)
from ssa.utils.costs import global_cost_tracker
from ssa.utils.logging import log
from ssa.utils.reasoning_trace import ReasoningTraceCollector, default_json_handler
from ssa.utils.system import default_context, get_subconfigs

# Retry configuration
# Uses coordinated tenacity retry logic to prevent retry multiplication
# Tenacity retries are managed separately from HTTP-level retries
# Retries only happen for errors classified as "retryable" (network issues, timeouts, etc.)
# Actual retry counts are determined by IgModel.retry_config
MAX_GENERATION_RETRIES = 4  # Deprecated - now managed by RetryConfig
GENERATION_RETRY_DELAY = 20  # seconds


def resolve_benchmark_config(scoring_keys, bench_config_overrides):
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
            raise ValueError(
                f"--bench-config {config_override} invalid, no equivalent scoring enabled: {scoring_keys}"
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


def get_config(igm, corpus, scorers, scoring_config, execution_config):
    """
    Builds a configuration dictionary for the benchmark run, including model, corpus, and scorer configs.
    Returns (name, config_dict).
    """
    # Scorers here is the dict from resolve_benchmark_config, i.e. {scoring_key: ssa.scorers.model_scorer.ModelScorer}
    # This should be fine.
    config = {
        **default_context(),
        **(scoring_config or {}),
        **(execution_config or {}),
        "scoring": list(scorers.keys()),
    }
    config.update({f"corpus.{key}": val for key, val in corpus.config().items()})
    config.update({f"model.{key}": val for key, val in igm.config().items()})
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
    log.info(f"config:\\n{pformat(config)}")
    name = unique_id(name=str(igm.version), random_chars=6)
    log.info(f"name={name}")
    return name, config


# Helper for Image Generation and Base Metric Aggregation
def _generate_image_and_update_base_aggregators(
    igm: IgModel,
    prompt_text: str,
    latency_agg: Aggregator,
    gpu_latency_agg: Aggregator,
    cost_agg: Aggregator,
    source_image=None,
):
    """Generates an image and updates base latency and cost aggregators.
    Returns the generated image object. Raises RetryableGenerationError for retryable failures.

    Args:
        igm: Image generation model
        prompt_text: Text prompt for generation
        latency_agg: Latency aggregator
        gpu_latency_agg: GPU latency aggregator
        cost_agg: Cost aggregator
        source_image: Optional source image for image-to-image generation
    """
    retryer = create_retryer(igm)

    for attempt in retryer:
        with attempt:
            generation_type = "i2i" if source_image is not None else "t2i"
            log.info(
                f"Generating image ({generation_type}) for prompt: {prompt_text[:50]}..."
            )
            t0 = time.time()
            c0 = global_cost_tracker.total_cost()
            kwargs = {}

            # Add source image for image-to-image generation

            try:
                if source_image is not None:
                    log.info("Using source image for i2i generation")
                    image = igm.edit_image(prompt_text, image=source_image)
                else:
                    image = igm.generate_image(prompt_text, **kwargs)
            except Exception as e:
                if is_retryable_error(e):
                    log.warning(f"Classified as retryable error. {e}. Retrying.")
                    raise RetryableGenerationError(
                        f"Generation retryable error: {e}"
                    ) from e
                else:
                    log.error(f"Classified as non-retryable error. {e}. Not retrying.")
                    raise

            latency = time.time() - t0
            latency_agg.add_datum(latency)
            cost = global_cost_tracker.total_cost() - c0
            cost_agg.add_datum(cost)
            if image and (gpu_duration := image.info.get("gpu_duration")):
                gpu_latency_agg.add_datum(gpu_duration)

            return image


def simple_eval(
    igm: IgModel,
    corpus: Corpus,
    model_scorers: Dict[str, ModelScorer],
    rescoring: bool = False,
    execution_config: Optional[Dict[str, Any]] = None,
    trace_collector: Optional[ReasoningTraceCollector] = None,
):
    """
    Main evaluation loop for a benchmark run using standardized scoring infrastructure.
    Iterates over prompts, generates images, scores them, and aggregates results.
    Returns (aggregates_dict, run_results_list).
    `model_scorers` is a dict of {scoring_key: ssa.scorers.model_scorer.ModelScorer} from `resolve_benchmark_config`.
     If `rescoring` is True, it will attempt to re-score previously generated images only.
    """
    # Delegate to standardized implementation
    return simple_eval_standardized(
        igm, corpus, model_scorers, rescoring, execution_config, trace_collector
    )


def benchmark_model(
    igm,
    corpus: Corpus,
    experiment_name,
    scorers,
    rescoring=False,
    scoring_config=None,
    execution_config=None,
):
    """
    Orchestrates a benchmark run and logs results.
    Handles model/corpus/scorer setup, evaluation, and artifact logging.
    """
    reset_run_dir()
    name, config = get_config(igm, corpus, scorers, scoring_config, execution_config)

    run_id = unique_id(name=experiment_name, random_chars=8)
    log.info(f"Run ID: {run_id} for Experiment: {experiment_name} (Run Name: {name})")

    # Create trace collector for this benchmark run
    trace_collector = ReasoningTraceCollector()

    # Inject trace collector into all scorers
    for scorer in scorers.values():
        scorer.set_trace_collector(trace_collector)

    run_results, active_benchmark_scorers = simple_eval(
        igm,
        corpus,
        scorers,
        rescoring=rescoring,
        execution_config=execution_config,
        trace_collector=trace_collector,
    )

    log.info(f"run_results:\n{pformat(run_results)}")

    # Aggregate metrics from benchmark scorers
    agg = {}
    for scorer_key, benchmark_scorer in active_benchmark_scorers.items():
        try:
            scorer_agg_metrics = benchmark_scorer.aggregate_metrics(
                final_agg_dict={}
            )
            agg.update(scorer_agg_metrics)
        except Exception as e:
            log.error(
                f"Error aggregating metrics for {scorer_key}: {e}", exc_info=True
            )

    log.info(f"aggregates:\n{pformat(agg)}")

    # Save aggregated results
    final_agg_path = prepare_artifact_path("aggregates", ".json", run_id)
    with open(final_agg_path, "w") as f:
        json.dump(agg, f, indent=4, sort_keys=True)

    # Save run results
    final_run_result_path = prepare_artifact_path("run_result", ".json", run_id)
    cleaned_run_results = remove_circular_refs(run_results)
    with open(final_run_result_path, "w") as f:
        json.dump(cleaned_run_results, f, indent=4, sort_keys=True)

    # Save reasoning traces
    reasoning_traces = trace_collector.get_traces()
    if reasoning_traces:
        final_reasoning_trace_path = prepare_artifact_path(
            "reasoning_trace", ".json", run_id
        )
        with open(final_reasoning_trace_path, "w") as f:
            json.dump(
                reasoning_traces,
                f,
                indent=4,
                sort_keys=True,
                default=default_json_handler,
            )

    log.info(f"Results saved to:")
    log.info(f"  - {final_agg_path}")
    log.info(f"  - {final_run_result_path}")
    if reasoning_traces:
        log.info(f"  - {final_reasoning_trace_path}")

    return run_id


def remove_circular_refs(obj, seen=None):
    """
    Recursively remove circular references and non-serializable objects from a data structure.
    Returns a version safe for JSON serialization.
    """
    primitive_types = (int, str, bool, float, type(None))
    if seen is None:
        seen = set()
    obj_id = id(obj)

    if not isinstance(obj, primitive_types) and obj_id in seen:
        return None
    seen.add(obj_id)
    if isinstance(obj, dict):
        return {
            k: remove_circular_refs(v, seen)
            for k, v in obj.items()
            if not callable(v) and not k.startswith("__")
        }
    elif isinstance(obj, list):
        return [remove_circular_refs(i, seen) for i in obj]
    elif isinstance(obj, primitive_types):
        return obj
    else:
        return str(obj)  # fallback for non-serializable objects


class ProcessingContext:
    """Context object to reduce argument count in _process_prompt_result."""

    def __init__(self, runner, total_prompts, job_t0, cost_agg):
        self.runner = runner
        self.total_prompts = total_prompts
        self.job_t0 = job_t0
        self.cost_agg = cost_agg


def _process_prompt_result(
    prompt,
    generated_image,
    error_message,
    pidx,
    context: ProcessingContext,
    failed_gen,
    failed_scoring_prompts,
):
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

    # Calculate running metrics for this step
    running_metrics = {
        "running/step": pidx,
        "running/percent_done": (
            float(pidx + 1) / context.total_prompts if context.total_prompts > 0 else 0
        ),
        "running/cost": context.cost_agg.get_sum(),
        "running/duration": (time.time() - context.job_t0),
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

# TODO: remove need for IgModel in this function. igm should be replaced with a local image path.
def simple_eval_standardized(
    igm: IgModel,
    corpus: Corpus,
    model_scorers: Dict[str, ModelScorer],
    rescoring: bool = False,
    execution_config: Optional[Dict[str, Any]] = None,
    trace_collector: Optional[ReasoningTraceCollector] = None,
):
    """
    Standardized evaluation loop using new scoring infrastructure.
    Returns same format as simple_eval for compatibility.
    """
    # from ssa.parallel_execution import (
    #     ParallelImageGenerator,
    #     generate_image_with_retry,
    #     get_parallel_config,
    # )
    from ssa.scoring import SCORING_METHODS
    from ssa.scoring.adapters import create_scorer_adapter
    from ssa.scoring.runner import (
        StandardizedBenchmarkRunner,
    )

    # # Initialize base aggregators (same as original)
    # latency_agg = Aggregator("latency")
    # gpu_latency_agg = Aggregator("gpu_latency")
    # cost_agg = Aggregator("cost")

    # # Initialize parallel execution
    # parallel_exec_config = get_parallel_config(execution_config, default_enabled=False)
    # parallel_generator = ParallelImageGenerator(parallel_exec_config)

    # log.info(
    #     f"Parallel execution: {'enabled' if parallel_exec_config.enabled else 'disabled'}"
    # )
    # if parallel_exec_config.enabled:
    #     log.info(
    #         f"Max workers: {parallel_exec_config.max_workers}, Batch size: {parallel_exec_config.batch_size}"
    #     )

    failed_gen = 0
    failed_scoring_prompts = 0

    # Create standardized scorers using the factory function
    standardized_scorers = []

    for scorer_key, model_scorer in model_scorers.items():
        if scorer_key in SCORING_METHODS:
            try:
                adapter = create_scorer_adapter(
                    scorer_key,  # Use scorer_key directly as it's the same as adapter_key
                    model_scorer,
                    corpus=corpus,
                    corpus_prompts=corpus.prompts,
                )
                standardized_scorers.append(adapter)
            except ValueError as e:
                log.error(f"Failed to create adapter for {scorer_key}: {e}")
        else:
            log.warning(f"No adapter available for scorer: {scorer_key}")

    # Create standardized runner
    runner = StandardizedBenchmarkRunner(standardized_scorers)

    run_results: List[Dict[str, Any]] = []

    if not rescoring:
        log.info("Generating warm-up image")
        igm._warmup(wait=True)

        # Warm up standardized scorers
        for scorer in standardized_scorers:
            try:
                log.info(f"Warming up {scorer.scorer_name} scorer...")
                scorer.warmup()
                log.info(f"Successfully warmed up {scorer.scorer_name} scorer.")
            except Exception as e:
                log.error(
                    f"Error during warmup for {scorer.scorer_name} scorer: {e}",
                    exc_info=True,
                )
    else:
        log.info("Warm up is not required for rescoring runs.")

    total_prompts = len(corpus.prompts)
    job_t0 = time.time()

    # Initialize trace collector
    if trace_collector:
        trace_collector.clear()

    if parallel_exec_config.enabled:
        # Use parallel execution
        log.info(f"Starting parallel generation for {total_prompts} prompts")
        parallel_results = parallel_generator.generate_images_parallel(
            igm, corpus.prompts, corpus, latency_agg, gpu_latency_agg, cost_agg
        )

        # Process results from parallel execution
        processing_context = ProcessingContext(runner, total_prompts, job_t0, cost_agg)
        for pidx, (prompt, generated_image, error_message) in enumerate(
            parallel_results
        ):
            # Start trace collection for this prompt
            if trace_collector:
                trace_collector.start_prompt(prompt.id, prompt.text)

            current_prompt_result, failed_gen, failed_scoring_prompts = (
                _process_prompt_result(
                    prompt,
                    generated_image,
                    error_message,
                    pidx,
                    processing_context,
                    failed_gen,
                    failed_scoring_prompts,
                )
            )
            run_results.append(current_prompt_result)

            # Finish trace collection for this prompt
            if trace_collector:
                trace_collector.finish_prompt()
    else:
        # Use sequential execution
        processing_context = ProcessingContext(runner, total_prompts, job_t0, cost_agg)
        for pidx, prompt in enumerate(corpus.prompts):
            log.info(
                f"Prompt {pidx + 1}/{total_prompts} id={prompt.id} : {prompt.text}"
            )

            # Start trace collection for this prompt
            if trace_collector:
                trace_collector.start_prompt(prompt.id, prompt.text)

            # Load source image if this is an image-to-image prompt
            source_image = None
            if prompt.image:
                try:
                    source_image = corpus.get_prompt_image(prompt)
                    if source_image:
                        log.info(
                            f"Loaded source image for i2i generation: {prompt.image}"
                        )
                    else:
                        log.warning(f"Failed to load source image: {prompt.image}")
                except Exception as e:
                    log.error(f"Error loading source image for prompt {prompt.id}: {e}")

            # Check if generation should be skipped
            skip_generation = prompt.extras.get("skip_generation", False)

            if skip_generation:
                # Skip generation, use base image directly
                if not prompt.text.isdigit():
                    log.warning(
                        f"skip_generation=true but text '{prompt.text}' is not numeric"
                    )
                generated_image = None
                error_message = None
            else:
                # Generate image with retry (tenacity handles retries automatically)
                try:
                    generated_image = generate_image_with_retry(
                        igm,
                        prompt.text,
                        latency_agg,
                        gpu_latency_agg,
                        cost_agg,
                        source_image,
                    )
                    error_message = None
                except Exception as e:
                    log.error(
                        f"Image generation failed permanently for prompt {prompt.id}: {e}",
                        exc_info=True,
                    )
                    generated_image = None
                    error_message = str(e)

            # Process the result using the common function
            current_prompt_result, failed_gen, failed_scoring_prompts = (
                _process_prompt_result(
                    prompt,
                    generated_image,
                    error_message,
                    pidx,
                    processing_context,
                    failed_gen,
                    failed_scoring_prompts,
                )
            )
            run_results.append(current_prompt_result)

            # Finish trace collection for this prompt
            if trace_collector:
                trace_collector.finish_prompt()

    # # Aggregate results using both new and legacy methods
    # agg = {
    #     "failed_gen": failed_gen,
    #     "failed_scoring_prompts": failed_scoring_prompts,
    #     "latency": latency_agg.get_aggregates(),
    #     "gpu_latency": gpu_latency_agg.get_aggregates(),
    #     "cost": cost_agg.get_aggregates(),
    # }

    # Get standardized aggregated metrics
    # standardized_agg = runner.compute_final_aggregates()
    # agg.update(standardized_agg)

    return (
        # agg,
        run_results,
        {scorer.scorer_name: scorer for scorer in standardized_scorers},
    )
