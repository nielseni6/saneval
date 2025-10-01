#!/usr/bin/env python3
# See docs in docs/BENCHMARKS.md for usage
import argparse
import json
import os
import shlex

# from ssa.aws.s3 import get_cached_html_report_path
# from ssa.benchmark_email import (
#     send_email_with_new_infographic,
#     send_email_with_s3_html_report,
# )
from ssa.benchmark_runner import (
    benchmark_model,
    resolve_benchmark_config,
)
from ssa.caching import (
    destroy_cache,
    disable_all_cache_flags,
    get_global_cache,
    set_config,
    setup_caching,
)
# from ssa.cloud import launch_fargate_tasks
# from ssa.ig import DEFAULT_VERSION, IgModel
# from ssa.ip_risk_audit.email_text_generator import (
#     get_detailed_analysis,
#     get_executive_summary,
# )
# from ssa.ip_risk_audit.generate_corpus import generate_corpus
from ssa.prompts import (
    get_corpus,
    read_corpus_dir,
)
from ssa.rescore import re_score
from ssa.scoring import (
    SPATIAL,
    SCORING_METHODS,
)
# from ssa.utils.costs import global_cost_tracker
from ssa.utils.logging import (
    add_log_args,
    heading,
    log,
    process_log_args,
)
from ssa.utils.system import (
    get_user,
    parse_config_arg,
)

CACHE_VERSION = os.getenv("SSA_CACHE_VERSION", "V3")


def run_benchmarks(
    model_keys,
    experiment_name,
    scoring_keys,
    corpus_keys,
    max_retries=0,
    total_expected_tasks=0,
    asset_name=None,
    asset_type=None,
    bench_config_overrides=None,
    execution_config_overrides=None,
    model_config_overrides=None,
    on_cloud=False,
    enable_cache=False,
):
    assert (
        corpus_keys is not None and len(corpus_keys) > 0
    ), "At least one corpus required"
    assert model_keys, "At least one model required"
    model_config_overrides = model_config_overrides or [None]

    corpora = []

    for key in corpus_keys:
        as_dir = read_corpus_dir(key)
        if as_dir:
            corpora.extend(as_dir)
            log.info(f"Found {len(as_dir)} corpora inside {key}: {as_dir}")
        else:
            # validate corpora keys before possible cloud launch
            get_corpus(key, validate_only=True)
            corpora.append(key)

    log.info(
        f"Running {len(model_keys)} model(s) over {len(corpora)} corpora over {len(model_config_overrides)} configs {'on cloud' if on_cloud else ''}"
    )
    if on_cloud:
        log.debug("Validating model keys and configs...")
        # Instantiate IGMs here just to validate keys and config-overrides
        log.info(
            f"Using tenacity-based retries: {max_retries + 1} attempts for generation (retryable errors only)"
        )
        for model_key in model_keys:
            for config_override in model_config_overrides:
                igm = IgModel(
                    model_key,
                    save=True,
                    max_retries=max_retries,
                    config_overrides=config_override,
                )
                log.info(f"Validated config-overrides: {igm.config()}")
        log.debug("Validation successful, constructing tasks...")

        # Construct commands for tasks
        task_idx = 0
        commands = []
        for corpus_key in corpora:
            for model_key in model_keys:
                for config_override in model_config_overrides:
                    task_idx += 1
                    command = f"python ssa/benchmark.py --experiment-name {shlex.quote(experiment_name)} --corpus {shlex.quote(corpus_key)} --model {shlex.quote(model_key)}"
                    command += f" --scoring {' '.join(scoring_keys)}"
                    if bench_config_overrides:
                        command += (
                            f" --bench-config '{json.dumps(bench_config_overrides)}'"
                        )
                    if execution_config_overrides:
                        command += f" --execution-config '{json.dumps(execution_config_overrides)}'"
                    if config_override:
                        command += f" --model-config '{json.dumps(config_override)}'"
                    if enable_cache:
                        command += " --enable-cache"
                    log.info(f"Subtask {task_idx}/{total_expected_tasks} : {command}")
                    commands.append(command)

        # Launch the tasks
        launch_fargate_tasks(commands)
    else:
        for corpus_key in corpora:
            corpus = get_corpus(corpus_key)
            log.info(f"Loaded corpus: {corpus}")
            if corpus.has_images():
                log.info(
                    f"Corpus contains {corpus.image_count()} prompts with images (i2i capable)"
                )
            else:
                log.info("Corpus contains text-only prompts (t2i mode)")

            scorers = resolve_benchmark_config(scoring_keys, bench_config_overrides)
            log.info(f"Loaded scorers: {scorers}")

            log.info(
                f"Using tenacity-based retries: {max_retries + 1} attempts for generation (retryable errors only)"
            )
            for model_key in model_keys:
                for config_override in model_config_overrides:
                    # Using clean tenacity-based retry logic - no nested retries
                    igm = IgModel(
                        model_key,
                        save=True,
                        max_retries=max_retries,
                        config_overrides=config_override,
                    )
                    log.info(heading(f"Benchmarking IGM {igm} {config_override or ''}"))
                    benchmark_model(
                        igm,
                        corpus,
                        experiment_name,
                        scorers,
                        scoring_config=bench_config_overrides,
                        execution_config=execution_config_overrides,
                    )
            log.info(
                f"Total benchmark run cost:\n{global_cost_tracker.total_cost_report()}"
            )

        log.info("Success!")


def run_ip_risk_audit_workflow(
    experiment_name,
    asset_name,
    asset_type,
    enable_cache=False,
    **kwargs,
):
    """
    IP risk audit workflow has been disabled (depends on removed features).
    """
    log.error("IP risk audit workflow is disabled (depends on removed features)")
    raise NotImplementedError("IP risk audit workflow requires removed features")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        "Run Benchmarking.  See docs/BENCHMARKS.md for usage"
    )
    DEFAULT_MODEL_ARG = [DEFAULT_VERSION]
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        help="ig model version(s) to benchmark",
        default=DEFAULT_MODEL_ARG,
    )
    DEFAULT_CORPUS_ARG = ["mini-corpus"]
    parser.add_argument(
        "--corpus",
        type=str,
        nargs="*",
        help="corpus key(s).  Pass a directory to run multiple corpora",
        default=DEFAULT_CORPUS_ARG,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="name for experiment",
        default=f"{get_user()}-bench-ig",
    )
    parser.add_argument(
        "--re-score",
        help="Re-score a previously generated image-set.  Pass in run-id(s), or experiment-name(s)",
        type=str,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Override config values of the IG-Model.  You can pass single fields 'key=val' , a dictionary '{\"key\":\"val\"}' or a filepath '/path/config.json'.  Multiple args are treated as independent runs, not combined",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--bench-config",
        type=str,
        help="Override config values of the benchmark itself.  You can pass single fields 'key=val' , a dictionary '{\"key\":\"val\"}' or a filepath '/path/config.json'.",
        default=None,
    )
    parser.add_argument(
        "--execution-config",
        type=str,
        help="Override execution settings (parallel execution, batch sizes, etc.). You can pass single fields 'key=val' , a dictionary '{\"key\":\"val\"}' or a filepath '/path/config.json'.",
        default=None,
    )
    parser.add_argument(
        "--scoring",
        nargs="*",
        help="Scoring methods to run, can be combined",
        default=[SPATIAL],
        choices=SCORING_METHODS,
    )
    parser.add_argument(
        "--on-cloud",
        action="store_true",
        help="Run the benchmark as a task on the cloud.  If multiple models/corpora are specified they will be launched as multiple concurrent tasks",
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--asset-type",
        type=str,
        help="What is the asset you want to generate a corpus for? (Asset e.g., Portrait, Cartoon, Logo)",
    )
    parser.add_argument(
        "--asset-name",
        type=str,
        help="Name of the celebrity, character or logo",
    )
    parser.add_argument(
        "--prompt-num",
        type=int,
        default=5,
        help="How many prompts to generate",
    )
    parser.add_argument("--total-expected-tasks", type=int)
    parser.add_argument(
        "--custom-corpus",
        action="store_true",
        help="Generate a custom corpus for the asset instead of using a pre-defined one.",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable prompt caching for this benchmark run.",
    )

    add_log_args(parser)
    args = parser.parse_args()
    process_log_args(args)

    # Handle cache disable flag
    if not args.enable_cache:
        # Destroy any existing cache and disable all cache flags
        try:
            destroy_cache()
            disable_all_cache_flags()
        except Exception as e:
            log.warning(f"Failed to destroy cache: {e}")
            # Continue execution even if cache destruction fails
    else:
        # Set the global config and use it to init the cache
        set_config({"enabled": True, "auto_init": True})
        setup_caching()
        if get_global_cache() is not None:
            log.info("Prompt caching enabled via --enable-cache flag")

    model_config_overrides = args.model_config or [None]
    bench_config_overrides = parse_config_arg(args.bench_config)
    execution_config_overrides = parse_config_arg(args.execution_config)

    if args.re_score:
        if args.model != DEFAULT_MODEL_ARG:
            raise ValueError("Cannot change --model when re-scoring")
        if args.corpus != DEFAULT_CORPUS_ARG:
            raise ValueError("Cannot change --corpus when re-scoring")
        if args.model_config:
            raise ValueError("Cannot change --model-config when re-scoring")
        re_score(
            args.re_score,
            args.experiment_name,
            args.scoring,
            bench_config_overrides,
            execution_config_overrides,
            on_cloud=args.on_cloud,
        )
    elif args.custom_corpus and not all(
        [
            args.asset_name,
            args.asset_type,
            args.scoring,
            args.total_expected_tasks,
        ]
    ):
        parser.error(
            "--experiment-name, --asset-name, --asset-type, --total-expected-tasks, and --scoring are required when running --custom-corpus."
        )
    else:
        if args.custom_corpus:
            run_ip_risk_audit_workflow(
                model_keys=args.model,
                experiment_name=args.experiment_name,
                scoring_keys=args.scoring,
                max_retries=args.max_retries,
                total_expected_tasks=args.total_expected_tasks,
                asset_name=args.asset_name,
                asset_type=args.asset_type,
                bench_config_overrides=bench_config_overrides,
                execution_config_overrides=execution_config_overrides,
                model_config_overrides=[
                    parse_config_arg(c) for c in model_config_overrides
                ],
                on_cloud=args.on_cloud,
                enable_cache=args.enable_cache,
            )
        else:
            run_benchmarks(
                args.model,
                args.experiment_name,
                args.scoring,
                args.corpus,
                args.max_retries,
                args.total_expected_tasks,
                args.asset_name,
                args.asset_type,
                bench_config_overrides,
                execution_config_overrides,
                [parse_config_arg(c) for c in model_config_overrides],
                on_cloud=args.on_cloud,
                enable_cache=args.enable_cache,
            )
