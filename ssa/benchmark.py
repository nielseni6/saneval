#!/usr/bin/env python3
# See docs in docs/BENCHMARKS.md for usage
import argparse

from ssa.benchmark_runner import (
    benchmark_model,
    resolve_benchmark_config,
)
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


def run_benchmarks(
    experiment_name,
    scoring_keys,
    images_dir=None,
    bench_config_overrides=None,
    execution_config_overrides=None,
):
    if not images_dir:
        raise ValueError("--images-dir is required for image loading mode")

    log.info(f"Loading images from directory: {images_dir}")

    scorers = resolve_benchmark_config(scoring_keys, bench_config_overrides)
    log.info(f"Loaded scorers: {scorers}")

    log.info(heading(f"Benchmarking with images from: {images_dir}"))
    benchmark_model(
        images_dir,
        experiment_name,
        scorers,
        scoring_config=bench_config_overrides,
        execution_config=execution_config_overrides,
    )

    log.info("Success!")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        "Run Benchmarking.  See docs/BENCHMARKS.md for usage"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="name for experiment",
        default=f"{get_user()}-bench-ig",
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
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images to evaluate (e.g., 'images/samples/attribute_binding')",
    )
    add_log_args(parser)
    args = parser.parse_args()
    process_log_args(args)

    bench_config_overrides = parse_config_arg(args.bench_config)
    execution_config_overrides = parse_config_arg(args.execution_config)

    run_benchmarks(
        args.experiment_name,
        args.scoring,
        args.images_dir,
        bench_config_overrides,
        execution_config_overrides,
    )
