#!/usr/bin/env python3
"""
Command-line interface for running SANEval benchmarks.

This script provides the main entry point for running benchmarks on
directories of images using various scoring methods. See docs/BENCHMARKS.md
for detailed usage documentation.
"""
import argparse

from ssa.benchmark_runner import benchmark_model, resolve_benchmark_config
from ssa.scoring import SCORING_METHODS, SPATIAL
from ssa.utils.logging import add_log_args, heading, log, process_log_args
from ssa.utils.security import validate_directory
from ssa.utils.system import parse_config_arg


def run_benchmarks(
    output_dir,
    scoring_keys,
    images_dir=None,
    bench_config_overrides=None,
    execution_config_overrides=None,
):
    if not images_dir:
        raise ValueError("--images-dir is required for image loading mode")

    if not output_dir:
        raise ValueError("--output-dir is required to specify where to save results")

    log.info(f"Loading images from directory: {images_dir}")
    log.info(f"Results will be saved to: {output_dir}")

    scorers = resolve_benchmark_config(scoring_keys, bench_config_overrides)
    log.info(f"Loaded scorers: {scorers}")

    log.info(heading(f"Benchmarking with images from: {images_dir}"))
    benchmark_model(
        images_dir,
        output_dir,
        scorers,
        scoring_config=bench_config_overrides,
        execution_config=execution_config_overrides,
    )

    log.info("Success!")


def main():
    """Main entry point for the benchmark script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        "Run Benchmarking.  See docs/BENCHMARKS.md for usage"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where results will be saved (run_results.json, aggregates.json, reasoning_trace.json)",
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

    # Validate input paths for security
    try:
        # Validate images directory (must exist and be readable)
        images_dir = validate_directory(
            args.images_dir, purpose="input", must_exist=True
        )
        log.info(f"Validated images directory: {images_dir}")

        # Validate output directory (can be created)
        output_dir = validate_directory(
            args.output_dir, purpose="output", allow_create=True
        )
        log.info(f"Validated output directory: {output_dir}")

    except (ValueError, FileNotFoundError, PermissionError) as e:
        log.error(f"Path validation failed: {e}")
        raise SystemExit(1) from e

    # Parse config args (these are validated inside parse_config_arg if they're file paths)
    bench_config_overrides = parse_config_arg(args.bench_config)
    execution_config_overrides = parse_config_arg(args.execution_config)

    run_benchmarks(
        str(output_dir),  # Convert Path to string for compatibility
        args.scoring,
        str(images_dir),  # Convert Path to string for compatibility
        bench_config_overrides,
        execution_config_overrides,
    )


if __name__ == "__main__":
    main()
