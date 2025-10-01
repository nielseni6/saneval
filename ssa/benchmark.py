#!/usr/bin/env python3
# See docs in docs/BENCHMARKS.md for usage
import argparse

from ssa.benchmark_runner import (
    benchmark_model,
    resolve_benchmark_config,
)

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


def run_benchmarks(
    experiment_name,
    scoring_keys,
    corpus_keys,
    images_dir=None,
    bench_config_overrides=None,
    execution_config_overrides=None,
):
    assert (
        corpus_keys is not None and len(corpus_keys) > 0
    ), "At least one corpus required"

    if images_dir:
        log.info(f"Loading images from directory: {images_dir}")
    else:
        raise ValueError("--images-dir is required for image loading mode")

    corpora = []

    for key in corpus_keys:
        as_dir = read_corpus_dir(key)
        if as_dir:
            corpora.extend(as_dir)
            log.info(f"Found {len(as_dir)} corpora inside {key}: {as_dir}")
        else:
            # validate corpora keys
            get_corpus(key, validate_only=True)
            corpora.append(key)

    log.info(
        f"Running benchmarks over {len(corpora)} corpora with images from {images_dir}"
    )

    for corpus_key in corpora:
        corpus = get_corpus(corpus_key)
        log.info(f"Loaded corpus: {corpus}")
        log.info(f"Corpus contains {len(corpus.prompts)} prompts")

        scorers = resolve_benchmark_config(scoring_keys, bench_config_overrides)
        log.info(f"Loaded scorers: {scorers}")

        log.info(heading(f"Benchmarking with images from: {images_dir}"))
        benchmark_model(
            images_dir,
            corpus,
            experiment_name,
            scorers,
            scoring_config=bench_config_overrides,
            execution_config=execution_config_overrides,
        )

    log.info("Success!")


def run_ip_risk_audit_workflow(
    experiment_name,
    asset_name,
    asset_type,
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

    if args.re_score:
        if args.corpus != DEFAULT_CORPUS_ARG:
            raise ValueError("Cannot change --corpus when re-scoring")
        re_score(
            args.re_score,
            args.experiment_name,
            args.scoring,
            bench_config_overrides,
            execution_config_overrides,
        )
    else:
        run_benchmarks(
            args.experiment_name,
            args.scoring,
            args.corpus,
            args.images_dir,
            bench_config_overrides,
            execution_config_overrides,
        )
