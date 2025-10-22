"""
Rescoring logic for SSA benchmarking.

NOTE: Rescoring functionality has been disabled as it depends on MLflow,
which has been removed from the codebase.
"""


def re_score_run_disabled(
    run,
    experiment_name,
    scoring_keys,
    bench_config_overrides,
    execution_config_overrides=None,
):
    """
    This function has been disabled as it depends on MLflow.

    This function was used for rescoring previous benchmark runs, but it depended
    on MLflow for artifact management. The functionality has been disabled and may
    be re-implemented with a local caching solution in the future.
    """
    from ssa.utils.logging import log

    log.error("Rescoring functionality is disabled (MLflow dependency removed)")
    raise NotImplementedError(
        "Rescoring functionality requires MLflow which has been removed"
    )


def find_runs_from_source_targets_disabled(source_targets):
    """
    This function has been disabled as it depends on MLflow.
    """
    from ssa.utils.logging import log

    log.error("Rescoring functionality is disabled (MLflow dependency removed)")
    raise NotImplementedError(
        "Rescoring functionality requires MLflow which has been removed"
    )


def re_score(
    source_targets,
    experiment_name,
    scoring_keys,
    bench_config_overrides,
    execution_config_overrides=None,
):
    """
    Rescoring has been disabled as it depends on MLflow.
    """
    from ssa.utils.logging import log

    log.error("Rescoring functionality is disabled (MLflow dependency removed)")
    raise NotImplementedError(
        "Rescoring functionality requires MLflow which has been removed"
    )
