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
    import json

    from ssa.benchmark_runner import benchmark_model, resolve_benchmark_config
    from ssa.prompts import Corpus, Prompt
    from ssa.utils.logging import log

    # from ssa.utils.mlflow import get_artifact_cached
    ## TODO: replace get_artifact_cached with local cache logic
    # run_result = get_artifact_cached(run, "run_result.json", use_cache=False)
    if not run_result:
        raise FileNotFoundError("run_result.json artifact not found for run")
    with open(run_result) as f:
        run_result = json.loads(f.read())

    prompts = []
    prompt_to_image_mapping = []

    for _index, item in enumerate(run_result):
        if "prompt" in item:
            pkwargs = item["prompt"]
            prompt = Prompt(pkwargs.pop("id"), pkwargs.pop("text"), **pkwargs)
        elif "prompt_id" in item and "prompt_text" in item:
            # Legacy support, remove once aged out
            prompt = Prompt(id=item["prompt_id"], text=item["prompt_text"])
            if "prompt_categories" in item:
                prompt.categories = item["prompt_categories"]
            if "prompt_extensions" in item:
                prompt.extensions = item["prompt_extensions"]
            if "prompt_num_extensions" in item:
                prompt.num_extensions = item["prompt_num_extensions"]
        else:
            log.warning(f"run_result row missing valid prompt info: {item}")
            continue

        if "image_id" not in item:
            log.warning(f"Skipping run_result row, no image: {item}")
            continue

        image_id = item["image_id"]
        log.debug(f"Downloading images {image_id}")
        ## TODO: replace get_artifact_cached with local cache logic
        # image_path = get_artifact_cached(run, f"images/{image_id}")

        # Store the mapping from prompt index to image path
        prompt_to_image_mapping.append(image_path)
        prompts.append(prompt)

    # Create a custom ReplayIgModel that handles sequential prompt-to-image mapping
    class SequentialReplayIgModel:
        """ReplayIgModel that maps prompts to images sequentially for rescoring"""

        def __init__(self, prompt_to_image_mapping, config=None):
            from ssa.retry import create_retry_config

            self.key = "replay-ig"
            self.version = "replay-ig"
            self._config = config or {}
            self._config["replay"] = True
            self.prompt_to_image_mapping = prompt_to_image_mapping
            self.current_index = 0
            # Add retry_config attribute to match IgModel interface
            self.retry_config = create_retry_config(
                max_retries=0
            )  # No retries needed for replay

        def config(self):
            return self._config

        def _warmup(self, wait=True):
            # No warmup needed for replay model
            pass

        def __str__(self):
            return (
                f"SequentialReplayIgModel({len(self.prompt_to_image_mapping)} images)"
            )

        def generate_image(self, prompt: str, **kwargs):
            from ssa.ig import Image

            if self.current_index >= len(self.prompt_to_image_mapping):
                raise Exception(
                    f"ReplayIg ran out of images at index {self.current_index}"
                )

            image_path = self.prompt_to_image_mapping[self.current_index]
            self.current_index += 1

            if not image_path:
                raise Exception("ReplayIg did not have valid image for this prompt")

            image = Image.open(image_path)
            # Set required attributes for compatibility
            image.info = {
                "path": image_path,
                "image_id": f"rescore_{self.current_index-1}.png",
            }
            return image

    # setup objects
    model_params = {
        k.replace("model.", ""): v
        for k, v in run.data.params.items()
        if k.startswith("model.")
    }
    rigm = SequentialReplayIgModel(prompt_to_image_mapping, model_params)
    orig_corpus = run.data.params.get("corpus.name", "replay")
    corpus = Corpus(orig_corpus, prompts, config={"replay": True})

    scorers = resolve_benchmark_config(scoring_keys, bench_config_overrides)
    log.info(f"Loaded scorers: {scorers}")

    # Combine bench config with rescore metadata
    rescore_scoring_config = {
        **(bench_config_overrides or {}),
        "rescore_of.run_id": run.info.run_id,
        "rescore_of.exp_id": run.info.experiment_id,
        "rescore_of.run_name": run.info.run_name,
    }

    benchmark_model(
        rigm,
        corpus,
        experiment_name,
        scorers,
        scoring_config=rescore_scoring_config,
        execution_config=execution_config_overrides,
        rescoring=True,
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
