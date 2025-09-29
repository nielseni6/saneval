"""
Standardized scorer adapters for specific scoring methods.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ssa.scoring.interfaces import BenchmarkScoreResult, ScoreResult, ScorerInterface
from ssa.utils.logging import log


class CriteriaVLMScorerAdapter(ScorerInterface, ABC):
    """Abstract base class for criteria-based VLM scorer adapters (DSG and BOC)."""

    def __init__(
        self,
        model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        """
        Initialize criteria VLM adapter.

        Args:
            model_scorer: The model scorer instance (ssa.scorers.model_scorer.ModelScorer)
            config: Optional configuration dict
            corpus: Optional corpus for benchmark scorer initialization
            corpus_prompts: Optional corpus prompts for benchmark scorer initialization
        """
        self.model_scorer = model_scorer
        self.config = config or {}
        self._metrics = ["adherence", "criteria_passed", "criteria_total"]

        # Create benchmark scorer instance to handle aggregation
        benchmark_scorer_class = self._get_benchmark_scorer_class()
        self.benchmark_scorer = benchmark_scorer_class(
            model_scorer=model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key=self.scorer_name,
        )

    @abstractmethod
    def _get_benchmark_scorer_class(self):
        """Get the benchmark scorer class for this adapter."""
        pass

    @abstractmethod
    def _get_scorer_display_name(self) -> str:
        """Get the display name for error messages (e.g., 'DSG', 'BOC VLM')."""
        pass

    # TODO: This needs to be updated to handle image prompts
    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using criteria VLM with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        prompt_obj = context.get("prompt_obj")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError(
                    f"Image not provided for {self._get_scorer_display_name()} scoring"
                )

            # IMPORTANT: Feed the scoring data to the benchmark scorer for category aggregation
            # Only do this if we have a proper prompt object with required attributes
            score_prompt_failed = False
            if (
                prompt_obj
                and hasattr(prompt_obj, "categories")
                and hasattr(prompt_obj.categories, "__iter__")
                and not isinstance(prompt_obj.categories, str)
            ):
                try:
                    prompt_result = {}

                    # Call the benchmark scorer's score_prompt method to populate category aggregators
                    log.debug("Calling benchmark_scorer.score_prompt")
                    prompt_result = self.benchmark_scorer.score_prompt(
                        prompt=prompt_obj,
                        image=image,
                        image_path=context.get("image_path"),
                        prompt_result=prompt_result,
                        running_metrics=None,
                    )
                    log.debug(
                        "received results from benchmark_scorer.score_prompt results"
                    )

                    criteria = prompt_result[self.scorer_name].get("criteria", [])
                    correct = prompt_result[self.scorer_name].get("passed_criteria", 0)
                    score_val = prompt_result[self.scorer_name].get("mean_score", 0)
                    nonconforming = prompt_result[self.scorer_name].get(
                        "nonconforming", []
                    )
                    scores = prompt_result[self.scorer_name].get("scores", None)
                    explanation = prompt_result[self.scorer_name].get(
                        "explanation", None
                    )

                except Exception as e:
                    log.warning(
                        f"Failed to call benchmark scorer for category aggregation: {e}"
                    )
                    score_prompt_failed = True
            else:
                score_prompt_failed = True

            if score_prompt_failed:
                # Get or generate criteria
                if (
                    prompt_obj
                    and hasattr(prompt_obj, "criteria")
                    and prompt_obj.criteria
                ):
                    log.info("Using corpus pre-generated criteria")
                    criteria = prompt_obj.criteria
                else:
                    log.info(f"Generating eval criteria for {prompt_id}")
                    criteria = self.model_scorer.model.create_eval_criteria(prompt, [])

                log.info(f"Scoring image on {len(criteria)} criteria")

                # Evaluate with model
                correct, score_val, nonconforming, scores, explanation = (
                    self.model_scorer.model.evaluate_with_subscores(
                        image, prompt, criteria
                    )
                )

            # Convert to standardized format
            results = {
                "adherence": ScoreResult(
                    score=score_val,
                    metadata={
                        "criteria": criteria,
                        "subscores": scores,
                        "nonconforming": nonconforming,
                    },
                    reasoning=explanation,
                    confidence=score_val if isinstance(score_val, float) else None,
                ),
                "criteria_passed": ScoreResult(
                    score=correct, metadata={"total_criteria": len(criteria)}
                ),
                "criteria_total": ScoreResult(
                    score=len(criteria), metadata={"criteria_list": criteria}
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "correct": correct,
                    "score": score_val,
                    "nonconforming": nonconforming,
                    "scores": scores,
                    "explanation": explanation,
                    "criteria": criteria,
                },
            )

        except Exception as e:
            log.error(
                f"{self._get_scorer_display_name()} scoring error for {prompt_id}: {e}",
                exc_info=True,
            )
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


class DSGVLMScorerAdapter(CriteriaVLMScorerAdapter):
    """Adapter to make DSG VLM conform to standard scorer interface."""

    def _get_benchmark_scorer_class(self):
        """Get the DSG benchmark scorer class."""
        from ssa.benchmark_scorers.dsg_scorer import DsgBenchmarkScorer

        return DsgBenchmarkScorer

    def _get_scorer_display_name(self) -> str:
        """Get the display name for DSG scorer."""
        return "DSG"

    @property
    def scorer_name(self) -> str:
        return "dsg-vlm"


class GenEvalScorerAdapter(ScorerInterface):
    """Adapter for GenEval scorer."""

    def __init__(self, geneval_model_scorer, config: Optional[Dict[str, Any]] = None):
        self.model_scorer = geneval_model_scorer
        self.config = config or {}
        self._metrics = ["correct", "partial_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using GenEval with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        prompt_obj = context.get("prompt_obj")
        image_path = context.get("image_path")

        try:
            if not image_path:
                raise ValueError("Image path not provided for GenEval scoring")

            # Prepare metadata
            metadata = {}
            if prompt_obj and hasattr(prompt_obj, "extras") and prompt_obj.extras:
                metadata.update(prompt_obj.extras)
            metadata["prompt"] = prompt

            # Evaluate with GenEval model
            res = self.model_scorer.model.evaluate_image(image_path, metadata)

            correct = res.get("correct")
            partial = res.get("partial_score", 0)
            reason = res.get("reason", "")

            # Convert to standardized format
            results = {
                "correct": ScoreResult(
                    score=correct, metadata=metadata, reasoning=reason
                ),
                "partial_score": ScoreResult(
                    score=partial, metadata=metadata, reasoning=reason
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data=res,
            )

        except Exception as e:
            log.error(f"GenEval scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "gen-eval"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying GenEval model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying GenEval scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class HPSv2ScorerAdapter(ScorerInterface):
    """Adapter for HPS-v2 aesthetic scorer."""

    def __init__(self, hpsv2_model_scorer, config: Optional[Dict[str, Any]] = None):
        self.model_scorer = hpsv2_model_scorer
        self.config = config or {}
        self._metrics = ["aesthetic_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using HPS-v2 with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for HPS-v2 scoring")
            prompt_arg = prompt
            prompt_obj = context.get("prompt_obj")
            if prompt_obj:
                output_caption = getattr(prompt_obj, "output_caption", None)
                if output_caption is not None:
                    log.debug("Found output caption, using output caption for scoring")
                    prompt_arg = output_caption

            # Score with HPS-v2 model - use raw_eval method
            log.info(f"Scoring on prompt: {prompt_arg} and image: {image}.")

            score = self.model_scorer.model.raw_eval(image, prompt_arg)

            results = {
                "aesthetic_score": ScoreResult(
                    score=score, metadata={"prompt": prompt_arg}
                )
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={"score": score},
            )

        except Exception as e:
            log.error(f"HPS-v2 scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "hps-v2"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying HPS-v2 model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying HPS-v2 scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class PickScoreScorerAdapter(ScorerInterface):
    """Adapter for PickScore aesthetic scorer."""

    def __init__(self, pickscore_model_scorer, config: Optional[Dict[str, Any]] = None):
        self.model_scorer = pickscore_model_scorer
        self.config = config or {}
        self._metrics = ["pick_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using PickScore with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for PickScore scoring")

            # Score with PickScore model - use evaluate method
            correct, score, nonconforming = self.model_scorer.model.evaluate(
                image, prompt
            )

            results = {
                "pick_score": ScoreResult(score=score, metadata={"prompt": prompt})
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={"score": score},
            )

        except Exception as e:
            log.error(f"PickScore scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "pick-score"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying PickScore model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying PickScore scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class CompBenchScorerAdapter(ScorerInterface):
    """Adapter for CompBench spatial reasoning scorer."""

    def __init__(
        self,
        compbench_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = compbench_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["spatial_2d_score", "numeracy_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using CompBench with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for CompBench scoring")

            # Use the correct method - CompBench.evaluate returns a tuple of (spatial_2d_score, numeracy_score, non_conformity, correctness)
            evaluation_result = self.model_scorer.model.evaluate(image, prompt)
            if not isinstance(evaluation_result, tuple) or len(evaluation_result) != 4:
                raise TypeError(
                    "CompBench model.evaluate did not return a tuple of 4 scores."
                )

            spatial_score, numeracy_score, non_conformity, correctness = (
                evaluation_result
            )

            results = {
                "spatial_2d_score": ScoreResult(
                    score=spatial_score, metadata={"prompt": prompt}
                ),
                "numeracy_score": ScoreResult(
                    score=numeracy_score, metadata={"prompt": prompt}
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "spatial_2d_score": spatial_score,
                    "numeracy_score": numeracy_score,
                    "non_conformity": non_conformity,
                    "correctness": correctness,
                },
            )

        except Exception as e:
            log.error(f"CompBench scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "spatial-numeracy"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying CompBench model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying CompBench scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class FaceAnalysisScorerAdapter(ScorerInterface):
    """Adapter for Face Analysis bias detection scorer."""

    def __init__(
        self,
        face_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = face_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["bias_score", "detected_faces", "demographic_analysis"]

        # Create benchmark scorer instance to handle detailed demographic aggregation
        from ssa.benchmark_scorers.face_analysis_scorer import (
            FaceAnalysisBenchmarkScorer,
        )

        self.benchmark_scorer = FaceAnalysisBenchmarkScorer(
            model_scorer=face_model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key="face-analysis",
        )

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Face Analysis with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for Face Analysis scoring")

            # Analyze faces using the get_face_attributes method for backward compatibility
            analysis_result = self.model_scorer.model.get_face_attributes(image, prompt)

            # analysis_result is a list of face analysis dicts
            face_count = len(analysis_result) if analysis_result else 0

            # Calculate basic bias metrics from the face analysis results
            bias_score = 0.0
            demographics = {}
            if analysis_result:
                # Simple bias score based on demographic diversity
                unique_genders = set(
                    face.get("gender", "unknown") for face in analysis_result
                )
                unique_ethnicities = set(
                    face.get("race_ethnicity", "unknown") for face in analysis_result
                )
                bias_score = min(len(unique_genders), len(unique_ethnicities)) / max(
                    face_count, 1
                )
                demographics = {
                    "genders": list(unique_genders),
                    "ethnicities": list(unique_ethnicities),
                }

            # Also call the benchmark scorer to populate its aggregators for detailed metrics
            # Convert context to prompt object for benchmark scorer
            from ssa.prompts import Prompt

            # Create a prompt object from the context
            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                # Fallback: create a basic prompt object if not provided
                prompt_obj = Prompt(text=prompt, id=prompt_id)

            # Call benchmark scorer to populate aggregators for detailed demographic metrics
            self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result={},
                running_metrics={},
            )

            results = {
                "bias_score": ScoreResult(
                    score=bias_score,
                    metadata={
                        "calculation": "demographic_diversity",
                        "demographics": demographics,
                    },
                ),
                "detected_faces": ScoreResult(
                    score=face_count,
                    metadata={"faces": analysis_result},
                ),
                "demographic_analysis": ScoreResult(
                    score=bias_score,
                    metadata=demographics,
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data=analysis_result,
            )

        except Exception as e:
            log.error(
                f"Face Analysis scoring error for {prompt_id}: {e}", exc_info=True
            )
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "face-analysis"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Face Analysis model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Face Analysis benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


class OdAttrBindingScorerAdapter(ScorerInterface):
    """Adapter for Object Detection-Based Attribute Binding scorer."""

    def __init__(
        self,
        od_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = od_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["binding_accuracy", "detection_count", "attribute_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using OD Attribute Binding with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image_path = context.get("image_path")
        prompt_obj = context.get("prompt_obj")

        try:
            if image_path is None:
                raise ValueError(
                    "Image path not provided for OD Attribute Binding scoring"
                )

            # Get criteria questions if available
            criteria_questions = None
            if prompt_obj and hasattr(prompt_obj, "criteria") and prompt_obj.criteria:
                criteria_questions = prompt_obj.criteria

            # Perform object detection and attribute binding analysis
            binding_score = self.model_scorer.model.evaluate(
                image_path, prompt, criteria_questions
            )

            results = {
                "binding_accuracy": ScoreResult(
                    score=binding_score,
                    metadata={"prompt": prompt, "image_path": image_path},
                ),
                "detection_count": ScoreResult(
                    score=len(
                        getattr(self.model_scorer.model, "attribute_objects", [])
                    ),
                    metadata={
                        "detected_objects": getattr(
                            self.model_scorer.model, "attribute_objects", []
                        )
                    },
                ),
                "attribute_score": ScoreResult(
                    score=binding_score,
                    metadata={"score": binding_score},
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "score": binding_score,
                    "categories": context.get("categories", []),
                },
            )

        except Exception as e:
            log.error(
                f"OD Attribute Binding scoring error for {prompt_id}: {e}",
                exc_info=True,
            )
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "od-attr-binding"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying OD model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying OD Attribute Binding scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class RoundtripScorerAdapter(ScorerInterface):
    """Adapter for Roundtrip consistency scorer."""

    def __init__(
        self,
        roundtrip_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = roundtrip_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["consistency_score", "similarity_score"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Roundtrip with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")
        prompt_obj = context.get("prompt_obj")

        try:
            if image is None:
                raise ValueError("Image not provided for Roundtrip scoring")

            # Use prompt object if available (contains criteria), otherwise fall back to string
            prompt_to_evaluate = prompt_obj if prompt_obj is not None else prompt

            # Perform roundtrip consistency analysis using evaluate method
            # Try multiple possible method names for compatibility
            if hasattr(self.model_scorer.model, "evaluate"):
                roundtrip_score = self.model_scorer.model.evaluate(
                    image, prompt_to_evaluate
                )
            elif hasattr(self.model_scorer.model, "analyze_consistency"):
                roundtrip_score = self.model_scorer.model.analyze_consistency(
                    image, prompt_to_evaluate
                )
            else:
                raise AttributeError(
                    "Roundtrip model has neither 'evaluate' nor 'analyze_consistency' method"
                )

            results = {
                "consistency_score": ScoreResult(
                    score=roundtrip_score,
                    metadata={"prompt": prompt},
                ),
                "similarity_score": ScoreResult(
                    score=roundtrip_score,  # Use same score for similarity
                    metadata={"prompt": prompt},
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={"score": roundtrip_score},
            )

        except Exception as e:
            log.error(f"Roundtrip scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "roundtrip"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Roundtrip model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Roundtrip scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class BOCVLMScorerAdapter(CriteriaVLMScorerAdapter):
    """Adapter to make BOC VLM conform to standard scorer interface."""

    def _get_benchmark_scorer_class(self):
        """Get the BOC benchmark scorer class."""
        from ssa.benchmark_scorers.bag_of_criteria_scorer import BocBenchmarkScorer

        return BocBenchmarkScorer

    def _get_scorer_display_name(self) -> str:
        """Get the display name for BOC scorer."""
        return "BOC VLM"

    @property
    def scorer_name(self) -> str:
        return "boc-vlm"


class FaceIdScorerAdapter(ScorerInterface):
    """Adapter for Face ID celebrity verification scorer."""

    def __init__(
        self,
        face_id_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = face_id_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["face_id_score", "accuracy", "match_confidence"]

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Face ID with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            # Face ID scorer can handle image=None for skip_generation mode
            # It will load the base image internally using the identity ID

            # Perform face ID evaluation using evaluate method
            # Face ID returns (correct, score, nonconforming_reasons)
            evaluation_result = self.model_scorer.model.evaluate(
                image=image,
                input_prompt=prompt,
            )

            if not isinstance(evaluation_result, tuple) or len(evaluation_result) != 3:
                raise TypeError(
                    "Face ID model.evaluate did not return a tuple of 3 values."
                )

            correct, face_id_score, nonconforming_reasons = evaluation_result

            # Extract celebrity name for additional metadata
            celebrity_name = self.model_scorer.model.extract_celebrity_name(prompt)

            # Generate centralized message
            message = self.model_scorer.model.get_evaluation_message(
                prompt, face_id_score, nonconforming_reasons
            )

            accuracy = 1.0 if correct else 0.0

            results = {
                "face_id_score": ScoreResult(
                    score=face_id_score,
                    metadata={
                        "prompt": prompt,
                        "celebrity": celebrity_name,
                        "message": message,
                        "nonconforming_reasons": nonconforming_reasons,
                    },
                ),
                "accuracy": ScoreResult(
                    score=accuracy,
                    metadata={"correct": correct, "celebrity": celebrity_name},
                ),
                "match_confidence": ScoreResult(
                    score=face_id_score,
                    metadata={"confidence_percent": int(face_id_score * 100)},
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "score": face_id_score,
                    "accuracy": accuracy,
                    "correct": correct,
                    "celebrity": celebrity_name,
                    "message": message,
                    "nonconforming_reasons": nonconforming_reasons,
                },
            )

        except Exception as e:
            log.error(f"Face ID scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "face-id"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Face ID model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Face ID scorer."""
        if hasattr(self.model_scorer, "aggregate_metrics"):
            return self.model_scorer.aggregate_metrics(final_agg_dict)
        return final_agg_dict


class SpatialScorerAdapter(ScorerInterface):
    """Adapter for Spatial reasoning scorer."""

    def __init__(
        self,
        spatial_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = spatial_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["spatial_score"]

        # Create benchmark scorer instance for proper aggregation
        from ssa.benchmark_scorers.spatial_scorer import SpatialBenchmarkScorer

        self.benchmark_scorer = SpatialBenchmarkScorer(
            model_scorer=spatial_model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key="spatial",
        )

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Spatial scorer with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for Spatial scoring")

            # Evaluate with Spatial model - returns (spatial_score, non_conformity, correctness)
            spatial_score, non_conformity, correctness = (
                self.model_scorer.model.evaluate(image, prompt)
            )

            # Also call the benchmark scorer to populate its aggregators
            from ssa.prompts import Prompt

            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                prompt_obj = Prompt(text=prompt, id=prompt_id)

            self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result={},
                running_metrics={},
            )

            results = {
                "spatial_score": ScoreResult(
                    score=spatial_score,
                    metadata={
                        "prompt": prompt,
                        "non_conformity": non_conformity,
                        "correctness": correctness,
                    },
                )
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "spatial_score": spatial_score,
                    "non_conformity": non_conformity,
                    "correctness": correctness,
                },
            )

        except Exception as e:
            log.error(f"Spatial scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "spatial"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Spatial model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Spatial benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


class NumeracyScorerAdapter(ScorerInterface):
    """Adapter for Numeracy counting scorer."""

    def __init__(
        self,
        numeracy_model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = numeracy_model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["numeracy_score"]

        # Create benchmark scorer instance for proper aggregation
        from ssa.benchmark_scorers.numeracy_scorer import NumeracyBenchmarkScorer

        self.benchmark_scorer = NumeracyBenchmarkScorer(
            model_scorer=numeracy_model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key="numeracy",
        )

    def score_prompt(
        self, prompt: str, response: Any, context: Dict[str, Any]
    ) -> BenchmarkScoreResult:
        """Score using Numeracy scorer with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = context.get("image")

        try:
            if image is None:
                raise ValueError("Image not provided for Numeracy scoring")

            # Evaluate with Numeracy model - returns (numeracy_score, non_conformity, correctness)
            numeracy_score, non_conformity, correctness = (
                self.model_scorer.model.evaluate(image, prompt)
            )

            # Also call the benchmark scorer to populate its aggregators
            from ssa.prompts import Prompt

            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                prompt_obj = Prompt(text=prompt, id=prompt_id)

            self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result={},
                running_metrics={},
            )

            results = {
                "numeracy_score": ScoreResult(
                    score=numeracy_score,
                    metadata={
                        "prompt": prompt,
                        "non_conformity": non_conformity,
                        "correctness": correctness,
                    },
                )
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "numeracy_score": numeracy_score,
                    "non_conformity": non_conformity,
                    "correctness": correctness,
                },
            )

        except Exception as e:
            log.error(f"Numeracy scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "numeracy"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the underlying Numeracy model."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Numeracy benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


# Factory function to create appropriate adapter
class I2ICRITERIAVLMScorerAdapter(ScorerInterface):
    """Adapter for Image-to-Image (I2I) Criteria scorer."""

    def __init__(
        self,
        model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["adherence", "criteria_passed", "criteria_total"]

        # Create benchmark scorer instance to handle aggregation
        from ssa.benchmark_scorers.i2i_criteria_scorer import I2ICriteriaBenchmarkScorer

        self.benchmark_scorer = I2ICriteriaBenchmarkScorer(
            model_scorer=model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key="i2i-criteria-vlm",
        )

    def score_prompt(
        self,
        prompt: str,
        response: Any,
        context: Dict[str, Any],
    ) -> BenchmarkScoreResult:
        """Score using I2I Criteria VLM with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = response  # The response is the generated image

        try:
            if image is None:
                raise ValueError(
                    "Generated image not provided for I2I Criteria VLM scoring"
                )

            # Get or create prompt object
            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                from ssa.prompts import Prompt

                prompt_obj = Prompt(text=prompt, id=prompt_id)

            # Call benchmark scorer to perform the actual I2I criteria evaluation
            prompt_result = {}
            prompt_result = self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result=prompt_result,
                running_metrics={},
            )

            # Extract real scoring results from benchmark scorer
            scorer_results = prompt_result.get(self.scorer_name, {})
            criteria = scorer_results.get("criteria", [])
            correct = scorer_results.get("passed_criteria", 0)
            score_val = scorer_results.get("mean_score", 0.0)
            nonconforming = scorer_results.get("nonconforming", [])
            scores = scorer_results.get("scores", [])
            explanation = scorer_results.get("explanation", "")

            # Convert to standardized format
            results = {
                "adherence": ScoreResult(
                    score=score_val,
                    metadata={
                        "criteria": criteria,
                        "subscores": scores,
                        "nonconforming": nonconforming,
                    },
                    reasoning=explanation,
                    confidence=score_val if isinstance(score_val, float) else None,
                ),
                "criteria_passed": ScoreResult(
                    score=correct, metadata={"total_criteria": len(criteria)}
                ),
                "criteria_total": ScoreResult(
                    score=len(criteria), metadata={"criteria_list": criteria}
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data={
                    "correct": correct,
                    "score": score_val,
                    "nonconforming": nonconforming,
                    "scores": scores,
                    "explanation": explanation,
                    "criteria": criteria,
                },
            )

        except Exception as e:
            log.error(
                f"I2I Criteria VLM scoring error for {prompt_id}: {e}", exc_info=True
            )
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "i2i-criteria-vlm"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the I2I Criteria VLM scorer."""
        if hasattr(self.model_scorer, "warmup"):
            self.model_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying I2I Criteria VLM benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


class MockI2IScorerAdapter(ScorerInterface):
    """Adapter for Mock Image-to-Image (I2I) scorer."""

    def __init__(
        self,
        model_scorer,
        config: Optional[Dict[str, Any]] = None,
        corpus=None,
        corpus_prompts=None,
    ):
        self.model_scorer = model_scorer
        self.config = config or {}
        self.corpus = corpus
        self.corpus_prompts = corpus_prompts
        self._metrics = ["similarity_score", "quality_score", "coherence_score"]

        # Create benchmark scorer instance to handle aggregation
        from ssa.benchmark_scorers.mock_i2i_scorer import MockI2IBenchmarkScorer

        self.benchmark_scorer = MockI2IBenchmarkScorer(
            model_scorer=model_scorer,
            corpus=corpus,
            corpus_prompts=corpus_prompts,
            scorer_key="mock-i2i",
        )

    def score_prompt(
        self,
        prompt: str,
        response: Any,
        context: Dict[str, Any],
    ) -> BenchmarkScoreResult:
        """Score using Mock I2I scorer with standardized interface."""
        start_time = time.time()
        prompt_id = context.get("prompt_id", "unknown")
        image = response  # The response is the generated image

        try:
            if image is None:
                raise ValueError("Generated image not provided for Mock I2I scoring")

            # Call the benchmark scorer to populate aggregators for detailed metrics
            from ssa.prompts import Prompt

            # Create a prompt object from the context
            prompt_obj = context.get("prompt_obj")
            if prompt_obj is None:
                # Fallback: create a basic prompt object if not provided
                prompt_obj = Prompt(text=prompt, id=prompt_id)

            input_caption = prompt_obj.input_caption
            output_caption = prompt_obj.output_caption

            log.debug(
                f"Mock-I2I scorer score prompt function: Input caption from prompt context: {input_caption}"
            )
            log.debug(
                f"Mock-I2I scorer score prompt function: Output caption from prompt context: {output_caption}"
            )

            # Call benchmark scorer to populate aggregators
            prompt_result = {}
            prompt_result = self.benchmark_scorer.score_prompt(
                prompt=prompt_obj,
                image=image,
                image_path=context.get("image_path"),
                prompt_result=prompt_result,
                running_metrics={},
            )

            # Extract results from benchmark scorer
            scorer_results = prompt_result.get(self.scorer_name, {})
            similarity_score = scorer_results.get("similarity_score", 0.0)
            quality_score = scorer_results.get("quality_score", 0.0)
            coherence_score = scorer_results.get("coherence_score", 0.0)
            has_source_image = scorer_results.get("has_source_image", False)
            source_image_uri = scorer_results.get("source_image_uri")

            results = {
                "similarity_score": ScoreResult(
                    score=similarity_score,
                    metadata={
                        "prompt": prompt,
                        "source_image_uri": source_image_uri,
                        "has_source_image": has_source_image,
                    },
                ),
                "quality_score": ScoreResult(
                    score=quality_score, metadata={"prompt": prompt}
                ),
                "coherence_score": ScoreResult(
                    score=coherence_score, metadata={"prompt": prompt}
                ),
            }

            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results=results,
                execution_time=time.time() - start_time,
                success=True,
                raw_data=scorer_results,
            )

        except Exception as e:
            log.error(f"Mock I2I scoring error for {prompt_id}: {e}", exc_info=True)
            return BenchmarkScoreResult(
                prompt_id=prompt_id,
                scorer_name=self.scorer_name,
                results={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    @property
    def scorer_name(self) -> str:
        return "mock-i2i"

    @property
    def supported_metrics(self) -> List[str]:
        return self._metrics

    def warmup(self) -> None:
        """Warmup the mock I2I scorer."""
        if hasattr(self.benchmark_scorer, "warmup"):
            self.benchmark_scorer.warmup()

    def aggregate_metrics(self, final_agg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate aggregate_metrics to the underlying Mock I2I benchmark scorer."""
        return self.benchmark_scorer.aggregate_metrics(final_agg_dict)


def create_scorer_adapter(
    scorer_key: str,
    model_scorer,
    config: Optional[Dict[str, Any]] = None,
    corpus=None,
    corpus_prompts=None,
) -> ScorerInterface:
    """
    Factory function to create the appropriate scorer adapter.

    Args:
        scorer_key: The scoring method key (e.g., 'dsg-vlm', 'gen-eval', etc.)
        model_scorer: The model scorer instance
        config: Optional configuration
        corpus: Optional corpus for adapters that need it
        corpus_prompts: Optional corpus prompts for adapters that need it

    Returns:
        ScorerInterface instance
    """
    adapter_map = {
        "dsg-vlm": DSGVLMScorerAdapter,
        "gen-eval": GenEvalScorerAdapter,
        "hps-v2": HPSv2ScorerAdapter,
        "pick-score": PickScoreScorerAdapter,
        "spatial-numeracy": CompBenchScorerAdapter,
        "spatial": SpatialScorerAdapter,
        "numeracy": NumeracyScorerAdapter,
        "face-analysis": FaceAnalysisScorerAdapter,
        "face-id": FaceIdScorerAdapter,
        "od-attr-binding": OdAttrBindingScorerAdapter,
        "roundtrip": RoundtripScorerAdapter,
        "boc-vlm": BOCVLMScorerAdapter,
        "mock-i2i": MockI2IScorerAdapter,
        "i2i-criteria-vlm": I2ICRITERIAVLMScorerAdapter,
    }

    adapter_class = adapter_map.get(scorer_key)
    if adapter_class is None:
        raise ValueError(f"No adapter available for scorer: {scorer_key}")

    # For adapters that need corpus information, pass it
    if scorer_key in [
        "dsg-vlm",  # Added dsg-vlm to the list
        "boc-vlm",  # Added boc-vlm to the list
        "spatial-numeracy",
        "spatial",
        "numeracy",
        "face-analysis",
        "face-id",
        "od-attr-binding",
        "roundtrip",
        "mock-i2i",  # Added mock-i2i to the list
        "i2i-criteria-vlm",  # Added i2i-criteria-vlm to the list
    ]:
        return adapter_class(
            model_scorer, config, corpus=corpus, corpus_prompts=corpus_prompts
        )

    else:
        return adapter_class(model_scorer, config)
