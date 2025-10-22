"""
Numeracy Scorer Implementation.

This module provides numeracy scoring functionality refactored from the original
spatial_numeracy_eval.py SANEval implementation. It focuses specifically on
counting and numerical evaluation of objects in images.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ssa.od import ObjectDetectionModel
from ssa.scorers.spatial_numeracy_base import SpatialNumeracyBase, SpatialNumeracyConfig
from ssa.scoring import NUMERIC_QUERIES
from ssa.utils.logging import get_log
from ssa.utils.od_tools import calculate_iou

log = get_log(__file__)


class NumeracyConfig(SpatialNumeracyConfig):
    """Configuration for Numeracy scoring."""

    pass


class NumeracyScorer(SpatialNumeracyBase):
    """
    Numeracy scorer for evaluating object counting and numerical relationships in images.

    This class is refactored from the original SANEval implementation to focus specifically
    on numeracy scoring functionality, returning only the numeracy score.
    """

    def __init__(
        self,
        config: NumeracyConfig = NumeracyConfig(),
        config_overrides: Optional[Dict[str, Any]] = None,
        object_detector: Optional[ObjectDetectionModel] = None,
    ):
        """
        Initialize the Numeracy scorer.

        Args:
            config: Configuration object for the scorer
            config_overrides: Dictionary of configuration overrides
            object_detector: Pre-initialized object detector (optional)
        """
        super().__init__(config, config_overrides, object_detector)

        # Initialize numeracy-specific objects list

    def get_scorer_name(self) -> str:
        """Get the name of the scorer for debug output."""
        return "Numeracy"

    def get_nonconformity_results(
        self, prompt: str, objs: List[str]
    ) -> Tuple[List[str], bool]:
        """
        Compare detected objects with expected numeric objects to find discrepancies.

        Args:
            prompt: Input prompt
            objs: Detected objects

        Returns:
            Tuple of numeric differences and correctness flag
        """
        extract_numeracy = self.object_attribute_extractor.extract_numeracy_objects(
            prompt
        )

        if not extract_numeracy:
            log.warning(
                "No numeracy objects extracted from the prompt. Please ensure the prompt contains valid numeracy descriptions."
            )
        numeric_difference = self.object_attribute_extractor.numeric_diff(
            objs,
            extract_numeracy,  # Use only actually detected objects, not norm_objects which includes prompt objects
        )
        count_correctness = len(numeric_difference) == 0
        return numeric_difference, count_correctness

    def evaluate(
        self, image, input_prompt, eval_criteria=None, context=None
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluate numeracy/counting in an image for a given prompt.

        Args:
            image: Image object with 'info' dictionary containing 'path' key
            input_prompt: The prompt to evaluate against the image

        Returns:
            Tuple containing:
                - numeracy_score: Float score for numerical accuracy
                - non_conformity: List of missing/non-conforming objects
                - correctness: Boolean indicating evaluation correctness
        """
        numeracy_score, objs_unique, norm_objects = self.eval_metrics(
            image.info["path"], input_prompt
        )

        log.debug(f"Normalized objects: {norm_objects}")
        numeric_diff, count_correctness = self.get_nonconformity_results(
            input_prompt, norm_objects
        )

        non_conformity = numeric_diff
        # Return BaseScorer format: (correct, score, nonconforming)
        return count_correctness, numeracy_score, non_conformity

    def get_specific_score(
        self,
        image_path: str,
        prompt: str,
        obj: List[str],
        obj_bounding_box: List[List[float]],
        instance_score: torch.Tensor,
    ) -> float:
        """
        Get the numeracy score for the image and prompt.

        Args:
            image_path: Path to the image
            prompt: Text prompt
            obj: List of detected objects
            obj_bounding_box: List of bounding boxes
            instance_score: Detection confidence scores

        Returns:
            Numeracy score as float
        """
        return self.eval_numeracy(
            image_path, prompt, obj, obj_bounding_box, instance_score
        )

    def eval_numeracy(
        self,
        image_path: str,
        prompt: str,
        obj: List[str],
        obj_bounding_box: List[List[float]],
        instance_score: torch.Tensor,
    ) -> float:
        """
        Evaluate numeracy/counting accuracy for objects in the image using LLM-based extraction.

        Args:
            image_path: Path to the image
            prompt: Text prompt
            obj: List of detected objects
            obj_bounding_box: List of bounding boxes
            instance_score: Detection confidence scores

        Returns:
            Numeracy score as float
        """
        # Use LLM-based extraction for numeracy relationships
        numeracy_data = self.object_attribute_extractor.extract_numeracy_relationship(
            prompt, NUMERIC_QUERIES
        )

        if not numeracy_data or not numeracy_data.get("objects"):
            # If LLM extraction fails, return 0 score
            log.warning(
                "LLM numeracy extraction failed - no numeracy relationships found"
            )
            return 0.0

        expected_objects = numeracy_data["objects"]  # Dict[str, int]
        my_obj = list(expected_objects.keys())
        num_obj = list(expected_objects.values())

        log.debug(f"Expected numeracy objects: {expected_objects}")

        score = 0.0

        # Handle case where no objects with numbers are found
        if len(my_obj) == 0:
            return score

        weight = 1.0 / len(my_obj)

        # Normalize detected object names
        for j, obj_i in enumerate(obj):
            obj[j] = self.normalize_object_name(obj[j])

        # Normalize expected object names and match with detected objects
        for i, my_obj_i in enumerate(my_obj):
            my_obj[i] = self.normalize_object_name(my_obj_i)

        # Remove duplicates using IoU calculation
        # This ensures deduplication is based on standardized object names
        new_obj = []
        new_bbox = []
        for i in range(len(obj)):
            flag = 0
            for j in range(len(new_obj)):
                if (
                    calculate_iou(obj_bounding_box[i], new_bbox[j])
                    and obj[i] == new_obj[j]  # Compare standardized object names
                ):
                    flag = 1
                    break
            if flag == 0:
                new_obj.append(obj[i])
                new_bbox.append(obj_bounding_box[i])

        # Calculate numeracy score
        for i, my_obj_i in enumerate(my_obj):
            if my_obj_i in new_obj:
                score += 0.5 * weight  # Object present
                num_detected = new_obj.count(my_obj_i)
                num_expected = num_obj[i]
                if num_detected == num_expected:
                    score += 0.5 * weight  # Correct count

        # Update numeracy objects for filtering - override base class default
        numeracy_aliases = []
        for obj_name, alias in [
            ("boat", "ship"),
            ("tv", "telivision"),
            ("fish", "goldfish"),
            ("picture", "painting"),
        ]:
            if obj_name in my_obj or alias in my_obj:
                numeracy_aliases.extend([obj_name, alias])

        self.filter_objects_list = my_obj + numeracy_aliases

        return float(score)

    @property
    def scorer_name(self) -> str:
        """Return the name of this scorer."""
        return "numeracy"
