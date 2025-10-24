"""Numeracy Scorer Implementation.

This module provides numeracy scoring functionality for evaluating the accuracy
of object counts in images. It assesses whether generated images contain the
correct number of objects as specified in text prompts.

The scorer uses a combination of:
- LLM-based extraction to parse numerical requirements from prompts
- Object detection models to identify and count objects in images
- Scoring algorithms to compare expected vs actual counts

Example:
    Basic usage of the numeracy scorer::

        from ssa.scorers.numeracy import NumeracyScorer, NumeracyConfig
        from PIL import Image

        # Initialize scorer
        config = NumeracyConfig()
        scorer = NumeracyScorer(config)

        # Evaluate an image
        image = Image.open("image.jpg")
        image.info = {"path": "image.jpg"}
        correct, score, issues = scorer.evaluate(image, "2 cats and 3 dogs")

        print(f"Correct: {correct}")
        print(f"Score: {score:.2f}")
        print(f"Issues: {issues}")

Note:
    The scorer expects images to have an 'info' dictionary with a 'path' key.
    This is typically set automatically when loading images through the
    benchmark runner.
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
    """Numeracy scorer for evaluating object counting accuracy in images.

    This scorer evaluates whether generated images contain the correct number of
    objects as specified in text prompts. It uses LLM-based extraction to identify
    numerical requirements and object detection models to count actual objects.

    The scoring algorithm:
        1. Extract expected object counts from prompt using LLM
        2. Detect objects in image using object detection model
        3. Normalize object names (e.g., "dog" and "dogs" → "dog")
        4. Remove duplicate detections using IoU thresholding
        5. Compare detected counts with expected counts
        6. Calculate score: 0.5 for object presence + 0.5 for correct count

    Attributes:
        object_detector (ObjectDetectionModel): Model for detecting objects
        object_attribute_extractor: LLM-based extractor for parsing prompts
        filter_objects_list (List[str]): Objects to focus on during evaluation

    Example:
        Basic evaluation::

            scorer = NumeracyScorer()
            image = Image.open("cats.jpg")
            image.info = {"path": "cats.jpg"}

            # Prompt expects 2 cats
            correct, score, issues = scorer.evaluate(image, "2 cats")

            # Perfect match: correct=True, score=1.0, issues=[]
            # One cat missing: correct=False, score=0.75, issues=["Expected 2 cats, found 1"]

        With custom configuration::

            config = NumeracyConfig(
                od_model="yolov8x",
                llm="gemini/2.5-flash",
                debug=True
            )
            scorer = NumeracyScorer(config)

        Using custom object detector::

            from ssa.od import ObjectDetectionModel, YOLOV11
            custom_od = ObjectDetectionModel(YOLOV11)
            scorer = NumeracyScorer(object_detector=custom_od)

    Note:
        - Supports plural/singular normalization (dogs → dog)
        - Handles common object aliases (boat/ship, tv/television)
        - Removes duplicate detections based on IoU overlap
        - Scores range from 0.0 (no match) to 1.0 (perfect match)
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
        """Evaluate numeracy/counting accuracy in an image for a given prompt.

        This is the main entry point for numeracy evaluation. It orchestrates
        the complete evaluation workflow: object detection, count extraction,
        comparison, and scoring.

        Args:
            image: PIL Image object with 'info' dict containing 'path' key.
                The path is used for object detection.
            input_prompt: Text prompt describing expected object counts.
                Examples: "2 cats", "3 dogs and 1 cat", "five birds"
            eval_criteria: Optional evaluation criteria (unused, for compatibility)
            context: Optional context information (unused, for compatibility)

        Returns:
            Tuple[bool, float, List[str]]: A tuple containing:
                - correct (bool): True if counts match perfectly, False otherwise
                - score (float): Numerical score from 0.0 to 1.0, where:
                    * 1.0 = all objects present with correct counts
                    * 0.5-0.99 = some objects correct, some incorrect
                    * 0.0 = no objects detected or all counts wrong
                - non_conformity (List[str]): List of discrepancies found,
                    e.g., ["Expected 2 cats, found 1", "Expected 1 dog, found 0"]

        Example:
            Perfect match::

                scorer = NumeracyScorer()
                image = Image.open("2cats.jpg")  # Image with 2 cats
                image.info = {"path": "2cats.jpg"}

                correct, score, issues = scorer.evaluate(image, "2 cats")
                # Returns: (True, 1.0, [])

            Partial match::

                image = Image.open("1cat.jpg")  # Image with only 1 cat
                image.info = {"path": "1cat.jpg"}

                correct, score, issues = scorer.evaluate(image, "2 cats")
                # Returns: (False, 0.75, ["Expected 2 cats, found 1"])

            Multiple objects::

                image = Image.open("pets.jpg")  # 2 cats, 1 dog
                image.info = {"path": "pets.jpg"}

                correct, score, issues = scorer.evaluate(image, "2 cats and 1 dog")
                # Returns: (True, 1.0, [])

            No match::

                image = Image.open("dogs.jpg")  # Only dogs, no cats
                image.info = {"path": "dogs.jpg"}

                correct, score, issues = scorer.evaluate(image, "2 cats")
                # Returns: (False, 0.0, ["Expected 2 cats, found 0"])

        Raises:
            KeyError: If image.info doesn't contain 'path' key
            ValueError: If image_path is invalid or inaccessible

        Note:
            - The scorer normalizes object names (e.g., "dogs" → "dog")
            - Handles number words (e.g., "two" → 2) via LLM extraction
            - Removes duplicate detections using IoU thresholding
            - Score calculation: each object gets equal weight, split 50/50
              between presence (0.5) and correct count (0.5)
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
