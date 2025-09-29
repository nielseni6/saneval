"""
Spatial Scorer Implementation.

This module provides spatial scoring functionality refactored from the original
spatial_numeracy_eval.py CompBench implementation. It focuses specifically on
spatial relationship evaluation between objects in images.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ssa.od import (
    ObjectDetectionModel,
)
from ssa.scorers.spatial_numeracy_base import SpatialNumeracyBase, SpatialNumeracyConfig
from ssa.scoring import SPATIAL_QUERIES
from ssa.utils.logging import get_log

log = get_log(__file__)


class SpatialConfig(SpatialNumeracyConfig):
    """Configuration for Spatial scoring."""

    # Class-level defaults
    distance_threshold: float = (
        150.0  # Default distance threshold for spatial relationships
    )
    iou_threshold: float = 0.5  # Default IoU threshold for spatial relationships


class SpatialScorer(SpatialNumeracyBase):
    """
    Spatial relationship scorer for evaluating spatial relationships between objects in images.

    This class is refactored from the original CompBench implementation to focus specifically
    on spatial scoring functionality, returning only the spatial2d score.
    """

    def __init__(
        self,
        config: SpatialConfig = SpatialConfig(),
        config_overrides: Optional[Dict[str, Any]] = None,
        object_detector: Optional[ObjectDetectionModel] = None,
    ):
        """
        Initialize the Spatial scorer.

        Args:
            config: Configuration object for the scorer
            config_overrides: Dictionary of configuration overrides
            object_detector: Pre-initialized object detector (optional)
        """
        super().__init__(config, config_overrides, object_detector)

        # Initialize spatial-specific objects list
        self.spatial_objects = []

    def get_scorer_name(self) -> str:
        """Get the name of the scorer for debug output."""
        return "Spatial"

    def get_nonconformity_results(
        self, prompt: str, objs: List[str]
    ) -> Tuple[List[str], bool]:
        """
        Compare detected objects with expected objects to find missing items.

        Args:
            prompt: Input prompt
            objs: Detected objects

        Returns:
            Tuple of missing objects and correctness flag
        """
        extracted_objs = self.object_attribute_extractor.extract_objects(prompt)

        if not extracted_objs:
            log.warning(
                "No objects extracted from the prompt. Please ensure the prompt contains valid object descriptions."
            )
        missing_objects = self.object_attribute_extractor.object_diff(
            objs, extracted_objs
        )
        count_correctness = len(missing_objects) == 0
        return missing_objects, count_correctness

    def evaluate(
        self, image, input_prompt, eval_criteria=None, context=None
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluate spatial relationships in an image for a given prompt.

        Args:
            image: Image object with 'info' dictionary containing 'path' key
            input_prompt: The prompt to evaluate against the image

        Returns:
            Tuple containing:
                - spatial_2d_score: Float score for spatial relationships
                - non_conformity: List of missing/non-conforming objects
                - correctness: Boolean indicating evaluation correctness
        """
        spatial_2d_score, objs_unique, norm_objects = self.eval_metrics(
            image.info["path"], input_prompt
        )

        log.debug(f"Normalized objects: {norm_objects}")
        missing_objects, count_correctness = self.get_nonconformity_results(
            input_prompt, objs_unique
        )

        non_conformity = missing_objects
        # Return BaseScorer format: (correct, score, nonconforming)
        return count_correctness, spatial_2d_score, non_conformity

    def get_specific_score(
        self,
        image_path: str,
        prompt: str,
        obj: List[str],
        obj_bounding_box: List[List[float]],
        instance_score: torch.Tensor,
    ) -> float:
        """
        Get the spatial 2D score for the image and prompt.

        Args:
            image_path: Path to the image
            prompt: Text prompt
            obj: List of detected objects
            obj_bounding_box: List of bounding boxes
            instance_score: Detection confidence scores

        Returns:
            Spatial 2D score as float
        """
        return self.eval_spatial_2d(
            image_path, prompt, obj, obj_bounding_box, instance_score
        )

    def eval_spatial_2d(
        self,
        image_path: str,
        prompt: str,
        obj: List[str],
        obj_bounding_box: List[List[float]],
        instance_score: torch.Tensor,
    ) -> float:
        """
        Evaluate spatial 2D relationships between objects.

        Args:
            image_path: Path to the image
            prompt: Text prompt
            obj: List of detected objects
            obj_bounding_box: List of bounding boxes
            instance_score: Detection confidence scores

        Returns:
            Spatial 2D score as float
        """
        vocab_spatial = SPATIAL_QUERIES  # locality words

        # Use LLM-based extraction for objects and spatial relationships
        spatial_data = self.object_attribute_extractor.extract_spatial_relationship(
            prompt, vocab_spatial
        )

        if not spatial_data:
            # If LLM extraction fails, return 0 score
            log.warning("LLM spatial extraction failed - no spatial relationship found")
            return 0.0

        obj1 = spatial_data["obj1"]
        obj2 = spatial_data["obj2"]
        locality = spatial_data["relationship"]

        # if the object name is in a larger string, isolate the object name
        for i, obj_name in enumerate(obj):
            if obj1 in obj_name:
                obj[i] = obj1
            if obj2 in obj_name:
                obj[i] = obj2

        if obj1 in obj and obj2 in obj:
            obj1_pos = obj.index(obj1)
            obj2_pos = obj.index(obj2)
            obj1_bb = obj_bounding_box[obj1_pos]
            obj2_bb = obj_bounding_box[obj2_pos]
            box1, box2 = {}, {}

            box1["x_min"] = obj1_bb[0]
            box1["y_min"] = obj1_bb[1]
            box1["x_max"] = obj1_bb[2]
            box1["y_max"] = obj1_bb[3]
            box2["x_min"] = obj2_bb[0]
            box2["y_min"] = obj2_bb[1]
            box2["x_max"] = obj2_bb[2]
            box2["y_max"] = obj2_bb[3]

            score = (
                0.25 * instance_score[obj1_pos].item()
                + 0.25 * instance_score[obj2_pos].item()
            )  # score = avg across two objects score
            score += self.determine_position(locality, box1, box2) / 2
        elif obj1 in obj:
            obj1_pos = obj.index(obj1)
            score = 0.25 * instance_score[obj1_pos].item()
        elif obj2 in obj:
            obj2_pos = obj.index(obj2)
            score = 0.25 * instance_score[obj2_pos].item()
        else:
            score = 0

        if score < 0.5:
            score = 0

        # Update spatial objects for filtering
        self.spatial_objects = [obj1, obj2]

        return float(score)

    def determine_position(
        self,
        locality: str,
        box1: Dict[str, float],
        box2: Dict[str, float],
        iou_threshold: float = None,
        distance_threshold: float = None,
    ) -> float:
        """
        Determine the spatial relationship score between two bounding boxes.

        Args:
            locality: Spatial relationship type
            box1: First bounding box
            box2: Second bounding box
            iou_threshold: IoU threshold for overlap (uses config default if None)
            distance_threshold: Distance threshold for proximity (uses config default if None)

        Returns:
            Spatial relationship score
        """
        # Use config defaults if not provided
        if iou_threshold is None:
            iou_threshold = self.config.iou_threshold
        if distance_threshold is None:
            distance_threshold = self.config.distance_threshold
        # Calculate centers of bounding boxes
        box1_center = (
            (box1["x_min"] + box1["x_max"]) / 2,
            (box1["y_min"] + box1["y_max"]) / 2,
        )
        box2_center = (
            (box2["x_min"] + box2["x_max"]) / 2,
            (box2["y_min"] + box2["y_max"]) / 2,
        )

        # Calculate horizontal and vertical distances
        x_distance = box2_center[0] - box1_center[0]
        y_distance = box2_center[1] - box1_center[1]

        # Calculate IoU
        x_overlap = max(
            0, min(box1["x_max"], box2["x_max"]) - max(box1["x_min"], box2["x_min"])
        )
        y_overlap = max(
            0, min(box1["y_max"], box2["y_max"]) - max(box1["y_min"], box2["y_min"])
        )
        intersection = x_overlap * y_overlap
        box1_area = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
        box2_area = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
        union = box1_area + box2_area - intersection
        iou = intersection / union

        # Determine position based on distances and IoU and give a soft score
        score = 0
        if locality in ["next to", "on side of", "near"]:
            if (
                abs(x_distance) < distance_threshold
                or abs(y_distance) < distance_threshold
            ):
                score = 1
            else:
                score = distance_threshold / max(abs(x_distance), abs(y_distance))
        elif locality == "on the right of":
            if x_distance < 0:
                if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                    score = 1
                elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                    score = iou_threshold / iou
            else:
                score = 0
        elif locality == "on the left of":
            if x_distance > 0:
                if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                    score = 1
                elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                    score = iou_threshold / iou
            else:
                score = 0
        elif locality == "on the bottom of":
            if y_distance < 0:
                if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                    score = 1
                elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                    score = iou_threshold / iou
        elif locality == "on the top of":
            if y_distance > 0:
                if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                    score = 1
                elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                    score = iou_threshold / iou
        else:
            score = 0
        return score

    @property
    def scorer_name(self) -> str:
        """Return the name of this scorer."""
        return "spatial"
