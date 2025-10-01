"""
Spatial Numeracy Base Scorer Implementation.

This module provides the base class for spatial and numeracy scorers, containing
all shared functionality that was previously duplicated between the two scorers.
"""

import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ssa.od import (
    DEFAULT_OD_VERSION,
    ObjectDetectionModel,
)
from ssa.scorers.scorer_base import BaseScorer, BaseScorerConfig
from ssa.utils.logging import get_log
from ssa.utils.nlp_tools import ObjectAttributeExtractor
from ssa.utils.od_tools import (
    filter_objects,
    plot_bboxes,
)
from ssa.utils.secrets import get_secret
from ssa.vlm import DEFAULT_LLM_VERSION, Llm

log = get_log(__file__)


class SpatialNumeracyConfig(BaseScorerConfig):
    """Base configuration for Spatial and Numeracy scoring."""

    od_model: str = DEFAULT_OD_VERSION
    od_classes: str = "data/pred_classes/compbench.json"
    pred_classes: str = "from_json"
    debug: bool = True
    llm: str = DEFAULT_LLM_VERSION


class SpatialNumeracyBase(BaseScorer):
    """
    Base class for spatial and numeracy scorers containing shared functionality.

    This class contains all the common logic that was previously duplicated between
    SpatialScorer and NumeracyScorer, including object detection, NLP processing,
    and common evaluation methods.
    """

    def __init__(
        self,
        config: SpatialNumeracyConfig,
        config_overrides: Optional[Dict[str, Any]] = None,
        object_detector: Optional[ObjectDetectionModel] = None,
    ):
        """
        Initialize the base scorer.

        Args:
            config: Configuration object for the scorer
            config_overrides: Dictionary of configuration overrides
            object_detector: Pre-initialized object detector (optional)
        """
        self.current_run_time = time.strftime("%Y%m%d_%H%M%S")

        # Install HuggingFace token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = get_secret("HUGGINGFACE_TOKEN")

        # Call parent constructor first
        super().__init__(config, config_overrides)

        # Initialize object detector
        if object_detector:
            self.object_detector = object_detector
        else:
            self.object_detector = ObjectDetectionModel(
                self.config.od_model, self.config.pred_classes, self.config.od_classes
            )

        # Load object names
        obj_names_path = (
            Path(__file__).parent.parent
            / "thirdparty"
            / "compbench"
            / "object_names.txt"
        )
        with obj_names_path.open() as cls_file:
            self.classnames = [line.strip() for line in cls_file]

        # Load objects data
        objects_path = (
            Path(__file__).parent.parent
            / "thirdparty"
            / "compbench"
            / "data"
            / "examples"
            / "new_objects.txt"
        )
        with objects_path.open() as f:
            objects = f.read().splitlines()
            self.single_plural_map = {
                obj.split(" - ")[1].strip().lower(): obj.split(" - ")[0].strip().lower()
                for obj in objects
            }

        # Initialize LLM and object attribute extractor
        self.llm = Llm(self.config.llm)
        self.object_attribute_extractor = ObjectAttributeExtractor(self.llm)

        # Initialize type-specific objects list (to be set by subclasses)
        self.filter_objects_list = []

    @abstractmethod
    def evaluate(
        self, image, input_prompt, eval_criteria=None, context=None
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluate the scorer for a given image and prompt.

        Args:
            image: Image object with 'info' dictionary containing 'path' key
            input_prompt: The prompt to evaluate against the image
            eval_criteria: Not used by spatial/numeracy scorers
            context: Additional context (optional)

        Returns:
            Tuple containing:
                - correct: Boolean indicating evaluation correctness
                - score: Float score for the evaluation
                - non_conforming: List of missing/non-conforming objects
        """
        pass

    @abstractmethod
    def get_specific_score(
        self,
        image_path: str,
        prompt: str,
        obj: List[str],
        obj_bounding_box: List[List[float]],
        instance_score: torch.Tensor,
    ) -> float:
        """
        Get the specific score for the scorer type (spatial or numeracy).

        Args:
            image_path: Path to the image
            prompt: Text prompt
            obj: List of detected objects
            obj_bounding_box: List of bounding boxes
            instance_score: Detection confidence scores

        Returns:
            Score specific to the scorer type
        """
        pass

    @abstractmethod
    def get_scorer_name(self) -> str:
        """Get the name of the scorer for debug output."""
        pass

    @property
    @abstractmethod
    def scorer_name(self) -> str:
        """Return the name of this scorer."""
        pass

    def create_eval_criteria(self, *args, **kwargs):
        """Spatial/numeracy scorers don't use eval criteria."""
        return None

    @property
    def uses_eval_criteria(self) -> bool:
        """Spatial/numeracy scorers don't use evaluation criteria."""
        return False

    def warmup(self, wait=False) -> None:
        """Warmup the underlying object detection and LLM models."""
        if hasattr(self.object_detector, "warmup"):
            self.object_detector.warmup(wait=wait)
        if hasattr(self.llm, "warmup"):
            self.llm.warmup(wait=wait)

    @abstractmethod
    def get_nonconformity_results(
        self, prompt: str, objs: List[str]
    ) -> Tuple[List[str], bool]:
        """
        Get non-conformity results specific to the scorer type.

        Args:
            prompt: Input prompt
            objs: Detected objects

        Returns:
            Tuple of non-conformity results and correctness flag
        """
        pass

    def eval_metrics(
        self, image_path: str, prompt: str, is_complex: bool = False
    ) -> Tuple[float, List[str], List[str]]:
        """
        Evaluate metrics for the given image and prompt.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for evaluation
            is_complex: Whether to use complex object extraction

        Returns:
            Tuple containing score, detected objects, and normalized objects
        """
        # Extract objects using LLM-based approach
        extracted_objects = self.object_attribute_extractor.extract_objects(prompt)

        # Set filter objects list for debug visualization (replaces old get_obj_from_prompt behavior)
        self.filter_objects_list = extracted_objects

        if self.config.pred_classes == "from_prompt":
            if len(extracted_objects) < 2:
                raise ValueError(
                    "Could not extract two objects from the prompt. Please provide a valid prompt."
                )
            self.object_detector.set_classes(extracted_objects)

        obj_original, obj_bounding_box, instance_score = self.get_bbox_from_img(
            image_path
        )
        log.debug(f"Original detected objects: {obj_original}")

        obj = self.object_attribute_extractor.normalize_detected_objects(
            prompt, obj_original
        )
        log.debug(f"Renamed detected objects: {obj}")

        # For comparison, use only detected objects
        detected_objects_only = [
            self.normalize_object_name(obj_name) for obj_name in obj
        ]

        obj_normalized_for_debug = [
            self.normalize_object_name(obj_name) for obj_name in obj
        ]

        # Filter objects for relevant ones
        obj_filtered, obj_bounding_box_filtered, instance_score_filtered = (
            filter_objects(
                obj_normalized_for_debug,
                obj_bounding_box,
                instance_score,
                self.filter_objects_list,
            )
        )

        score = self.get_specific_score(
            image_path,
            prompt,
            obj_filtered,
            obj_bounding_box_filtered,
            instance_score_filtered,
        )

        # Debug output
        if self.config.debug:

            log.debug(f"Detected objects: {obj_normalized_for_debug}")
            log.debug(f"Bounding boxes: {obj_bounding_box}")
            log.debug(f"Instance scores: {instance_score}")

            debug_img_path = plot_bboxes(
                image_path,
                obj_filtered,
                obj_bounding_box_filtered,
                instance_score_filtered,
                title=f'Prompt: "{prompt}" ({self.get_scorer_name()}: {score:.2f})',
                save_path=f"ssa/thirdparty/compbench/data/examples/debug/{self.get_scorer_name().lower()}/{self.config.od_model.replace('/', '')}/{self.config.pred_classes}/run-{self.current_run_time}/",
            )
            log.debug(f"Debug image saved to: {debug_img_path}")

        return score, detected_objects_only, obj_filtered

    def normalize_object_name(self, obj_name: str) -> str:
        """Normalize object names to standard forms."""
        normalization_map = {
            "ship": "boat",
            "telivision": "tv",
            "goldfish": "fish",
            "painting": "picture",
        }
        return normalization_map.get(obj_name, obj_name)

    def get_bbox_from_img(
        self, image_path: str
    ) -> Tuple[List[str], List[List[float]], torch.Tensor]:
        """
        Detect objects in the image and return bounding box information.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of detected objects, bounding boxes, and confidence scores
        """
        od_output = self.object_detector(image_path)
        obj, obj_bounding_box, instance_score = self.format_preds(od_output)
        return obj, obj_bounding_box, instance_score

    def format_preds(self, od_output):
        """Format object detection predictions."""
        # Process YOLO predictions
        results = []
        obj = []
        for i, result in enumerate(od_output):
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = self.object_detector.model.names[class_id]
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                results.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
                obj.append(class_name)
            obj_bounding_box = [result["bbox"] for result in results]
            instance_score = torch.tensor([result["confidence"] for result in results])

        return obj, obj_bounding_box, instance_score
