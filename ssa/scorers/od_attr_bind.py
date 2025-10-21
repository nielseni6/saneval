import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ssa.od import (
    DEFAULT_OD_VERSION,
    ObjectDetectionModel,
)
from ssa.scorers.scorer_base import BaseScorer, BaseScorerConfig
from ssa.utils.logging import get_log
from ssa.utils.nlp_tools import ObjectAttributeExtractor
from ssa.utils.od_tools import (
    calculate_iou,
    plot_bboxes,
)
from ssa.vlm import DEFAULT_LLM_VERSION, DEFAULT_VLM_VERSION, Llm, Vlm

log = get_log(__file__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OdAttrBindConfig(BaseScorerConfig):
    od_model: str = DEFAULT_OD_VERSION
    vlm_model: str = DEFAULT_VLM_VERSION
    llm_model: str = DEFAULT_LLM_VERSION
    od_classes: str = "data/pred_classes/saneval.json"
    pred_classes: str = "from_json"  # "from_json", "from_prompt", or "unspecified"
    debug: bool = True
    include_explanation: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.1
    min_crop_size: int = 32  # Minimum size for object crops
    generate_criteria: bool = False
    score_threshold: float = (
        0.75  # Minimum score threshold for attribute binding success
    )


class OdAttrBind(BaseScorer):
    """
    Object Detection-based Attribute Binding (ODBAB) evaluation.

    This class evaluates how well a vision-language model can identify
    attributes of specific objects detected in an image by:
    1. Detecting objects using object detection
    2. Cropping individual objects from the image
    3. Using a VLM to evaluate attribute binding on cropped objects
    """

    def __init__(
        self,
        config: OdAttrBindConfig = None,
        config_overrides=None,
        object_detector: ObjectDetectionModel = None,
    ):
        self.current_run_time = time.strftime("%Y%m%d_%H%M%S")

        self.ig_version = "Unknown"
        super().__init__(config or OdAttrBindConfig(), config_overrides)

        # Initialize object detector
        if object_detector:
            self.object_detector = object_detector
        else:
            self.object_detector = ObjectDetectionModel(
                self.config.od_model, self.config.pred_classes, self.config.od_classes
            )

        # Initialize VLM for attribute evaluation
        self.vlm = Vlm(self.config.vlm_model)

        # Initialize LLM for attribute type classification
        self.llm = Llm(self.config.llm_model)

        vlm_config = self.vlm.config()
        vlm_config = self.vlm.config()
        if vlm_batch_size := vlm_config.get("config.batch_size"):
            self.config.batch_size = vlm_batch_size
            log.debug(f"Using VLM-config batch_size: {vlm_batch_size}")
        if "config.include_explanation" in vlm_config:
            self.config.include_explanation = vlm_config["config.include_explanation"]
            log.debug(
                f"Using VLM-config include_explanation : {self.config.include_explanation}"
            )

        # Initialize object attribute extractor
        self.attribute_extractor = ObjectAttributeExtractor(self.llm)

        self.attribute_objects = []

    def get_nonconformity_results(
        self,
        prompt: str,
        object_attributes: Dict[str, List[str]],
        binding_score: float,
        per_object_scores: Dict[str, List[float]] = None,
        score_threshold: float = None,
    ) -> Tuple[List[str], bool]:
        """
        Determine which objects or attributes are missing or non-conforming based on detection and binding success.

        Args:
            prompt: Input prompt
            object_attributes: Dict of expected objects and their attributes
            binding_score: Overall binding score (0.0 to 1.0)
            per_object_scores: Dict mapping object names to their individual attribute scores
            score_threshold: Minimum score threshold for considering binding successful (defaults to config value)

        Returns:
            Tuple of (non_conformity_list, attribute_correctness):
                - non_conformity_list: List of missing/non-conforming items (List[str])
                - attribute_correctness: Boolean indicating if all requirements are met (bool)
        """
        # Use config value if no score_threshold provided
        if score_threshold is None:
            score_threshold = self.config.score_threshold

        non_conformity = []

        if not object_attributes:
            log.warning(
                "No object attributes extracted from the prompt. Please ensure the prompt contains valid object-attribute descriptions."
            )
            return ["No objects detected in prompt"], False

        # Check if we have stored attribute objects from the last evaluation
        per_object_scores = per_object_scores or {}
        detected_objects = list(per_object_scores.keys())

        # Find missing objects (objects in prompt but not detected)
        for expected_object, attributes in object_attributes.items():
            attributes_str = ", ".join(attributes)
            if expected_object not in detected_objects:
                # Add missing object with its attributes to non-conformity
                non_conformity.append(
                    f"Missing object: {expected_object} with attributes [{attributes_str}]"
                )
            # Object was detected, check if attribute binding was successful
            elif expected_object in per_object_scores:
                obj_scores = per_object_scores[expected_object]
                avg_obj_score = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0

                if avg_obj_score < score_threshold:
                    non_conformity.append(
                        f"Poor attribute binding for {expected_object}: expected [{attributes_str}], score: {avg_obj_score:.2f} < {score_threshold}"
                    )

        # Additional check: if overall binding score is too low, add general failure
        if binding_score < score_threshold and not non_conformity:
            # No specific object failures detected, but overall score is low
            non_conformity.append(
                f"Overall attribute binding score too low: {binding_score:.2f} < {score_threshold}"
            )

        # Count correctness: True if no non-conforming items AND binding score meets threshold
        attribute_correctness = (
            len(non_conformity) == 0 and binding_score >= score_threshold
        )

        return non_conformity, attribute_correctness

    def evaluate(
        self, image, input_prompt: str, eval_criteria=None, context=None
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluate object detection-based attribute binding for an image.

        Args:
            image: Image object with 'info' dict containing 'path' or path string
            input_prompt: The prompt to evaluate against the image
            eval_criteria: Not used (for BaseScorer compatibility)
            context: Additional context (optional)

        Returns:
            Tuple containing:
                - attribute_correctness: Boolean indicating if all expected objects were detected
                - binding_score: Float score for attribute binding accuracy
                - non_conformity: List of missing/non-conforming objects or attributes
        """
        # Handle both image paths and image objects (like spatial/numeracy scorers)
        if hasattr(image, "info") and "path" in image.info:
            actual_image_path = image.info["path"]
        else:
            actual_image_path = image

        # Get the binding score
        binding_score = self.eval_attr_binding(
            actual_image_path, input_prompt, eval_criteria
        )

        # Extract expected objects and attributes from prompt for conformity checking
        object_attributes = self.attribute_extractor.extract(input_prompt)

        # Get per-object scores from the last evaluation (stored in eval_attr_binding)
        per_object_scores = getattr(self, "_last_per_object_scores", {})

        # Determine non-conformity and correctness
        non_conformity, attribute_correctness = self.get_nonconformity_results(
            input_prompt, object_attributes, binding_score, per_object_scores
        )

        return attribute_correctness, binding_score, non_conformity

    def eval_attr_binding(
        self, image_path: str, prompt: str, criteria_questions: str = None
    ) -> float:
        """
        Evaluate attribute binding for objects in the image.

        Args:
            image_path: Path to the image file
            prompt: Text prompt describing object attributes
            criteria_questions: Optional criteria questions (will be generated if not provided)

        Returns:
            float: Attribute binding score
        """
        # Extract object and attributes from prompt
        object_attributes = self.attribute_extractor.extract(prompt)

        if not object_attributes:
            log.warning(
                f"Could not extract object and attributes from prompt: {prompt}"
            )
            return 0.0

        # Generate enhanced class names including attributes for better detection
        enhanced_classes = []
        for obj_name, attributes in object_attributes.items():
            # Add the base object name
            enhanced_classes.append(obj_name)

            # Add object-attribute combinations
            for attr in attributes:
                enhanced_classes.append(f"{attr} {obj_name}")

        log.debug(f"Enhanced object classes with attributes: {enhanced_classes}")
        self.object_detector.set_classes(enhanced_classes)

        # Update object detector classes if needed
        if self.config.pred_classes == "from_prompt":
            self.object_detector.set_classes(list(object_attributes.keys()))

        # Get object detections
        obj_names, obj_bboxes, obj_scores = self.get_bbox_from_img(image_path)
        # Filter objects by confidence and other criteria

        if not obj_names:
            log.debug(f"No objects detected in image: {image_path}")
            return 0.0

        # Filter for target object
        target_detections = self.filter_target_objects(
            obj_names, obj_bboxes, obj_scores, list(object_attributes.keys())
        )

        if not target_detections:
            log.debug(f"Target objects '{list(object_attributes.keys())}' not detected")
            return 0.0

        # Evaluate attributes for each detected target object
        # Group scores by object type for proper averaging
        object_scores = {}  # Dict mapping object names to lists of scores
        image = Image.open(image_path)

        for detection in target_detections:
            bbox = detection["bbox"]
            detected_object_name = detection["name"]

            # Find matching object in our extracted attributes
            target_object = None
            attributes = []
            for obj_name, obj_attrs in object_attributes.items():
                if self.object_matches_target(detected_object_name, [obj_name]):
                    target_object = obj_name
                    attributes = obj_attrs
                    break

            if not target_object:
                continue

            # Crop object from image
            cropped_obj = self.crop_object(image, bbox)

            if cropped_obj is None:
                continue

            # Evaluate attributes on cropped object
            attr_score = self.evaluate_attributes_on_crop(
                cropped_obj, target_object, attributes, prompt, criteria_questions
            )

            # Group scores by object type
            if target_object not in object_scores:
                object_scores[target_object] = []
            object_scores[target_object].append(attr_score)

        # Calculate final score using the required formula:
        # 1. Calculate average score per object type from the prompt
        # 2. Sum those averages and divide by total number of object types in the prompt
        total_objects_in_prompt = len(object_attributes)

        if not object_scores or total_objects_in_prompt == 0:
            final_score = 0.0
            all_scores = []
        else:
            # Calculate mean score for each object type that was detected
            all_scores = []  # For debugging output compatibility

            # Initialize scores for all objects in prompt (some may not be detected)
            prompt_object_scores = {}

            for obj_name in object_attributes.keys():
                if obj_name in object_scores:
                    # Calculate mean for detected objects
                    obj_mean = np.mean(object_scores[obj_name])
                    prompt_object_scores[obj_name] = obj_mean
                    all_scores.extend(
                        object_scores[obj_name]
                    )  # Keep all individual scores for debug
                    log.debug(
                        f"Object '{obj_name}': {len(object_scores[obj_name])} detections, mean score: {obj_mean:.3f}"
                    )
                else:
                    # Object from prompt was not detected, score = 0
                    prompt_object_scores[obj_name] = 0.0
                    log.debug(f"Object '{obj_name}': not detected, score: 0.0")

            # Final score: sum of per-object averages divided by total number of objects in prompt
            sum_of_averages = sum(prompt_object_scores.values())
            final_score = sum_of_averages / total_objects_in_prompt
            log.debug(
                f"Final score calculation: sum({list(prompt_object_scores.values())}) / {total_objects_in_prompt} = {final_score:.3f}"
            )

        # Debugging output
        if self.config.debug and all_scores:
            # Pass all targets and attributes for debug output, including object scores
            self.debug_output(
                image_path,
                prompt,
                target_detections,  # Pass target_detections instead of separate lists
                object_attributes,
                object_scores,  # Pass the per-object scores dict
                final_score,
            )

        # Store per-object scores for conformity checking
        self._last_per_object_scores = object_scores

        # Return final score
        self.attribute_objects = list(object_attributes.keys())
        return float(final_score)

    def extract_object_attributes(self, prompt: str) -> Dict[str, List[str]]:
        """
        Extract all objects and their attributes from the prompt using LLM.

        Args:
            prompt: Text prompt

        Returns:
            Dict mapping object names to their list of attributes
        """
        return self.attribute_extractor.extract(prompt)

    def get_bbox_from_img(
        self, image_path: str
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """
        Get object detections from image.

        Returns:
            Tuple of (object_names, bounding_boxes, confidence_scores)
        """
        od_output = self.object_detector(image_path)
        return self.format_preds(od_output)

    def format_preds(
        self, od_output
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """
        Format object detection predictions.

        Returns:
            Tuple of (object_names, bounding_boxes, confidence_scores)
        """
        obj_names = []
        obj_bboxes = []
        obj_scores = []

        for result in od_output:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf.item()

                    # Filter by confidence threshold
                    if confidence < self.config.confidence_threshold:
                        continue

                    class_id = int(box.cls.item())
                    class_name = self.object_detector.model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy.squeeze().tolist()

                    obj_names.append(class_name)
                    obj_bboxes.append([x1, y1, x2, y2])
                    obj_scores.append(confidence)

        return obj_names, obj_bboxes, obj_scores

    def filter_target_objects(
        self,
        obj_names: List[str],
        obj_bboxes: List[List[float]],
        obj_scores: List[float],
        target_objects: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Filter detections to only include the target object and remove duplicates with high overlap.

        Returns:
            List of detection dictionaries
        """
        target_detections = []

        for i, obj_name in enumerate(obj_names):
            # Check if this detection matches the target object
            for target_object in target_objects:
                if self.object_matches_target(obj_name, [target_object]):
                    target_detections.append(
                        {
                            "name": target_object,  # Use the target object name instead of detected name
                            "bbox": obj_bboxes[i],
                            "confidence": obj_scores[i],
                        }
                    )

        # Remove duplicates using IoU calculation (similar to eval_numeracy in spatial_numeracy_eval.py)

        deduplicated_detections = []
        for i, detection in enumerate(target_detections):
            is_duplicate = False
            for j, existing_detection in enumerate(deduplicated_detections):
                # Check if same object type and high IoU overlap
                if detection["name"] == existing_detection["name"] and calculate_iou(
                    detection["bbox"], existing_detection["bbox"]
                ):
                    is_duplicate = True
                    # Keep the detection with higher confidence
                    if detection["confidence"] > existing_detection["confidence"]:
                        deduplicated_detections[j] = detection
                    break

            if not is_duplicate:
                deduplicated_detections.append(detection)

        log.debug(
            f"Filtered {len(target_detections)} detections to {len(deduplicated_detections)} after duplicate removal"
        )
        return deduplicated_detections

    def object_matches_target(
        self, detected_name: str, target_names: List[str]
    ) -> bool:
        """
        Check if detected object name matches any of the target object names.
        Handles variations and synonyms.
        """
        detected_name = detected_name.lower()

        for target_name in target_names:
            target_name = target_name.lower()

            # Direct match
            if detected_name == target_name:
                return True

            # Partial match (one contains the other)
            if target_name in detected_name or detected_name in target_name:
                return True

            # Handle common synonyms
            synonyms = {
                "person": ["man", "woman", "boy", "girl", "human"],
                "car": ["vehicle", "automobile"],
                "bike": ["bicycle", "motorcycle"],
                "tv": ["television"],
                "phone": ["cellphone", "smartphone"],
            }

            for canonical, synonym_list in synonyms.items():
                if (detected_name == canonical and target_name in synonym_list) or (
                    target_name == canonical and detected_name in synonym_list
                ):
                    return True

        return False

    def crop_object(
        self, image: Image.Image, bbox: List[float]
    ) -> Optional[Image.Image]:
        """
        Crop object from image using bounding box.

        Args:
            image: PIL Image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Cropped PIL Image or None if crop is too small
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        # Check if crop is large enough
        if (x2 - x1) < self.config.min_crop_size or (
            y2 - y1
        ) < self.config.min_crop_size:
            log.debug(f"Crop too small: {x2-x1}x{y2-y1}")
            return None

        # Crop the image
        cropped = image.crop((x1, y1, x2, y2))

        # Save cropped object to debug folder if debug is enabled
        if (
            self.config.debug
            and hasattr(self, "_save_cropped_images")
            and self._save_cropped_images
        ):
            self._save_cropped_object_debug(cropped, bbox)

        return cropped

    def _save_cropped_object_debug(self, cropped_image: Image.Image, bbox: List[float]):
        """
        Save cropped object image to debug folder using the same structured path as other debug images.

        Args:
            cropped_image: The cropped PIL Image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        """
        import os
        import time

        try:
            # Use the same structured debug directory as regular debug images
            debug_dir = f"data/debug/scorers/od-attr-bind/{self.config.od_model.replace('/', '')}/{self.config.pred_classes}/{self.ig_version.replace('/', '')}/run-{self.current_run_time}/"
            os.makedirs(debug_dir, exist_ok=True)

            # Generate unique filename with timestamp and bbox info
            timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
            x1, y1, x2, y2 = map(int, bbox)
            filename = f"cropped_obj_{timestamp}_{x1}_{y1}_{x2}_{y2}.png"
            save_path = os.path.join(debug_dir, filename)

            # Save the cropped image
            cropped_image.save(save_path)
            log.info(f"Cropped object saved to: {save_path}")

        except Exception as e:
            log.error(f"Failed to save cropped object debug image: {e}")

    def evaluate_attributes_on_crop(
        self,
        cropped_image: Image.Image,
        object_name: str,
        attributes: List[str],
        original_prompt: str,
        criteria_questions: List[str] = None,
    ) -> float:
        """
        Evaluate whether the cropped object has the specified attributes.

        Args:
            cropped_image: Cropped object image
            object_name: Name of the object
            attributes: List of attributes to check
            original_prompt: Original prompt for context
            criteria_questions: List of criteria questions from the prompt

        Returns:
            Attribute binding score between 0 and 1
        """
        if not attributes:
            return 1.0  # No attributes to check

        scores = []

        for attribute in attributes:
            # Create an open-ended question about this attribute type
            if not self.config.generate_criteria and criteria_questions:
                question = self.select_corpus_question(
                    attribute, object_name, criteria_questions
                )
            else:
                question = None

            if question is None or self.config.generate_criteria:
                # Generate criteria using LLM
                question = self.create_attribute_question(attribute, object_name)

            try:
                # Use VLM to answer the question
                response = self.vlm.call(question, image=cropped_image)

                # Score based on how well the response matches the expected attribute
                score = self.score_attribute_match(response, attribute)
                scores.append(score)

                log.debug(
                    f"Attribute '{attribute}' for {object_name}: Q: '{question}' A: '{response}' -> {score}"
                )

            except Exception as e:
                log.error(f"Error evaluating attribute '{attribute}': {e}")
                scores.append(0.0)

        # Return average score across all attributes
        return np.mean(scores) if scores else 0.0

    def create_attribute_question(self, attribute: str, object_name: str) -> str:
        """
        Create an appropriate open-ended question for the given attribute type using LLM classification.

        Args:
            attribute: The attribute to ask about
            object_name: The name of the object

        Returns:
            Question string
        """
        # Use LLM to classify the attribute type
        attribute_type = self.classify_attribute_type(attribute)

        # Generate question based on LLM-determined type
        if attribute_type == "color":
            return f"What color is this {object_name}?"
        elif attribute_type == "size":
            return f"What size is this {object_name}?"
        elif attribute_type == "material":
            return f"What material is this {object_name} made of?"
        elif attribute_type == "shape":
            return f"What shape is this {object_name}?"
        elif attribute_type == "texture":
            return f"What is the texture of this {object_name}?"
        elif attribute_type == "pattern":
            return f"What pattern does this {object_name} have?"
        elif attribute_type == "state":
            return f"What is the state or condition of this {object_name}?"
        elif attribute_type == "position":
            return f"What is the position or orientation of this {object_name}?"
        else:
            # General attribute question for unknown types
            return f"Describe the appearance of this {object_name}. List some of its attributes?"

    def select_corpus_question(
        self, attribute: str, object_name: str, criteria_questions: list[str]
    ) -> str:
        """
        Select a pre-stored question from the corpus based on the attribute type.

        Args:
            attribute: The attribute to ask about
            object_name: The name of the object
        """
        # Use LLM to classify the attribute type

        attribute_type = self.classify_attribute_type(attribute)
        for q in criteria_questions:
            if attribute_type in q.lower() and object_name.lower() in q.lower():
                question = q
                log.debug(f"Selected corpus question: {question}")
                return question

        # If no specific question found, return a generic one
        log.debug(
            f"No specific corpus question found for {attribute} of {object_name}, using generic question"
        )
        return None

    def classify_attribute_type(self, attribute: str) -> str:
        """
        Use LLM to classify the type of attribute.

        Args:
            attribute: The attribute word to classify

        Returns:
            Attribute type category
        """
        prompt = f"""Classify the following attribute word into one of these categories:
- color: describes color or hue (e.g., red, blue, dark, bright)
- size: describes size, scale, or magnitude (e.g., large, small, tiny, huge)
- material: describes what something is made of (e.g., wooden, metal, plastic, glass)
- shape: describes form or geometric properties (e.g., round, square, curved, angular)
- texture: describes surface feel or appearance (e.g., smooth, rough, soft, bumpy)
- pattern: describes visual patterns or designs (e.g., striped, spotted, checkered)
- state: describes condition, status, or temporary properties (e.g., broken, new, old, clean)
- position: describes spatial orientation or pose (e.g., upright, tilted, horizontal)
- other: for attributes that don't fit the above categories

Attribute word: "{attribute}"

Respond with only the category name (e.g., "color", "size", etc.)."""

        try:
            response = self.llm.call(prompt)
            # Extract the category from the response
            category = response.strip().lower()

            # Validate the category
            valid_categories = [
                "color",
                "size",
                "material",
                "shape",
                "texture",
                "pattern",
                "state",
                "position",
                "other",
            ]
            if category in valid_categories:
                log.debug(f"LLM classified attribute '{attribute}' as '{category}'")
                return category
            else:
                # If LLM returns invalid category, fall back to "other"
                log.debug(
                    f"LLM returned invalid category '{category}' for attribute '{attribute}', using 'other'"
                )
                return "other"

        except Exception as e:
            log.error(f"Error classifying attribute '{attribute}' with LLM: {e}")
            return "other"

    def score_attribute_match(self, response: str, expected_attribute: str) -> float:
        """
        Score how well the VLM response matches the expected attribute using LLM evaluation.

        Args:
            response: VLM response text
            expected_attribute: The expected attribute

        Returns:
            Score: 1.0 for exact match, 0.5 for close match, 0.0 for no match
        """
        response_lower = response.lower().strip()
        expected_lower = expected_attribute.lower().strip()

        # Quick exact match check first
        if expected_lower in response_lower:
            return 1.0

        # Use LLM to evaluate semantic similarity and match quality
        return self.evaluate_attribute_match_with_llm(response, expected_attribute)

    def evaluate_attribute_match_with_llm(
        self, response: str, expected_attribute: str
    ) -> float:
        """
        Use LLM to evaluate how well the response matches the expected attribute.

        Args:
            response: VLM response describing the object
            expected_attribute: The attribute we're looking for

        Returns:
            Score between 0.0 and 1.0
        """
        prompt = f"""You are evaluating whether a description matches a specific attribute.

Expected attribute: "{expected_attribute}"
Description: "{response}"

Rate how well the description matches the expected attribute on a scale from 0.0 to 1.0:
- 1.0: Perfect match (the description clearly contains or describes the expected attribute)
- 0.8-0.9: Very good match (the description strongly suggests the expected attribute)
- 0.5-0.7: Partial match (the description somewhat relates to the expected attribute)
- 0.2-0.4: Weak match (there might be some connection but it's unclear)
- 0.0-0.1: No match (the description doesn't relate to the expected attribute at all)

Consider semantic similarity, synonyms, and related concepts. For example:
- "red" matches "crimson" or "scarlet" (0.8-0.9)
- "large" matches "big" or "huge" (0.8-0.9)
- "wooden" matches "made of wood" (1.0)
- "smooth" might partially match "polished" (0.6-0.7)

Respond with only a number between 0.0 and 1.0 (e.g., "0.8")."""

        try:
            response_text = self.llm.call(prompt)
            # Extract the numeric score from the response
            score_str = response_text.strip()

            # Try to parse the score
            try:
                score = float(score_str)
                # Ensure score is within valid range
                if 0.0 <= score <= 1.0:
                    log.debug(
                        f"LLM scored attribute match '{expected_attribute}' vs '{response}': {score}"
                    )
                    return score
                else:
                    log.debug(
                        f"LLM returned out-of-range score {score}, clamping to [0,1]"
                    )
                    return max(0.0, min(1.0, score))
            except ValueError:
                # If we can't parse as float, try to extract number from text
                import re

                numbers = re.findall(r"\d+\.?\d*", score_str)
                if numbers:
                    try:
                        score = float(numbers[0])
                        score = max(0.0, min(1.0, score))
                        log.debug(f"LLM extracted score {score} from '{score_str}'")
                        return score
                    except ValueError:
                        pass

                log.debug(f"Could not parse LLM score '{score_str}', using fallback")
                return self.fallback_attribute_match(response, expected_attribute)

        except Exception as e:
            log.error(f"Error evaluating attribute match with LLM: {e}")
            return self.fallback_attribute_match(response, expected_attribute)

    def fallback_attribute_match(self, response: str, expected_attribute: str) -> float:
        """
        Fallback method for attribute matching when LLM fails.
        Uses simple string matching as backup.

        Args:
            response: VLM response text
            expected_attribute: The expected attribute

        Returns:
            Score between 0.0 and 1.0
        """
        response_lower = response.lower().strip()
        expected_lower = expected_attribute.lower().strip()

        # Check for partial matches (at least 3 characters)
        if len(expected_lower) >= 3:
            if (
                expected_lower[:3] in response_lower
                or expected_lower[-3:] in response_lower
            ):
                return 0.5

        # Check for similar words (basic similarity)
        words_in_response = response_lower.split()
        for word in words_in_response:
            if len(word) >= 3 and len(expected_lower) >= 3:
                # Simple similarity check
                if (word in expected_lower or expected_lower in word) and len(word) > 2:
                    return 0.5

        return 0.0

    def debug_output(
        self,
        image_path: str,
        prompt: str,
        target_detections: List[Dict[str, Any]],  # List of detection dictionaries
        object_attributes: Dict[str, List[str]],
        per_object_scores: Dict[str, List[float]],  # Per-object scores dict
        final_score: float,  # Final calculated score
    ):
        """
        Generate debug output including visualizations for all targets and attributes with individual scores.
        """
        log.debug(f"Image: {image_path}")
        log.debug(f"Prompt: {prompt}")
        log.debug(f"All target objects and attributes: {object_attributes}")
        log.debug(f"Detected objects: {[d['name'] for d in target_detections]}")
        log.debug(f"Per-object scores: {per_object_scores}")
        log.debug(f"Final score: {final_score}")

        if target_detections:
            # Create detailed title showing objects, their attributes, and attribute scores
            target_details = []
            for obj_name, attributes in object_attributes.items():
                if obj_name in per_object_scores:
                    obj_mean = np.mean(per_object_scores[obj_name])
                    # Format as: object[attr1,attr2]:score
                    attrs_str = ",".join(attributes)
                    target_details.append(f"{obj_name}[{attrs_str}]:{obj_mean:.2f}")
                else:
                    # Object not detected
                    attrs_str = ",".join(attributes)
                    target_details.append(f"{obj_name}[{attrs_str}]:0.0")

            targets_str = " | ".join(target_details)

            debug_img_path = plot_bboxes(
                image_path,
                [d["name"] for d in target_detections],
                [d["bbox"] for d in target_detections],
                [d["confidence"] for d in target_detections],
                title=f'Prompt: "{prompt}" | {targets_str} | Final: {final_score:.2f}',
                save_path=f"data/debug/scorers/od-attr-bind/",
            )
            log.debug(f"Debug image saved to: {debug_img_path}")

    def eval_full_directory(self, imagedir: str, outfile: str):
        """
        Evaluate all images in a directory (placeholder for batch processing).
        """
        # This would be implemented for batch evaluation
        # Similar to SANEval.eval_full_directory
        pass

    def create_eval_criteria(self, *args, **kwargs):
        """OdAttrBind doesn't use eval criteria."""
        return None

    @property
    def uses_eval_criteria(self) -> bool:
        """OdAttrBind doesn't use evaluation criteria."""
        return False

    def warmup(self, wait=False) -> None:
        """Warmup the underlying models."""
        if hasattr(self.object_detector, "warmup"):
            self.object_detector.warmup(wait=wait)
        if hasattr(self.vlm, "warmup"):
            self.vlm.warmup(wait=wait)
        if hasattr(self.llm, "warmup"):
            self.llm.warmup(wait=wait)

    @property
    def scorer_name(self) -> str:
        """Return the name of this scorer."""
        return "od-attr-binding"
