# Object Detection (od) model bank
# This leverages different object detection models
# to identify and localize objects in images

# Object Detection models
#  - yolov11      : YOLOv11 for general object detection
#  - yolov12      : YOLOv12 for improved detection capabilities
#  - yoloworld    : YOLOWorld for world-centric object detection
#  - yoloe        : YOLOE for everything detection

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ssa.exceptions import InvalidConfigurationError, ValidationError

YOLOV11 = "yolo/v11"
YOLOV12 = "yolo/v12"
YOLOWORLD = "yolo/world"
YOLOEVERYTHING = "yolo/e"  # YOLOE for everything detection

DETECTION_MODELS = [
    YOLOV11,
    YOLOV12,
    YOLOWORLD,
    YOLOEVERYTHING,
]
DEFAULT_OD_VERSION = YOLOEVERYTHING

# Global dictionary mapping model types to their import functions
MODEL_TYPE_IMPORTS = {
    YOLOV11: lambda: __import__("ultralytics", fromlist=["YOLO"]).YOLO,
    YOLOV12: lambda: __import__("ultralytics", fromlist=["YOLO"]).YOLO,
    YOLOWORLD: lambda: __import__("ultralytics", fromlist=["YOLOWorld"]).YOLOWorld,
    YOLOEVERYTHING: lambda: __import__("ultralytics", fromlist=["YOLOE"]).YOLOE,
}


def _get_yolo_class(model_type: str) -> Type:
    """Lazy import function for YOLO classes to avoid import-time warnings."""
    if model_type not in MODEL_TYPE_IMPORTS:
        raise InvalidConfigurationError(
            f"Unknown model type '{model_type}'. "
            f"Supported types: {', '.join(MODEL_TYPE_IMPORTS.keys())}"
        )

    return MODEL_TYPE_IMPORTS[model_type]()


# Model configuration mapping - acts as a switch statement
MODEL_CONFIGS = {
    YOLOV11: {
        "model_class": lambda: _get_yolo_class(YOLOV11),
        "weights": "yolo11n.pt",
        "supports_class_setting": False,
        "supports_unspecified": False,
    },
    YOLOV12: {
        "model_class": lambda: _get_yolo_class(YOLOV12),
        "weights": "yolo12x.pt",
        "supports_class_setting": False,
        "supports_unspecified": False,
    },
    YOLOWORLD: {
        "model_class": lambda: _get_yolo_class(YOLOWORLD),
        "weights": "yolov8x-worldv2.pt",
        "supports_class_setting": True,
        "supports_unspecified": False,
        "set_classes_on_init": True,
    },
    YOLOEVERYTHING: {
        "model_class": lambda: _get_yolo_class(YOLOEVERYTHING),
        "weights": "yoloe-11l-seg.pt",
        "supports_class_setting": True,
        "supports_unspecified": True,
        "requires_text_pe": True,
    },
}


class ObjectDetectionModel:
    """
    Object Detection model wrapper supporting multiple YOLO variants.

    This class provides a unified interface for different YOLO-based object detection models,
    handling model initialization, class configuration, and inference.

    Args:
        key: Model identifier (e.g., 'yolo/v11', 'yolo/world', 'yolo/e')
        pred_classes: How to configure prediction classes. Valid options:
            - "from_json" (default): Load classes from JSON file
            - "unspecified": Use open-vocabulary prediction (YOLO-E only)
        json_file_path: Optional path to custom JSON file containing class definitions.
            If None, uses default path: ../data/pred_classes/saneval.json

    Raises:
        InvalidConfigurationError: If key is not a recognized model type
        ValidationError: If pred_classes has an invalid value
        FileNotFoundError: If json_file_path is specified but doesn't exist

    Example:
        >>> model = ObjectDetectionModel("yolo/e", pred_classes="from_json")
        >>> results = model(image_path)
    """

    # Valid values for pred_classes parameter
    VALID_PRED_CLASSES = {"from_json", "unspecified"}

    def __init__(
        self,
        key: str,
        pred_classes: str = "from_json",
        json_file_path: Optional[str] = None,
    ) -> None:
        # Validate model key
        if key not in MODEL_CONFIGS:
            raise InvalidConfigurationError(
                f"Unrecognized model key '{key}'. "
                f"Valid options are: {', '.join(MODEL_CONFIGS.keys())}"
            )

        # Validate pred_classes parameter
        if pred_classes not in self.VALID_PRED_CLASSES:
            raise ValidationError(
                f"Invalid pred_classes value '{pred_classes}'. "
                f"Valid options are: {', '.join(self.VALID_PRED_CLASSES)}"
            )

        # Validate json_file_path if provided
        if json_file_path is not None and not os.path.exists(json_file_path):
            raise FileNotFoundError(
                f"Specified json_file_path does not exist: {json_file_path}"
            )

        self.key: str = key
        self.config: Dict[str, Any] = MODEL_CONFIGS[key]
        self.categories_file_path: str = ""
        self._setup_categories_path(json_file_path)
        self.model: Any = self._create_model()
        self._configure_classes(pred_classes)

    def _setup_categories_path(self, json_file_path: Optional[str]) -> None:
        """Setup the path to the categories JSON file."""
        if json_file_path is None:
            current_script_path = os.path.dirname(os.path.abspath(__file__))
            self.categories_file_path = os.path.join(
                current_script_path, "..", "data", "pred_classes", "saneval.json"
            )
        else:
            self.categories_file_path = json_file_path

    def _create_model(self) -> Any:
        """Create model instance using switch-like dispatch."""
        model_class_getter = self.config["model_class"]
        model_class = model_class_getter()  # Call the lambda to get the actual class
        weights = self.config["weights"]
        return model_class(weights)

    def _configure_classes(self, pred_classes: str) -> None:
        """Configure model classes based on model type and prediction settings."""
        # Validate unspecified setting - only supported by YOLOE
        if pred_classes == "unspecified" and not self.config.get(
            "supports_unspecified", False
        ):
            raise ValidationError(
                f"Open-vocabulary prediction ('unspecified') is only supported by the "
                f"{YOLOEVERYTHING} model. Model '{self.key}' requires pred_classes='from_json'."
            )

        # Configure classes using switch-like dispatch
        class_handlers = {
            YOLOWORLD: self._setup_yoloworld_classes,
            YOLOEVERYTHING: lambda: self._setup_yoloeverything_classes(pred_classes),
        }

        handler = class_handlers.get(self.key)
        if handler:
            handler()

    def _setup_yoloworld_classes(self) -> None:
        """Setup classes for YOLO World model."""
        classes = self.get_classes()
        if not classes:
            logging.warning(
                f"No classes found for model '{self.key}'. Skipping class configuration. "
                f"Model will use default classes or may not work properly."
            )
            return
        self.model.set_classes(classes)

    def _setup_yoloeverything_classes(self, pred_classes: str) -> None:
        """Setup classes for YOLO Everything model."""
        if pred_classes != "unspecified":
            classes = self.get_classes()
            if not classes:
                logging.warning(
                    f"No classes found for model '{self.key}'. Skipping class configuration. "
                    f"Model will use default classes or may not work properly."
                )
                return
            self.model.set_classes(classes, self.model.get_text_pe(classes))

    def __call__(self, image: Union[str, Any]) -> Any:
        """
        Detect objects in the given image using the specified model.

        Args:
            image: Path to the image file or an image array.

        Returns:
            List of detected objects with bounding boxes and labels.
        """
        return self.model(image)

    def set_classes(self, classes: List[str]) -> None:
        """
        Set the categories of objects that the model can detect.

        Args:
            classes: List of category names to set for the model.
        """
        if not self.config.get("supports_class_setting", False):
            logging.warning(f"Setting classes is not supported for model '{self.key}'.")
            return

        # Validate classes before setting
        if not classes:
            logging.warning(
                f"Cannot set classes for model '{self.key}': empty class list provided."
            )
            return

        # Switch-like dispatch for setting classes
        class_setters = {
            YOLOWORLD: lambda: self.model.set_classes(classes),
            YOLOEVERYTHING: lambda: self.model.set_classes(
                classes, self.model.get_text_pe(classes)
            ),
        }

        setter = class_setters.get(self.key)
        if setter:
            setter()

    def get_classes(self) -> List[str]:
        """
        Get the categories of objects that the model can detect.

        Returns:
            List of category names.
        """
        try:
            with open(self.categories_file_path, "r") as f:
                data = json.load(f)
            category_names = data.get("categories", [])
            if not category_names:
                logging.warning(
                    f"No categories found in '{self.categories_file_path}' "
                    f"or key 'categories' is missing/empty."
                )
                return []
        except FileNotFoundError:
            logging.error(f"Categories file not found: '{self.categories_file_path}'")
            return []
        except json.JSONDecodeError:
            logging.error(
                f"Invalid JSON in categories file: '{self.categories_file_path}'"
            )
            return []

        return category_names
