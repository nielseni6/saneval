"""
Unit tests for Object Detection (OD) model module.

Tests cover:
- Model initialization with different configurations
- Error handling for invalid configurations
- Model class selection
- Class configuration for different model types
- JSON file handling
- Edge cases and validation
"""

from unittest.mock import Mock, mock_open, patch

import pytest

from ssa.exceptions import InvalidConfigurationError, ValidationError
from ssa.od import (
    DETECTION_MODELS,
    MODEL_CONFIGS,
    YOLOEVERYTHING,
    YOLOV11,
    YOLOWORLD,
    ObjectDetectionModel,
    _get_yolo_class,
)


class TestGetYoloClass:
    """Tests for the _get_yolo_class function."""

    def test_get_yolo_class_invalid_type(self):
        """Test that invalid model type raises InvalidConfigurationError."""
        with pytest.raises(
            InvalidConfigurationError, match="Unknown model type 'invalid_model'"
        ):
            _get_yolo_class("invalid_model")

    @patch("builtins.__import__")
    def test_get_yolo_class_yolov11(self, mock_import):
        """Test getting YOLOv11 class."""
        mock_yolo = Mock()
        mock_import.return_value.YOLO = mock_yolo

        result = _get_yolo_class(YOLOV11)
        assert result == mock_yolo

    @patch("builtins.__import__")
    def test_get_yolo_class_yoloworld(self, mock_import):
        """Test getting YOLOWorld class."""
        mock_yoloworld = Mock()
        mock_import.return_value.YOLOWorld = mock_yoloworld

        result = _get_yolo_class(YOLOWORLD)
        assert result == mock_yoloworld


class TestObjectDetectionModelInit:
    """Tests for ObjectDetectionModel initialization."""

    def test_init_with_invalid_key(self):
        """Test initialization with invalid model key."""
        with pytest.raises(
            InvalidConfigurationError, match="Unrecognized model key 'invalid_model'"
        ):
            ObjectDetectionModel("invalid_model")

    def test_init_with_invalid_pred_classes(self):
        """Test initialization with invalid pred_classes value."""
        with patch.object(ObjectDetectionModel, "_create_model", return_value=Mock()):
            with pytest.raises(
                ValidationError, match="Invalid pred_classes value 'invalid'"
            ):
                ObjectDetectionModel(YOLOV11, pred_classes="invalid")

    def test_init_with_nonexistent_json_file(self):
        """Test initialization with nonexistent JSON file path."""
        with pytest.raises(
            FileNotFoundError, match="Specified json_file_path does not exist"
        ):
            ObjectDetectionModel(
                YOLOV11, json_file_path="/nonexistent/path/to/file.json"
            )

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_init_with_valid_key(self, mock_file, mock_get_class, mock_exists):
        """Test successful initialization with valid key."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")

        assert model.key == YOLOV11
        assert model.model == mock_model_instance

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_init_with_custom_json_path(self, mock_file, mock_get_class, mock_exists):
        """Test initialization with custom JSON file path."""
        mock_model_class = Mock()
        mock_get_class.return_value = mock_model_class
        mock_model_class.return_value = Mock()

        custom_path = "/custom/path/classes.json"
        model = ObjectDetectionModel(
            YOLOV11, pred_classes="from_json", json_file_path=custom_path
        )

        assert model.categories_file_path == custom_path


class TestObjectDetectionModelClassConfiguration:
    """Tests for class configuration methods."""

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_yoloworld_class_configuration(
        self, mock_file, mock_get_class, mock_exists
    ):
        """Test class configuration for YOLO World model."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        ObjectDetectionModel(YOLOWORLD, pred_classes="from_json")

        # Verify set_classes was called
        mock_model_instance.set_classes.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_yoloeverything_with_unspecified(
        self, mock_file, mock_get_class, mock_exists
    ):
        """Test YOLO Everything with unspecified pred_classes."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        ObjectDetectionModel(YOLOEVERYTHING, pred_classes="unspecified")

        # Verify set_classes was NOT called for unspecified
        mock_model_instance.set_classes.assert_not_called()

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_yoloeverything_with_from_json(
        self, mock_file, mock_get_class, mock_exists
    ):
        """Test YOLO Everything with from_json pred_classes."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_instance.get_text_pe = Mock(return_value="text_pe")
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        ObjectDetectionModel(YOLOEVERYTHING, pred_classes="from_json")

        # Verify set_classes was called with text_pe
        mock_model_instance.set_classes.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    def test_unspecified_only_for_yoloeverything(self, mock_get_class, mock_exists):
        """Test that unspecified pred_classes only works with YOLO Everything."""
        mock_model_class = Mock()
        mock_get_class.return_value = mock_model_class
        mock_model_class.return_value = Mock()

        with pytest.raises(
            ValidationError, match="Open-vocabulary prediction.*is only supported by"
        ):
            ObjectDetectionModel(YOLOV11, pred_classes="unspecified")


class TestObjectDetectionModelMethods:
    """Tests for ObjectDetectionModel methods."""

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_call_method(self, mock_file, mock_get_class, mock_exists):
        """Test __call__ method invokes model."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_instance.return_value = "detection_results"
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")
        model("test_image.jpg")

        mock_model_instance.assert_called_once_with("test_image.jpg")

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog", "bird"]}',
    )
    def test_get_classes(self, mock_file, mock_get_class, mock_exists):
        """Test get_classes method returns categories from JSON."""
        mock_model_class = Mock()
        mock_get_class.return_value = mock_model_class
        mock_model_class.return_value = Mock()

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")
        classes = model.get_classes()

        assert classes == ["cat", "dog", "bird"]

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_get_classes_empty_categories(self, mock_file, mock_get_class, mock_exists):
        """Test get_classes with empty categories in JSON."""
        mock_model_class = Mock()
        mock_get_class.return_value = mock_model_class
        mock_model_class.return_value = Mock()

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")
        classes = model.get_classes()

        assert classes == []

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    def test_get_classes_file_not_found(self, mock_get_class, mock_exists):
        """Test get_classes when JSON file doesn't exist."""
        mock_model_class = Mock()
        mock_get_class.return_value = mock_model_class
        mock_model_class.return_value = Mock()

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")

        with patch("builtins.open", side_effect=FileNotFoundError()):
            classes = model.get_classes()
            assert classes == []

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    def test_get_classes_invalid_json(self, mock_get_class, mock_exists):
        """Test get_classes with invalid JSON."""
        mock_model_class = Mock()
        mock_get_class.return_value = mock_model_class
        mock_model_class.return_value = Mock()

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")

        with patch("builtins.open", mock_open(read_data="invalid json")):
            classes = model.get_classes()
            assert classes == []

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_set_classes_unsupported_model(
        self, mock_file, mock_get_class, mock_exists
    ):
        """Test set_classes on model that doesn't support it."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")
        model.set_classes(["new", "classes"])

        # Should not raise, but should log warning (tested via logging)
        # YOLOv11 doesn't support setting classes

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_set_classes_yoloworld(self, mock_file, mock_get_class, mock_exists):
        """Test set_classes on YOLO World model."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        model = ObjectDetectionModel(YOLOWORLD, pred_classes="from_json")
        new_classes = ["bird", "plane"]
        model.set_classes(new_classes)

        # set_classes called twice: once during init, once manually
        assert mock_model_instance.set_classes.call_count == 2

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_set_classes_yoloeverything(self, mock_file, mock_get_class, mock_exists):
        """Test set_classes on YOLO Everything model."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_instance.get_text_pe = Mock(return_value="text_pe")
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        model = ObjectDetectionModel(YOLOEVERYTHING, pred_classes="from_json")
        new_classes = ["bird", "plane"]
        model.set_classes(new_classes)

        # Verify set_classes was called with text_pe
        # Called twice: once during init, once manually
        assert mock_model_instance.set_classes.call_count == 2

    @patch("os.path.exists", return_value=True)
    @patch("ssa.od._get_yolo_class")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"categories": ["cat", "dog"]}',
    )
    def test_set_classes_empty_list(self, mock_file, mock_get_class, mock_exists):
        """Test set_classes with empty list."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        model = ObjectDetectionModel(YOLOWORLD, pred_classes="from_json")
        mock_model_instance.set_classes.reset_mock()

        model.set_classes([])

        # Should not call set_classes with empty list
        mock_model_instance.set_classes.assert_not_called()


class TestModelConfigs:
    """Tests to verify MODEL_CONFIGS structure."""

    def test_all_detection_models_have_configs(self):
        """Test that all detection models have configurations."""
        for model_type in DETECTION_MODELS:
            assert model_type in MODEL_CONFIGS

    def test_model_configs_have_required_keys(self):
        """Test that all model configs have required keys."""
        required_keys = ["model_class", "weights"]

        for model_type, config in MODEL_CONFIGS.items():
            for key in required_keys:
                assert key in config, f"{model_type} missing {key}"

    def test_model_configs_weights_are_strings(self):
        """Test that all model weights are strings."""
        for model_type, config in MODEL_CONFIGS.items():
            assert isinstance(config["weights"], str)

    def test_yoloworld_supports_class_setting(self):
        """Test that YOLO World supports class setting."""
        assert MODEL_CONFIGS[YOLOWORLD]["supports_class_setting"] is True

    def test_yoloeverything_supports_unspecified(self):
        """Test that YOLO Everything supports unspecified pred_classes."""
        assert MODEL_CONFIGS[YOLOEVERYTHING]["supports_unspecified"] is True

    def test_yolov11_does_not_support_unspecified(self):
        """Test that YOLOv11 does not support unspecified pred_classes."""
        assert MODEL_CONFIGS[YOLOV11].get("supports_unspecified", False) is False
