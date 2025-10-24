"""
Integration tests for scorer components.

These tests verify the integration between scorers, object detection,
and LLM components working together.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from ssa.od import ObjectDetectionModel
from ssa.scorers.numeracy import NumeracyConfig, NumeracyScorer
from ssa.scorers.spatial import SpatialConfig, SpatialScorer


@pytest.fixture(autouse=True)
def mock_secrets():
    """Mock secrets retrieval to prevent API key lookups."""
    with patch("ssa.utils.secrets.get_secret", return_value="mock_api_key"):
        yield


@pytest.fixture(autouse=True)
def mock_genai_client():
    """Mock Google GenAI client to prevent real API connections."""
    mock_client = Mock()
    mock_client._api_client = Mock()
    mock_client.close = Mock()
    with patch("google.genai.Client", return_value=mock_client):
        yield


@pytest.fixture
def mock_provider_factory():
    """Create a mock provider factory for LLM/VLM testing."""

    def create_mock_provider(version="test-model-1.0"):
        """Create a properly configured mock provider."""
        mock_provider_class = Mock()
        mock_instance = Mock()
        mock_instance.version = version
        mock_instance.config = {"test": "config"}
        mock_instance.call = Mock(return_value="test response")
        mock_instance.warmup = Mock()
        mock_provider_class.return_value = mock_instance
        return mock_provider_class

    return create_mock_provider


@pytest.mark.integration
class TestNumeracyScorerIntegration:
    """Integration tests for NumeracyScorer with dependencies."""

    @patch("ssa.vlm.Llm")
    @patch("ssa.od.ObjectDetectionModel")
    def test_numeracy_scorer_full_pipeline(self, mock_od, mock_llm):
        """Test complete numeracy scoring pipeline."""
        # Setup mock object detector
        mock_detector_instance = Mock()
        mock_detector_instance.model = Mock()
        mock_detector_instance.model.names = {0: "cat", 1: "dog"}

        # Mock detection results
        mock_box1 = Mock()
        mock_box1.cls = Mock()
        mock_box1.cls.item.return_value = 0
        mock_box1.conf = Mock()
        mock_box1.conf.item.return_value = 0.95
        mock_box1.xyxy = Mock()
        mock_box1.xyxy.squeeze.return_value.tolist.return_value = [0, 0, 10, 10]

        mock_box2 = Mock()
        mock_box2.cls = Mock()
        mock_box2.cls.item.return_value = 0
        mock_box2.conf = Mock()
        mock_box2.conf.item.return_value = 0.92
        mock_box2.xyxy = Mock()
        mock_box2.xyxy.squeeze.return_value.tolist.return_value = [20, 20, 30, 30]

        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]

        mock_detector_instance.return_value = [mock_result]
        mock_od.return_value = mock_detector_instance

        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create scorer
        scorer = NumeracyScorer(config=NumeracyConfig(debug=False))

        # Replace scorer's object detector with our mock
        scorer.object_detector = mock_detector_instance

        # Mock the object attribute extractor methods
        scorer.object_attribute_extractor.extract_objects = Mock(return_value=["cat"])
        scorer.object_attribute_extractor.normalize_detected_objects = Mock(
            return_value=["cat", "cat"]
        )
        scorer.object_attribute_extractor.extract_numeracy_relationship = Mock(
            return_value={"objects": {"cat": 2}}
        )

        # Create test image
        test_image = Mock()
        test_image.info = {"path": "/path/to/test.jpg"}

        # Evaluate
        correct, score, non_conformity = scorer.evaluate(
            test_image, "2 cats in the image"
        )

        # Verify results
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(non_conformity, list)
        assert isinstance(correct, bool)

    @patch("ssa.vlm.Llm")
    @patch("ssa.od.ObjectDetectionModel")
    def test_numeracy_scorer_with_missing_objects(self, mock_od, mock_llm):
        """Test numeracy scorer when expected objects are missing."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.model = Mock()
        mock_detector_instance.model.names = {}
        mock_result = Mock()
        mock_result.boxes = []
        mock_detector_instance.return_value = [mock_result]
        mock_od.return_value = mock_detector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create scorer
        scorer = NumeracyScorer(config=NumeracyConfig(debug=False))

        # Replace scorer's object detector with our mock
        scorer.object_detector = mock_detector_instance

        # Mock extractor to return expected objects but no detections
        scorer.object_attribute_extractor.extract_objects = Mock(return_value=["cat"])
        scorer.object_attribute_extractor.normalize_detected_objects = Mock(
            return_value=[]
        )
        scorer.object_attribute_extractor.extract_numeracy_objects = Mock(
            return_value=["2 cats"]
        )
        scorer.object_attribute_extractor.numeric_diff = Mock(
            return_value=["Expected 2 cats, found 0"]
        )
        scorer.object_attribute_extractor.extract_numeracy_relationship = Mock(
            return_value={"objects": {"cat": 2}}
        )

        test_image = Mock()
        test_image.info = {"path": "/path/to/test.jpg"}

        correct, score, non_conformity = scorer.evaluate(
            test_image, "2 cats in the image"
        )

        assert correct is False
        assert len(non_conformity) > 0


@pytest.mark.integration
class TestSpatialScorerIntegration:
    """Integration tests for SpatialScorer with dependencies."""

    @patch("ssa.vlm.Llm")
    @patch("ssa.od.ObjectDetectionModel")
    def test_spatial_scorer_full_pipeline(self, mock_od, mock_llm):
        """Test complete spatial scoring pipeline."""
        # Setup mock object detector
        mock_detector_instance = Mock()
        mock_detector_instance.model = Mock()
        mock_detector_instance.model.names = {0: "cat", 1: "dog"}

        # Mock detection results - cat and dog in spatial relationship
        mock_box1 = Mock()
        mock_box1.cls = Mock()
        mock_box1.cls.item.return_value = 0  # cat
        mock_box1.conf = Mock()
        mock_box1.conf.item.return_value = 0.95
        mock_box1.xyxy = Mock()
        mock_box1.xyxy.squeeze.return_value.tolist.return_value = [0, 0, 10, 10]

        mock_box2 = Mock()
        mock_box2.cls = Mock()
        mock_box2.cls.item.return_value = 1  # dog
        mock_box2.conf = Mock()
        mock_box2.conf.item.return_value = 0.92
        mock_box2.xyxy = Mock()
        mock_box2.xyxy.squeeze.return_value.tolist.return_value = [15, 0, 25, 10]

        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]

        mock_detector_instance.return_value = [mock_result]
        mock_od.return_value = mock_detector_instance

        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create scorer
        scorer = SpatialScorer(config=SpatialConfig(debug=False))

        # Replace scorer's object detector with our mock
        scorer.object_detector = mock_detector_instance

        # Mock the object attribute extractor methods
        scorer.object_attribute_extractor.extract_objects = Mock(
            return_value=["cat", "dog"]
        )
        scorer.object_attribute_extractor.normalize_detected_objects = Mock(
            return_value=["cat", "dog"]
        )
        scorer.object_attribute_extractor.extract_spatial_relationship = Mock(
            return_value={"obj1": "cat", "obj2": "dog", "relationship": "next to"}
        )
        scorer.object_attribute_extractor.object_diff = Mock(return_value=[])

        # Create test image
        test_image = Mock()
        test_image.info = {"path": "/path/to/test.jpg"}

        # Evaluate
        correct, score, non_conformity = scorer.evaluate(test_image, "cat next to dog")

        # Verify results
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(non_conformity, list)
        assert isinstance(correct, bool)

    @patch("ssa.vlm.Llm")
    @patch("ssa.od.ObjectDetectionModel")
    def test_spatial_scorer_position_calculation(self, mock_od, mock_llm):
        """Test spatial position calculation with real bounding boxes."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_od.return_value = mock_detector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create scorer
        scorer = SpatialScorer()

        # Test "next to" relationship
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        box2 = {"x_min": 15, "y_min": 0, "x_max": 25, "y_max": 10}
        score = scorer.determine_position("next to", box1, box2)
        assert score > 0.0

        # Test "on the left of" relationship
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        box2 = {"x_min": 20, "y_min": 0, "x_max": 30, "y_max": 10}
        score = scorer.determine_position("on the left of", box1, box2)
        assert score > 0.0

        # Test wrong relationship
        score = scorer.determine_position("on the right of", box1, box2)
        assert score == 0.0


@pytest.mark.integration
class TestObjectDetectionIntegration:
    """Integration tests for ObjectDetectionModel."""

    @patch("ssa.od._get_yolo_class")
    @patch("os.path.exists", return_value=True)
    def test_object_detection_model_initialization_flow(
        self, mock_exists, mock_get_class
    ):
        """Test complete object detection model initialization."""
        from unittest.mock import mock_open

        from ssa.od import YOLOV11

        # Setup mocks
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_instance.names = {0: "cat", 1: "dog"}
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        # Mock JSON file content
        json_content = '{"categories": ["cat", "dog"]}'
        with patch("builtins.open", mock_open(read_data=json_content)):
            # Initialize model
            model = ObjectDetectionModel(YOLOV11, pred_classes="from_json")

            # Verify initialization
            assert model.key == YOLOV11
            assert model.model is not None

            # Verify get_classes works
            classes = model.get_classes()
            assert classes == ["cat", "dog"]

    @patch("ssa.od._get_yolo_class")
    @patch("os.path.exists", return_value=True)
    def test_yoloworld_class_setting_integration(self, mock_exists, mock_get_class):
        """Test YOLO World class setting integration."""
        from unittest.mock import mock_open

        from ssa.od import YOLOWORLD

        # Setup mocks
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_class.return_value = mock_model_class

        # Mock JSON file content
        json_content = '{"categories": ["person", "car"]}'
        with patch("builtins.open", mock_open(read_data=json_content)):
            # Initialize model
            model = ObjectDetectionModel(YOLOWORLD, pred_classes="from_json")

            # Verify set_classes was called during init
            mock_model_instance.set_classes.assert_called_once()

            # Test manual class setting
            mock_model_instance.set_classes.reset_mock()
            model.set_classes(["bird", "plane"])
            mock_model_instance.set_classes.assert_called_once_with(["bird", "plane"])


@pytest.mark.integration
class TestScorerWithRealODModel:
    """Integration tests combining scorers with real OD model flow."""

    @patch("ssa.vlm.Llm")
    def test_numeracy_scorer_with_od_model_mock(self, mock_llm):
        """Test numeracy scorer with mocked but realistic OD model."""
        # Create mock OD model with realistic behavior
        mock_od = Mock(spec=ObjectDetectionModel)
        mock_od.model = Mock()
        mock_od.model.names = {0: "cat"}

        # Setup mock detection result
        mock_box = Mock()
        mock_box.cls = Mock()
        mock_box.cls.item.return_value = 0
        mock_box.conf = Mock()
        mock_box.conf.item.return_value = 0.95
        mock_box.xyxy = Mock()
        mock_box.xyxy.squeeze.return_value.tolist.return_value = [0, 0, 10, 10]

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_od.return_value = [mock_result]

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create scorer with custom OD model
        scorer = NumeracyScorer(
            config=NumeracyConfig(debug=False), object_detector=mock_od
        )

        # Verify scorer uses custom OD model
        assert scorer.object_detector == mock_od

    @patch("ssa.vlm.Llm")
    def test_spatial_scorer_with_od_model_mock(self, mock_llm):
        """Test spatial scorer with mocked but realistic OD model."""
        # Create mock OD model
        mock_od = Mock(spec=ObjectDetectionModel)
        mock_od.model = Mock()
        mock_od.model.names = {0: "cat", 1: "dog"}

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create scorer with custom OD model
        scorer = SpatialScorer(
            config=SpatialConfig(debug=False), object_detector=mock_od
        )

        # Verify scorer uses custom OD model
        assert scorer.object_detector == mock_od
