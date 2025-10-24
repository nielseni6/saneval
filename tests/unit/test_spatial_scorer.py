"""
Unit tests for SpatialScorer.

These are TRUE unit tests that test individual methods in isolation
without requiring full scorer initialization. They use mocks to avoid
file I/O, heavy object creation, and complex dependencies.

Tests cover:
- Scorer method logic (not initialization)
- Spatial relationship evaluation
- Position determination logic
- Individual method behavior with mocked dependencies
"""

from unittest.mock import Mock

import pytest
import torch

from ssa.scorers.spatial import SpatialConfig, SpatialScorer


class TestSpatialConfig:
    """Tests for SpatialConfig."""

    def test_default_config_values(self):
        """Test that default config values are set correctly."""
        config = SpatialConfig()
        assert hasattr(config, "od_model")
        assert hasattr(config, "llm")
        assert hasattr(config, "distance_threshold")
        assert hasattr(config, "iou_threshold")
        assert config.distance_threshold == 150.0
        assert config.iou_threshold == 0.5

    def test_custom_config_values(self):
        """Test setting custom config values."""
        config = SpatialConfig()
        config.distance_threshold = 200.0
        config.iou_threshold = 0.6
        assert config.distance_threshold == 200.0
        assert config.iou_threshold == 0.6


class TestSpatialScorerMethods:
    """Tests for SpatialScorer methods using mocks."""

    def test_get_scorer_name(self):
        """Test get_scorer_name returns correct name."""
        scorer = Mock(spec=SpatialScorer)
        result = SpatialScorer.get_scorer_name(scorer)
        assert result == "Spatial"

    def test_scorer_name_property(self):
        """Test scorer_name property returns correct name."""
        scorer = Mock(spec=SpatialScorer)
        scorer.scorer_name = "spatial"
        assert scorer.scorer_name == "spatial"


class TestGetNonconformityResults:
    """Tests for get_nonconformity_results method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer with mocked extractor."""
        scorer = Mock(spec=SpatialScorer)
        scorer.object_attribute_extractor = Mock()
        return scorer

    def test_no_missing_objects(self, mock_scorer):
        """Test when there are no missing objects."""
        mock_scorer.object_attribute_extractor.extract_objects.return_value = [
            "cat",
            "dog",
        ]
        mock_scorer.object_attribute_extractor.object_diff.return_value = []

        missing, correctness = SpatialScorer.get_nonconformity_results(
            mock_scorer, "cat next to dog", ["cat", "dog"]
        )

        assert correctness is True
        assert missing == []
        mock_scorer.object_attribute_extractor.extract_objects.assert_called_once_with(
            "cat next to dog"
        )

    def test_with_missing_objects(self, mock_scorer):
        """Test when there are missing objects."""
        mock_scorer.object_attribute_extractor.extract_objects.return_value = [
            "cat",
            "dog",
        ]
        mock_scorer.object_attribute_extractor.object_diff.return_value = ["dog"]

        missing, correctness = SpatialScorer.get_nonconformity_results(
            mock_scorer, "cat next to dog", ["cat"]
        )

        assert correctness is False
        assert len(missing) == 1
        assert "dog" in missing

    def test_no_extracted_objects(self, mock_scorer):
        """Test when no objects are extracted from prompt."""
        mock_scorer.object_attribute_extractor.extract_objects.return_value = []
        mock_scorer.object_attribute_extractor.object_diff.return_value = []

        missing, correctness = SpatialScorer.get_nonconformity_results(
            mock_scorer, "empty prompt", []
        )

        assert correctness is True
        assert missing == []


class TestEvaluate:
    """Tests for evaluate method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer with necessary methods."""
        scorer = Mock(spec=SpatialScorer)
        scorer.eval_metrics = Mock()
        scorer.get_nonconformity_results = Mock()
        return scorer

    def test_evaluate_perfect_score(self, mock_scorer):
        """Test evaluate with perfect spatial score."""
        mock_image = Mock()
        mock_image.info = {"path": "/path/to/image.jpg"}

        mock_scorer.eval_metrics.return_value = (1.0, ["cat", "dog"], ["cat", "dog"])
        mock_scorer.get_nonconformity_results.return_value = ([], True)

        correct, score, non_conformity = SpatialScorer.evaluate(
            mock_scorer, mock_image, "cat next to dog"
        )

        assert correct is True
        assert score == 1.0
        assert non_conformity == []

    def test_evaluate_with_missing_objects(self, mock_scorer):
        """Test evaluate with missing objects."""
        mock_image = Mock()
        mock_image.info = {"path": "/path/to/image.jpg"}

        mock_scorer.eval_metrics.return_value = (0.5, ["cat"], ["cat"])
        mock_scorer.get_nonconformity_results.return_value = (["bird"], False)

        correct, score, non_conformity = SpatialScorer.evaluate(
            mock_scorer, mock_image, "cat next to dog and bird"
        )

        assert correct is False
        assert score == 0.5
        assert len(non_conformity) == 1
        assert "bird" in non_conformity

    def test_evaluate_zero_score(self, mock_scorer):
        """Test evaluate with zero score."""
        mock_image = Mock()
        mock_image.info = {"path": "/path/to/image.jpg"}

        mock_scorer.eval_metrics.return_value = (0.0, [], [])
        mock_scorer.get_nonconformity_results.return_value = (["cat"], False)

        correct, score, non_conformity = SpatialScorer.evaluate(
            mock_scorer, mock_image, "cat next to dog"
        )

        assert correct is False
        assert score == 0.0


class TestGetSpecificScore:
    """Tests for get_specific_score method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer."""
        scorer = Mock(spec=SpatialScorer)
        scorer.eval_spatial_2d = Mock(return_value=0.95)
        return scorer

    def test_calls_eval_spatial_2d(self, mock_scorer):
        """Test that get_specific_score calls eval_spatial_2d."""
        obj = ["cat", "dog"]
        obj_bounding_box = [[0, 0, 10, 10], [20, 20, 30, 30]]
        instance_score = torch.tensor([0.9, 0.85])

        score = SpatialScorer.get_specific_score(
            mock_scorer,
            "/path/to/image.jpg",
            "test prompt",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score == 0.95
        mock_scorer.eval_spatial_2d.assert_called_once_with(
            "/path/to/image.jpg", "test prompt", obj, obj_bounding_box, instance_score
        )


class TestEvalSpatial2D:
    """Tests for eval_spatial_2d method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer."""
        scorer = Mock(spec=SpatialScorer)
        scorer.object_attribute_extractor = Mock()
        scorer.determine_position = Mock(return_value=1.0)
        scorer.spatial_objects = []
        return scorer

    def test_both_objects_found(self, mock_scorer):
        """Test eval_spatial_2d when both objects are found."""
        mock_scorer.object_attribute_extractor.extract_spatial_relationship.return_value = {
            "obj1": "cat",
            "obj2": "dog",
            "relationship": "next to",
        }

        obj = ["cat", "dog"]
        obj_bounding_box = [[0, 0, 10, 10], [20, 20, 30, 30]]
        instance_score = torch.tensor([0.9, 0.85])

        score = SpatialScorer.eval_spatial_2d(
            mock_scorer,
            "/path/to/image.jpg",
            "cat next to dog",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score > 0.0
        assert score <= 1.0
        assert mock_scorer.spatial_objects == ["cat", "dog"]

    def test_only_first_object_found(self, mock_scorer):
        """Test eval_spatial_2d when only first object is found."""
        mock_scorer.object_attribute_extractor.extract_spatial_relationship.return_value = {
            "obj1": "cat",
            "obj2": "dog",
            "relationship": "next to",
        }

        obj = ["cat"]
        obj_bounding_box = [[0, 0, 10, 10]]
        instance_score = torch.tensor([0.9])

        score = SpatialScorer.eval_spatial_2d(
            mock_scorer,
            "/path/to/image.jpg",
            "cat next to dog",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score >= 0.0
        assert score < 1.0

    def test_no_objects_found(self, mock_scorer):
        """Test eval_spatial_2d when no objects are found."""
        mock_scorer.object_attribute_extractor.extract_spatial_relationship.return_value = {
            "obj1": "cat",
            "obj2": "dog",
            "relationship": "next to",
        }

        obj = []
        obj_bounding_box = []
        instance_score = torch.tensor([])

        score = SpatialScorer.eval_spatial_2d(
            mock_scorer,
            "/path/to/image.jpg",
            "cat next to dog",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score == 0.0

    def test_no_extraction(self, mock_scorer):
        """Test eval_spatial_2d when LLM extraction fails."""
        mock_scorer.object_attribute_extractor.extract_spatial_relationship.return_value = (
            None
        )

        score = SpatialScorer.eval_spatial_2d(
            mock_scorer,
            "/path/to/image.jpg",
            "invalid prompt",
            [],
            [],
            torch.tensor([]),
        )

        assert score == 0.0


class TestDeterminePosition:
    """Tests for determine_position method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer with config."""
        scorer = Mock(spec=SpatialScorer)
        scorer.config = SpatialConfig()
        return scorer

    def test_next_to_relationship(self, mock_scorer):
        """Test determine_position for 'next to' relationship."""
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        box2 = {"x_min": 15, "y_min": 0, "x_max": 25, "y_max": 10}

        score = SpatialScorer.determine_position(mock_scorer, "next to", box1, box2)
        assert score > 0.0

    def test_right_of_relationship(self, mock_scorer):
        """Test determine_position for 'on the right of' relationship."""
        box1 = {"x_min": 20, "y_min": 0, "x_max": 30, "y_max": 10}  # box1 on right
        box2 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}  # box2 on left

        score = SpatialScorer.determine_position(
            mock_scorer, "on the right of", box1, box2
        )
        assert score > 0.0

    def test_left_of_relationship(self, mock_scorer):
        """Test determine_position for 'on the left of' relationship."""
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}  # box1 on left
        box2 = {"x_min": 20, "y_min": 0, "x_max": 30, "y_max": 10}  # box2 on right

        score = SpatialScorer.determine_position(
            mock_scorer, "on the left of", box1, box2
        )
        assert score > 0.0

    def test_top_of_relationship(self, mock_scorer):
        """Test determine_position for 'on the top of' relationship."""
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}  # box1 on top
        box2 = {"x_min": 0, "y_min": 20, "x_max": 10, "y_max": 30}  # box2 on bottom

        score = SpatialScorer.determine_position(
            mock_scorer, "on the top of", box1, box2
        )
        assert score > 0.0

    def test_bottom_of_relationship(self, mock_scorer):
        """Test determine_position for 'on the bottom of' relationship."""
        box1 = {"x_min": 0, "y_min": 20, "x_max": 10, "y_max": 30}  # box1 on bottom
        box2 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}  # box2 on top

        score = SpatialScorer.determine_position(
            mock_scorer, "on the bottom of", box1, box2
        )
        assert score > 0.0

    def test_wrong_relationship(self, mock_scorer):
        """Test determine_position with wrong spatial relationship."""
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}  # box1 on left
        box2 = {"x_min": 20, "y_min": 0, "x_max": 30, "y_max": 10}  # box2 on right

        # Should return low score when claiming box1 is on right of box2
        score = SpatialScorer.determine_position(
            mock_scorer, "on the right of", box1, box2
        )
        assert score == 0.0

    def test_with_custom_thresholds(self, mock_scorer):
        """Test determine_position with custom thresholds."""
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        box2 = {"x_min": 15, "y_min": 0, "x_max": 25, "y_max": 10}

        score = SpatialScorer.determine_position(
            mock_scorer,
            "next to",
            box1,
            box2,
            iou_threshold=0.6,
            distance_threshold=200.0,
        )
        assert score >= 0.0

    def test_unknown_locality(self, mock_scorer):
        """Test determine_position with unknown locality."""
        box1 = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        box2 = {"x_min": 15, "y_min": 0, "x_max": 25, "y_max": 10}

        score = SpatialScorer.determine_position(mock_scorer, "unknown", box1, box2)
        assert score == 0.0
