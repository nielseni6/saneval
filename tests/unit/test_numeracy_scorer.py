"""
Unit tests for NumeracyScorer.

These are TRUE unit tests that test individual methods in isolation
without requiring full scorer initialization. They use mocks to avoid
file I/O, heavy object creation, and complex dependencies.

Tests cover:
- Scorer method logic (not initialization)
- Individual method behavior with mocked dependencies
- Return values and error handling
"""

from unittest.mock import Mock

import pytest
import torch

from ssa.scorers.numeracy import NumeracyConfig, NumeracyScorer


class TestNumeracyConfig:
    """Tests for NumeracyConfig."""

    def test_default_config_values(self):
        """Test that default config values are set correctly."""
        config = NumeracyConfig()
        assert hasattr(config, "od_model")
        assert hasattr(config, "llm")


class TestNumeracyScorerMethods:
    """Tests for NumeracyScorer methods using mocks."""

    def test_get_scorer_name(self):
        """Test get_scorer_name returns correct name."""
        scorer = Mock(spec=NumeracyScorer)
        # Call the real method
        result = NumeracyScorer.get_scorer_name(scorer)
        assert result == "Numeracy"

    def test_scorer_name_property(self):
        """Test scorer_name property returns correct name."""
        scorer = Mock(spec=NumeracyScorer)
        scorer.scorer_name = "numeracy"
        assert scorer.scorer_name == "numeracy"


class TestGetNonconformityResults:
    """Tests for get_nonconformity_results method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer with mocked extractor."""
        scorer = Mock(spec=NumeracyScorer)
        scorer.object_attribute_extractor = Mock()
        return scorer

    def test_no_numeric_differences(self, mock_scorer):
        """Test when there are no numeric differences."""
        mock_scorer.object_attribute_extractor.extract_numeracy_objects.return_value = [
            "2 cats"
        ]
        mock_scorer.object_attribute_extractor.numeric_diff.return_value = []

        differences, correctness = NumeracyScorer.get_nonconformity_results(
            mock_scorer, "2 cats", ["cat", "cat"]
        )

        assert correctness is True
        assert differences == []
        mock_scorer.object_attribute_extractor.extract_numeracy_objects.assert_called_once_with(
            "2 cats"
        )
        mock_scorer.object_attribute_extractor.numeric_diff.assert_called_once()

    def test_with_numeric_differences(self, mock_scorer):
        """Test when there are numeric differences."""
        mock_scorer.object_attribute_extractor.extract_numeracy_objects.return_value = [
            "2 cats"
        ]
        mock_scorer.object_attribute_extractor.numeric_diff.return_value = [
            "Expected 2, found 1"
        ]

        differences, correctness = NumeracyScorer.get_nonconformity_results(
            mock_scorer, "2 cats", ["cat"]
        )

        assert correctness is False
        assert len(differences) == 1
        assert "Expected 2, found 1" in differences

    def test_no_extracted_objects(self, mock_scorer):
        """Test when no objects are extracted from prompt."""
        mock_scorer.object_attribute_extractor.extract_numeracy_objects.return_value = (
            []
        )
        mock_scorer.object_attribute_extractor.numeric_diff.return_value = []

        differences, correctness = NumeracyScorer.get_nonconformity_results(
            mock_scorer, "empty prompt", []
        )

        assert correctness is True
        assert differences == []


class TestEvaluate:
    """Tests for evaluate method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer with necessary methods."""
        scorer = Mock(spec=NumeracyScorer)
        scorer.eval_metrics = Mock()
        scorer.get_nonconformity_results = Mock()
        return scorer

    def test_evaluate_perfect_score(self, mock_scorer):
        """Test evaluate with perfect numeracy score."""
        mock_image = Mock()
        mock_image.info = {"path": "/path/to/image.jpg"}

        mock_scorer.eval_metrics.return_value = (1.0, ["cat", "dog"], ["cat", "dog"])
        mock_scorer.get_nonconformity_results.return_value = ([], True)

        correct, score, non_conformity = NumeracyScorer.evaluate(
            mock_scorer, mock_image, "Image with 2 cats and 1 dog"
        )

        assert correct is True
        assert score == 1.0
        assert non_conformity == []

    def test_evaluate_with_differences(self, mock_scorer):
        """Test evaluate with numeric differences."""
        mock_image = Mock()
        mock_image.info = {"path": "/path/to/image.jpg"}

        mock_scorer.eval_metrics.return_value = (0.5, ["cat"], ["cat"])
        mock_scorer.get_nonconformity_results.return_value = (
            ["Expected 2 cats, found 1"],
            False,
        )

        correct, score, non_conformity = NumeracyScorer.evaluate(
            mock_scorer, mock_image, "Image with 2 cats"
        )

        assert correct is False
        assert score == 0.5
        assert len(non_conformity) == 1

    def test_evaluate_zero_score(self, mock_scorer):
        """Test evaluate with zero score."""
        mock_image = Mock()
        mock_image.info = {"path": "/path/to/image.jpg"}

        mock_scorer.eval_metrics.return_value = (0.0, [], [])
        mock_scorer.get_nonconformity_results.return_value = (
            ["Expected 2 cats, found 0"],
            False,
        )

        correct, score, non_conformity = NumeracyScorer.evaluate(
            mock_scorer, mock_image, "Image with 2 cats"
        )

        assert correct is False
        assert score == 0.0


class TestGetSpecificScore:
    """Tests for get_specific_score method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer."""
        scorer = Mock(spec=NumeracyScorer)
        scorer.eval_numeracy = Mock(return_value=0.95)
        return scorer

    def test_calls_eval_numeracy(self, mock_scorer):
        """Test that get_specific_score calls eval_numeracy."""
        obj = ["cat", "dog"]
        obj_bounding_box = [[0, 0, 10, 10], [20, 20, 30, 30]]
        instance_score = torch.tensor([0.9, 0.85])

        score = NumeracyScorer.get_specific_score(
            mock_scorer,
            "/path/to/image.jpg",
            "test prompt",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score == 0.95
        mock_scorer.eval_numeracy.assert_called_once_with(
            "/path/to/image.jpg", "test prompt", obj, obj_bounding_box, instance_score
        )


class TestEvalNumeracy:
    """Tests for eval_numeracy method."""

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer."""
        scorer = Mock(spec=NumeracyScorer)
        scorer.object_attribute_extractor = Mock()
        scorer.normalize_object_name = Mock(side_effect=lambda x: x)
        return scorer

    def test_perfect_match(self, mock_scorer):
        """Test eval_numeracy with perfect count match."""
        mock_scorer.object_attribute_extractor.extract_numeracy_relationship.return_value = {
            "objects": {"cat": 2, "dog": 1}
        }

        obj = ["cat", "cat", "dog"]
        obj_bounding_box = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        instance_score = torch.tensor([0.9, 0.8, 0.85])

        score = NumeracyScorer.eval_numeracy(
            mock_scorer,
            "/path/to/image.jpg",
            "2 cats and 1 dog",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score > 0.0
        assert score <= 1.0

    def test_partial_match(self, mock_scorer):
        """Test eval_numeracy with partial count match."""
        mock_scorer.object_attribute_extractor.extract_numeracy_relationship.return_value = {
            "objects": {"cat": 2, "dog": 1}
        }

        obj = ["cat", "dog"]  # Missing one cat
        obj_bounding_box = [[0, 0, 10, 10], [20, 20, 30, 30]]
        instance_score = torch.tensor([0.9, 0.85])

        score = NumeracyScorer.eval_numeracy(
            mock_scorer,
            "/path/to/image.jpg",
            "2 cats and 1 dog",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score >= 0.0
        assert score < 1.0

    def test_no_extraction(self, mock_scorer):
        """Test eval_numeracy when LLM extraction fails."""
        mock_scorer.object_attribute_extractor.extract_numeracy_relationship.return_value = (
            None
        )

        score = NumeracyScorer.eval_numeracy(
            mock_scorer,
            "/path/to/image.jpg",
            "invalid prompt",
            [],
            [],
            torch.tensor([]),
        )

        assert score == 0.0

    def test_empty_objects_in_extraction(self, mock_scorer):
        """Test eval_numeracy with empty objects in extraction."""
        mock_scorer.object_attribute_extractor.extract_numeracy_relationship.return_value = {
            "objects": {}
        }

        score = NumeracyScorer.eval_numeracy(
            mock_scorer, "/path/to/image.jpg", "prompt", [], [], torch.tensor([])
        )

        assert score == 0.0

    def test_no_matches(self, mock_scorer):
        """Test eval_numeracy when no objects match."""
        mock_scorer.object_attribute_extractor.extract_numeracy_relationship.return_value = {
            "objects": {"cat": 2}
        }

        obj = ["dog", "bird"]  # No cats
        obj_bounding_box = [[0, 0, 10, 10], [20, 20, 30, 30]]
        instance_score = torch.tensor([0.9, 0.85])

        score = NumeracyScorer.eval_numeracy(
            mock_scorer,
            "/path/to/image.jpg",
            "2 cats",
            obj,
            obj_bounding_box,
            instance_score,
        )

        assert score == 0.0
