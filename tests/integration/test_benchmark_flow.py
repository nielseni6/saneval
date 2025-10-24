"""Integration tests for benchmark workflow."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ssa.benchmark_runner import (
    _load_and_parse_images,
    benchmark_model,
    resolve_benchmark_config,
)


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


class TestImageLoading:
    """Tests for image loading pipeline."""

    def test_load_and_parse_images_from_spatial(self, sample_images_dir):
        """Test loading images from spatial sample directory."""
        spatial_dir = sample_images_dir / "spatial"
        if spatial_dir.exists():
            pairs = _load_and_parse_images(str(spatial_dir))
            assert len(pairs) > 0
            # Each pair should be (image_path, Prompt)
            assert all(len(pair) == 2 for pair in pairs)
            # Check that paths exist
            assert all(pair[0].exists() for pair in pairs)

    def test_load_and_parse_images_from_numeracy(self, sample_images_dir):
        """Test loading images from numeracy sample directory."""
        numeracy_dir = sample_images_dir / "numeracy"
        if numeracy_dir.exists():
            pairs = _load_and_parse_images(str(numeracy_dir))
            assert len(pairs) > 0

    def test_load_and_parse_images_from_attribute_binding(self, sample_images_dir):
        """Test loading images from attribute_binding sample directory."""
        attr_dir = sample_images_dir / "attribute_binding"
        if attr_dir.exists():
            pairs = _load_and_parse_images(str(attr_dir))
            assert len(pairs) > 0

    def test_load_and_parse_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = _load_and_parse_images(tmpdir)
            assert len(pairs) == 0

    def test_load_and_parse_invalid_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            _load_and_parse_images("/path/that/does/not/exist")


class TestBenchmarkWorkflow:
    """Tests for end-to-end benchmark workflows."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_benchmark_spatial(
        self, sample_images_dir, mock_provider_factory
    ):
        """Test complete spatial benchmark run with mocked dependencies."""
        import json
        import shutil

        # Create temporary output directory
        output_dir = Path(tempfile.mkdtemp(prefix="benchmark_test_"))

        try:
            # Create mock provider
            mock_provider = mock_provider_factory("gemini-2.5-flash")

            # Mock ObjectDetectionModel
            mock_od_class = Mock()
            mock_od_instance = Mock()
            mock_od_instance.model = Mock()
            mock_od_instance.model.names = {0: "cat", 1: "dog"}
            mock_od_class.return_value = mock_od_instance

            # Setup mock scorer evaluate method
            def mock_evaluate(image, input_prompt, **kwargs):
                """Mock evaluate that returns successful spatial score."""
                return (True, 0.95, [])

            with patch.dict(
                "ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}
            ):
                with patch(
                    "ssa.scorers.spatial_numeracy_base.ObjectDetectionModel",
                    mock_od_class,
                ):
                    with patch(
                        "ssa.scorers.spatial_numeracy_base.Llm",
                        return_value=mock_provider.return_value,
                    ):
                        with patch(
                            "ssa.scorers.spatial.SpatialScorer.evaluate",
                            side_effect=mock_evaluate,
                        ):
                            # Get spatial images directory
                            spatial_dir = sample_images_dir / "spatial"
                            if not spatial_dir.exists():
                                pytest.skip(
                                    f"Spatial sample directory not found: {spatial_dir}"
                                )

                            # Create scorer configuration
                            scoring_keys = ["spatial"]
                            scorers_dict = resolve_benchmark_config(scoring_keys, {})

                            # Run benchmark
                            result_dir = benchmark_model(
                                images_dir=str(spatial_dir),
                                output_dir=str(output_dir),
                                scorers=scorers_dict,
                                rescoring=False,
                            )

                            # Verify output directory exists
                            assert Path(result_dir).exists()

                            # Verify output files were created
                            aggregates_file = Path(result_dir) / "aggregates.json"
                            run_results_file = Path(result_dir) / "run_results.json"

                            assert (
                                aggregates_file.exists()
                            ), "aggregates.json should be created"
                            assert (
                                run_results_file.exists()
                            ), "run_results.json should be created"

                            # Verify aggregates content
                            with open(aggregates_file) as f:
                                aggregates = json.load(f)
                                assert isinstance(aggregates, dict)
                                assert "failed_gen" in aggregates
                                assert "failed_scoring_prompts" in aggregates

                            # Verify run results content
                            with open(run_results_file) as f:
                                run_results = json.load(f)
                                assert isinstance(run_results, list)
                                # Should have at least one result if images exist
                                if run_results:
                                    assert "prompt" in run_results[0]

        finally:
            # Clean up temporary directory
            if output_dir.exists():
                shutil.rmtree(output_dir)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_benchmark_numeracy(
        self, sample_images_dir, mock_provider_factory
    ):
        """Test complete numeracy benchmark run with mocked dependencies."""
        import json
        import shutil

        # Create temporary output directory
        output_dir = Path(tempfile.mkdtemp(prefix="benchmark_test_"))

        try:
            # Create mock provider
            mock_provider = mock_provider_factory("gemini-2.5-flash")

            # Mock ObjectDetectionModel
            mock_od_class = Mock()
            mock_od_instance = Mock()
            mock_od_instance.model = Mock()
            mock_od_instance.model.names = {0: "cat", 1: "dog"}
            mock_od_class.return_value = mock_od_instance

            # Setup mock scorer evaluate method
            def mock_evaluate(image, input_prompt, **kwargs):
                """Mock evaluate that returns successful numeracy score."""
                return (True, 0.88, [])

            with patch.dict(
                "ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}
            ):
                with patch(
                    "ssa.scorers.spatial_numeracy_base.ObjectDetectionModel",
                    mock_od_class,
                ):
                    with patch(
                        "ssa.scorers.spatial_numeracy_base.Llm",
                        return_value=mock_provider.return_value,
                    ):
                        with patch(
                            "ssa.scorers.numeracy.NumeracyScorer.evaluate",
                            side_effect=mock_evaluate,
                        ):
                            # Get numeracy images directory
                            numeracy_dir = sample_images_dir / "numeracy"
                            if not numeracy_dir.exists():
                                pytest.skip(
                                    f"Numeracy sample directory not found: {numeracy_dir}"
                                )

                            # Create scorer configuration
                            scoring_keys = ["numeracy"]
                            scorers_dict = resolve_benchmark_config(scoring_keys, {})

                            # Run benchmark
                            result_dir = benchmark_model(
                                images_dir=str(numeracy_dir),
                                output_dir=str(output_dir),
                                scorers=scorers_dict,
                                rescoring=False,
                            )

                            # Verify output directory exists
                            assert Path(result_dir).exists()

                            # Verify output files were created
                            aggregates_file = Path(result_dir) / "aggregates.json"
                            run_results_file = Path(result_dir) / "run_results.json"

                            assert (
                                aggregates_file.exists()
                            ), "aggregates.json should be created"
                            assert (
                                run_results_file.exists()
                            ), "run_results.json should be created"

                            # Verify aggregates content
                            with open(aggregates_file) as f:
                                aggregates = json.load(f)
                                assert isinstance(aggregates, dict)
                                assert "failed_gen" in aggregates
                                assert "failed_scoring_prompts" in aggregates

                            # Verify run results content
                            with open(run_results_file) as f:
                                run_results = json.load(f)
                                assert isinstance(run_results, list)
                                # Should have at least one result if images exist
                                if run_results:
                                    assert "prompt" in run_results[0]

        finally:
            # Clean up temporary directory
            if output_dir.exists():
                shutil.rmtree(output_dir)

    @pytest.mark.slow
    def test_benchmark_config_creation(self, mock_provider_factory):
        """Test benchmark configuration can be created with mocked dependencies."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")

        # Mock ObjectDetectionModel to avoid model loading
        mock_od_class = Mock()
        mock_od_instance = Mock()
        mock_od_instance.model = Mock()
        mock_od_instance.model.names = {0: "cat", 1: "dog"}
        mock_od_class.return_value = mock_od_instance

        # Patch dependencies for scorer initialization
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            with patch(
                "ssa.scorers.spatial_numeracy_base.ObjectDetectionModel", mock_od_class
            ):
                with patch(
                    "ssa.scorers.spatial_numeracy_base.Llm",
                    return_value=mock_provider.return_value,
                ):
                    # Create scorer configuration for spatial and numeracy
                    scoring_keys = ["spatial", "numeracy"]
                    bench_config_overrides = {}

                    # Call resolve_benchmark_config
                    scorers_dict = resolve_benchmark_config(
                        scoring_keys, bench_config_overrides
                    )

                    # Verify results
                    assert len(scorers_dict) == 2
                    assert "spatial" in scorers_dict
                    assert "numeracy" in scorers_dict

                    # Verify scorers are ModelScorer instances
                    from ssa.scorers.model_scorer import ModelScorer

                    assert isinstance(scorers_dict["spatial"], ModelScorer)
                    assert isinstance(scorers_dict["numeracy"], ModelScorer)

                    # Verify each scorer has correct key
                    assert scorers_dict["spatial"].key == "spatial"
                    assert scorers_dict["numeracy"].key == "numeracy"

                    # Verify scorers have underlying models
                    assert scorers_dict["spatial"].model is not None
                    assert scorers_dict["numeracy"].model is not None


class TestPromptParsing:
    """Tests for prompt parsing from filenames."""

    def test_prompt_extracted_from_filename(self, sample_images_dir):
        """Test that prompts are correctly extracted from image filenames."""
        spatial_dir = sample_images_dir / "spatial"
        if spatial_dir.exists():
            pairs = _load_and_parse_images(str(spatial_dir))
            if pairs:
                _, prompt = pairs[0]
                # Prompt should have required attributes
                assert hasattr(prompt, "id")
                assert hasattr(prompt, "text")
                assert hasattr(prompt, "source")
                # Verify source is populated with actual filename, not the literal string "filename"
                assert prompt.source is not None
                assert prompt.source != "filename"
                assert isinstance(prompt.source, str) and len(prompt.source) > 0
