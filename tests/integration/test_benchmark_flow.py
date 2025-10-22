"""Integration tests for benchmark workflow."""

import tempfile
from pathlib import Path

import pytest

from ssa.benchmark_runner import _load_and_parse_images


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
        with pytest.raises(ValueError):
            _load_and_parse_images("/path/that/does/not/exist")


class TestBenchmarkWorkflow:
    """Tests for end-to-end benchmark workflows."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_benchmark_spatial(self, sample_images_dir):
        """Test complete spatial benchmark run (requires API keys)."""
        # This test would require actual API credentials
        # Mark as slow and skip if credentials not available
        pytest.skip("Requires API credentials - run manually with valid credentials")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_benchmark_numeracy(self, sample_images_dir):
        """Test complete numeracy benchmark run (requires API keys)."""
        pytest.skip("Requires API credentials - run manually with valid credentials")

    @pytest.mark.slow
    def test_benchmark_config_creation(self):
        """Test benchmark configuration can be created (requires API keys)."""
        # Skip this test as it requires actual API credentials
        pytest.skip("Requires API credentials - test scorer initialization separately")


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
                assert prompt.source == "filename"
