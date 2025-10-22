"""Unit tests for ssa.config module."""
import pytest
from pathlib import Path
from ssa.config import SUPPORTED_IMAGE_FORMATS, DEFAULT_OUTPUT_BASE


class TestConfig:
    """Tests for configuration constants."""

    def test_supported_image_formats(self):
        """Supported formats should be a set."""
        assert isinstance(SUPPORTED_IMAGE_FORMATS, set)
        assert len(SUPPORTED_IMAGE_FORMATS) > 0

    def test_supported_formats_contents(self):
        """Common image formats should be supported."""
        assert '.png' in SUPPORTED_IMAGE_FORMATS
        assert '.jpg' in SUPPORTED_IMAGE_FORMATS
        assert '.jpeg' in SUPPORTED_IMAGE_FORMATS

    def test_formats_lowercase(self):
        """All formats should be lowercase."""
        for fmt in SUPPORTED_IMAGE_FORMATS:
            assert fmt == fmt.lower()

    def test_formats_have_dot_prefix(self):
        """All formats should start with a dot."""
        for fmt in SUPPORTED_IMAGE_FORMATS:
            assert fmt.startswith('.')

    def test_default_output_base_is_path(self):
        """Default output base should be a Path."""
        assert isinstance(DEFAULT_OUTPUT_BASE, Path)

    def test_default_output_base_not_empty(self):
        """Default output base should not be empty."""
        assert str(DEFAULT_OUTPUT_BASE) != ""
