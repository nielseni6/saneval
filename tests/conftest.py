"""Shared pytest fixtures for SANEval tests."""
import pytest
from pathlib import Path
from PIL import Image
import numpy as np


@pytest.fixture
def test_data_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_images_dir():
    """Path to sample images directory."""
    return Path(__file__).parent.parent / "images" / "samples"


@pytest.fixture
def mock_image():
    """Create a simple test image."""
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def test_prompt():
    """Sample test prompt."""
    from ssa.prompts import Prompt

    return Prompt(
        id="test-prompt-001",
        text="A red ball next to a blue cube",
        source="test"
    )


@pytest.fixture
def test_corpus(test_prompt):
    """Sample test corpus."""
    from ssa.prompts import Corpus, Prompt

    prompts = [
        test_prompt,
        Prompt(
            id="test-prompt-002",
            text="Three yellow stars",
            source="test"
        ),
    ]
    return Corpus("test-corpus", prompts)
