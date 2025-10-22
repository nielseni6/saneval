"""Unit tests for ssa.prompts module."""

import pytest

from ssa.prompts import Corpus, Prompt, generate_prompt_hash, hash_prompts


class TestPromptHashing:
    """Tests for prompt hashing functionality."""

    def test_generate_prompt_hash_deterministic(self):
        """Hash generation should be deterministic."""
        text = "test prompt"
        hash1 = generate_prompt_hash(text)
        hash2 = generate_prompt_hash(text)
        assert hash1 == hash2
        assert hash1.startswith("prompt-")

    def test_generate_prompt_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        hash1 = generate_prompt_hash("prompt 1")
        hash2 = generate_prompt_hash("prompt 2")
        assert hash1 != hash2

    def test_hash_with_image(self):
        """Hash should incorporate image data."""
        text = "test"
        hash_no_img = generate_prompt_hash(text)
        hash_with_img = generate_prompt_hash(text, image="image_data")
        assert hash_no_img != hash_with_img

    def test_hash_length(self):
        """Hash should have expected length."""
        hash_result = generate_prompt_hash("test")
        # Format is "prompt-" + 22 chars
        assert len(hash_result) == len("prompt-") + 22


class TestPrompt:
    """Tests for Prompt dataclass."""

    def test_prompt_creation(self):
        """Prompt object should be created correctly."""
        prompt = Prompt(id="test-1", text="test prompt", source="test")
        assert prompt.id == "test-1"
        assert prompt.text == "test prompt"
        assert prompt.source == "test"

    def test_prompt_to_dict(self):
        """Prompt should serialize to dict."""
        prompt = Prompt(id="test-1", text="test", source="test")
        data = prompt.to_dict()
        assert isinstance(data, dict)
        assert data["id"] == "test-1"
        assert data["text"] == "test"

    def test_prompt_with_categories(self):
        """Prompt should support categories."""
        prompt = Prompt(id="test-1", text="test", categories=["spatial", "numeracy"])
        assert len(prompt.categories) == 2
        assert "spatial" in prompt.categories

    def test_prompt_with_extras(self):
        """Prompt should support extras metadata."""
        prompt = Prompt(
            id="test-1", text="test", extras={"key1": "value1", "key2": 123}
        )
        assert prompt.extras["key1"] == "value1"
        assert prompt.extras["key2"] == 123


class TestCorpus:
    """Tests for Corpus class."""

    def test_corpus_creation(self):
        """Corpus should be created with prompts."""
        prompts = [
            Prompt(id="p1", text="prompt 1"),
            Prompt(id="p2", text="prompt 2"),
        ]
        corpus = Corpus("test-corpus", prompts)
        assert corpus.name == "test-corpus"
        assert len(corpus.prompts) == 2

    def test_corpus_hash(self):
        """Corpus should have consistent hash."""
        prompts = [Prompt(id="p1", text="prompt 1")]
        corpus1 = Corpus("test", prompts)
        corpus2 = Corpus("test", prompts)
        assert corpus1.hash == corpus2.hash

    def test_corpus_id(self):
        """Corpus should have ID with hash."""
        prompts = [Prompt(id="p1", text="prompt 1")]
        corpus = Corpus("test", prompts)
        assert corpus.id.startswith("corpus-")

    def test_corpus_config(self):
        """Corpus config should contain metadata."""
        prompts = [Prompt(id="p1", text="prompt 1")]
        corpus = Corpus("test", prompts)
        config = corpus.config()
        assert config["name"] == "test"
        assert config["size"] == 1
        assert "hash" in config

    def test_corpus_has_images_false(self):
        """has_images should return False when no images."""
        prompts = [Prompt(id="p1", text="prompt 1")]
        corpus = Corpus("test", prompts)
        assert corpus.has_images() is False

    def test_corpus_has_images_true(self):
        """has_images should return True when images present."""
        prompts = [Prompt(id="p1", text="prompt 1", image="/path/to/image.png")]
        corpus = Corpus("test", prompts)
        assert corpus.has_images() is True

    def test_corpus_image_count(self):
        """image_count should count prompts with images."""
        prompts = [
            Prompt(id="p1", text="prompt 1", image="/path/1.png"),
            Prompt(id="p2", text="prompt 2"),
            Prompt(id="p3", text="prompt 3", image="/path/2.png"),
        ]
        corpus = Corpus("test", prompts)
        assert corpus.image_count() == 2
