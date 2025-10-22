"""Unit tests for ssa.utils.base module."""

from ssa.utils.base import (
    DEFAULT_RANDOM_CHARS,
    hash_str_to_alphanumeric,
    short_randomstr,
    unique_id,
)


class TestHashing:
    """Tests for hashing utilities."""

    def test_hash_str_to_alphanumeric(self):
        """Hash should be alphanumeric."""
        result = hash_str_to_alphanumeric("test string", chars=10)
        assert len(result) == 10
        assert result.isalnum()

    def test_hash_deterministic(self):
        """Hash should be deterministic."""
        hash1 = hash_str_to_alphanumeric("test", chars=10)
        hash2 = hash_str_to_alphanumeric("test", chars=10)
        assert hash1 == hash2

    def test_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        hash1 = hash_str_to_alphanumeric("input1", chars=10)
        hash2 = hash_str_to_alphanumeric("input2", chars=10)
        assert hash1 != hash2

    def test_hash_length_parameter(self):
        """Hash length should respect chars parameter."""
        for length in [5, 10, 15, 20]:
            result = hash_str_to_alphanumeric("test", chars=length)
            assert len(result) == length

    def test_hash_no_special_chars(self):
        """Hash should contain only letters and numbers."""
        result = hash_str_to_alphanumeric("test!@#$%", chars=15)
        assert result.replace("_", "").replace("-", "").isalnum()


class TestUniqueId:
    """Tests for unique ID generation."""

    def test_unique_id_format(self):
        """Unique ID should have correct format."""
        uid = unique_id(name="test")
        # Format is: YYYY-MM-DD-HHMM-name-randomstr
        assert "test" in uid
        parts = uid.split("-")
        assert len(parts) >= 5  # date, time, name, random

    def test_unique_id_without_name(self):
        """Unique ID without name should have default prefix."""
        uid = unique_id()
        assert isinstance(uid, str)
        assert len(uid) > 0

    def test_unique_id_with_random_chars(self):
        """Unique ID should include random characters."""
        uid = unique_id(name="test", random_chars=8)
        parts = uid.split("-")
        assert len(parts) >= 2
        # Last part should be the random string
        assert len(parts[-1]) == 8

    def test_unique_ids_are_different(self):
        """Multiple calls should produce different IDs."""
        uid1 = unique_id(name="test")
        uid2 = unique_id(name="test")
        assert uid1 != uid2


class TestRandomString:
    """Tests for random string generation."""

    def test_short_randomstr(self):
        """Random string should have correct length."""
        random_str = short_randomstr(num=8)
        assert len(random_str) == 8
        assert random_str.isalnum()

    def test_randomstr_different_calls(self):
        """Multiple calls should produce different strings."""
        str1 = short_randomstr(num=10)
        str2 = short_randomstr(num=10)
        # Very unlikely to be equal with 10 random characters
        assert str1 != str2

    def test_randomstr_various_lengths(self):
        """Random string should work with various lengths."""
        for length in [1, 5, 10, 20, 50]:
            random_str = short_randomstr(num=length)
            assert len(random_str) == length

    def test_default_random_chars_constant(self):
        """DEFAULT_RANDOM_CHARS should be defined."""
        assert isinstance(DEFAULT_RANDOM_CHARS, int)
        assert DEFAULT_RANDOM_CHARS > 0
