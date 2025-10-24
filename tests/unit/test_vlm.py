"""
Unit tests for VLM and LLM classes.

Tests cover:
- Model initialization and configuration
- Error handling for unsupported versions
- Caching functionality
- Model call functionality
- Trace recording
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from PIL import Image

from ssa.vlm import (
    LLM_VERSIONS,
    VLM_VERSIONS,
    Llm,
    Vlm,
    _cached_model_call,
    _format_unsupported_version_error,
    _try_get_from_cache,
    _try_put_to_cache,
)


class TestFormatUnsupportedVersionError:
    """Tests for the _format_unsupported_version_error function."""

    def test_format_error_message(self):
        """Test that error message is formatted correctly."""
        available = {"v1": Mock, "v2": Mock}
        error = _format_unsupported_version_error("VLM", "v3", available)
        assert "Unsupported VLM version: 'v3'" in error
        assert "'v1'" in error
        assert "'v2'" in error


class TestCacheHelpers:
    """Tests for cache helper functions."""

    def test_try_get_from_cache_hit(self):
        """Test successful cache retrieval."""
        mock_cache = Mock()
        mock_cache.get.return_value = "cached_response"

        result = _try_get_from_cache(
            mock_cache, "model_key", "query", None, None, None, None
        )

        assert result == "cached_response"
        mock_cache.get.assert_called_once()

    def test_try_get_from_cache_miss(self):
        """Test cache miss returns None."""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        result = _try_get_from_cache(
            mock_cache, "model_key", "query", None, None, None, None
        )

        assert result is None

    def test_try_get_from_cache_error(self):
        """Test cache error handling returns None."""
        mock_cache = Mock()
        mock_cache.get.side_effect = ValueError("Cache error")

        result = _try_get_from_cache(
            mock_cache, "model_key", "query", None, None, None, None
        )

        assert result is None

    def test_try_put_to_cache_success(self):
        """Test successful cache storage."""
        mock_cache = Mock()

        _try_put_to_cache(
            mock_cache,
            "model_key",
            "query",
            "response",
            None,
            None,
            None,
            None,
            {"metadata": "test"},
        )

        mock_cache.put.assert_called_once()

    def test_try_put_to_cache_error(self):
        """Test cache storage error is handled gracefully."""
        mock_cache = Mock()
        mock_cache.put.side_effect = ValueError("Cache error")

        # Should not raise exception
        _try_put_to_cache(
            mock_cache,
            "model_key",
            "query",
            "response",
            None,
            None,
            None,
            None,
            {"metadata": "test"},
        )


class TestLlm:
    """Tests for the Llm class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider class."""
        provider = Mock()
        provider.return_value.version = "1.0"
        provider.return_value.call = Mock(return_value="test_response")
        return provider

    def test_llm_init_with_valid_key(self, mock_provider):
        """Test LLM initialization with valid key."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            llm = Llm("test_llm", warmup=False)
            assert llm.key == "test_llm"
            assert llm.version == "1.0"

    def test_llm_init_with_invalid_key(self):
        """Test LLM initialization with invalid key raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM version"):
            Llm("invalid_key", warmup=False)

    def test_llm_config(self, mock_provider):
        """Test LLM config method returns correct information."""
        # Setup mock with iterable config
        mock_instance = Mock()
        mock_instance.version = "1.0"
        mock_instance.config = [("param1", "value1")]  # Make config iterable
        mock_provider.return_value = mock_instance

        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            llm = Llm("test_llm", warmup=False)
            config = llm.config()

            assert config["key"] == "test_llm"
            assert config["version"] == "1.0"
            assert "class" in config

    def test_llm_str_representation(self, mock_provider):
        """Test LLM string representation."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            llm = Llm("test_llm", warmup=False)
            assert str(llm) == "<LLM:test_llm>"

    def test_llm_call_without_cache(self, mock_provider):
        """Test LLM call without caching."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            llm = Llm("test_llm", warmup=False, enable_caching=False)

            with patch("ssa.vlm.get_global_cache", return_value=None):
                response = llm.call("test_query", use_cache=False)

                assert response == "test_response"
                llm.model.call.assert_called_once()

    def test_llm_call_with_schema(self, mock_provider):
        """Test LLM call with schema parameter."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            llm = Llm("test_llm", warmup=False, enable_caching=False)

            with patch("ssa.vlm.get_global_cache", return_value=None):
                schema = {"type": "object"}
                response = llm.call("test_query", schema=schema, use_cache=False)

                assert response == "test_response"
                llm.model.call.assert_called_with(
                    "test_query", image=None, schema=schema, seed=None, temperature=None
                )

    def test_llm_trace_recording(self, mock_provider):
        """Test that traces are recorded when trace_collector is provided."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            mock_trace_collector = Mock()
            llm = Llm(
                "test_llm",
                warmup=False,
                enable_caching=False,
                trace_collector=mock_trace_collector,
            )

            with patch("ssa.vlm.get_global_cache", return_value=None):
                llm.call("test_query", use_cache=False)

                mock_trace_collector.record_call.assert_called_once()

    def test_llm_warmup_called(self, mock_provider):
        """Test that warmup is called when enabled."""
        mock_instance = Mock()
        mock_instance.version = "1.0"
        mock_instance.warmup = Mock()
        mock_provider.return_value = mock_instance

        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            llm = Llm("test_llm", warmup=True)
            mock_instance.warmup.assert_called_once_with(wait=False)

    def test_llm_get_cache_stats(self, mock_provider):
        """Test get_cache_stats method."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            with patch("ssa.vlm.get_cache_statistics") as mock_stats:
                mock_stats.return_value = {"hits": 10, "misses": 5}
                llm = Llm("test_llm", warmup=False)

                stats = llm.get_cache_stats()
                assert stats == {"hits": 10, "misses": 5}

    def test_llm_setup_cache(self, mock_provider):
        """Test setup_cache method."""
        with patch.dict(LLM_VERSIONS, {"test_llm": mock_provider}):
            with patch("ssa.vlm.setup_caching") as mock_setup:
                llm = Llm("test_llm", warmup=False)
                llm.setup_cache(capacity=100, ttl=3600)

                mock_setup.assert_called_once_with(capacity=100, ttl=3600)


class TestVlm:
    """Tests for the Vlm class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider class."""
        provider = Mock()
        provider.return_value.version = "1.0"
        provider.return_value.call = Mock(return_value="test_response")
        return provider

    def test_vlm_init_with_valid_key(self, mock_provider):
        """Test VLM initialization with valid key."""
        with patch.dict(VLM_VERSIONS, {"test_vlm": mock_provider}):
            vlm = Vlm("test_vlm", warmup=False)
            assert vlm.key == "test_vlm"
            assert vlm.version == "1.0"

    def test_vlm_init_with_invalid_key(self):
        """Test VLM initialization with invalid key raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported VLM version"):
            Vlm("invalid_key", warmup=False)

    def test_vlm_str_representation(self, mock_provider):
        """Test VLM string representation."""
        with patch.dict(VLM_VERSIONS, {"test_vlm": mock_provider}):
            vlm = Vlm("test_vlm", warmup=False)
            assert str(vlm) == "<VLM:test_vlm>"

    def test_vlm_call_with_image(self, mock_provider):
        """Test VLM call with image parameter."""
        with patch.dict(VLM_VERSIONS, {"test_vlm": mock_provider}):
            vlm = Vlm("test_vlm", warmup=False, enable_caching=False)

            with patch("ssa.vlm.get_global_cache", return_value=None):
                mock_image = Mock(spec=Image.Image)
                response = vlm.call("test_query", image=mock_image, use_cache=False)

                assert response == "test_response"
                vlm.model.call.assert_called_with(
                    "test_query",
                    image=mock_image,
                    schema=None,
                    seed=None,
                    temperature=None,
                )

    def test_vlm_call_with_all_parameters(self, mock_provider):
        """Test VLM call with all optional parameters."""
        with patch.dict(VLM_VERSIONS, {"test_vlm": mock_provider}):
            vlm = Vlm("test_vlm", warmup=False, enable_caching=False)

            with patch("ssa.vlm.get_global_cache", return_value=None):
                mock_image = Mock(spec=Image.Image)
                schema = {"type": "object"}
                response = vlm.call(
                    "test_query",
                    image=mock_image,
                    schema=schema,
                    seed=42,
                    temperature=0.7,
                    use_cache=False,
                    scorer_name="test_scorer",
                    step_description="test_step",
                )

                assert response == "test_response"

    def test_vlm_inherits_from_llm(self, mock_provider):
        """Test that VLM inherits from Llm."""
        with patch.dict(VLM_VERSIONS, {"test_vlm": mock_provider}):
            vlm = Vlm("test_vlm", warmup=False)
            assert isinstance(vlm, Llm)


class TestCachedModelCall:
    """Tests for the _cached_model_call function."""

    @pytest.fixture
    def mock_model_instance(self):
        """Create a mock model instance."""
        instance = Mock()
        instance.key = "test_model"
        instance.enable_caching = False
        instance.model = Mock()
        instance.model.call = Mock(return_value="fresh_response")
        instance._record_trace = Mock()
        return instance

    def test_cached_model_call_cache_hit(self, mock_model_instance):
        """Test cached model call with cache hit."""
        mock_cache = Mock()
        mock_cache.get.return_value = "cached_response"

        with patch("ssa.vlm.get_global_cache", return_value=mock_cache):
            response = _cached_model_call(
                mock_model_instance, "llm", "test_query", use_cache=True
            )

            assert response == "cached_response"
            mock_model_instance.model.call.assert_not_called()

    def test_cached_model_call_cache_miss(self, mock_model_instance):
        """Test cached model call with cache miss."""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        with patch("ssa.vlm.get_global_cache", return_value=mock_cache):
            response = _cached_model_call(
                mock_model_instance, "llm", "test_query", use_cache=True
            )

            assert response == "fresh_response"
            mock_model_instance.model.call.assert_called_once()
            mock_cache.put.assert_called_once()

    def test_cached_model_call_no_cache(self, mock_model_instance):
        """Test cached model call without caching."""
        with patch("ssa.vlm.get_global_cache", return_value=None):
            response = _cached_model_call(
                mock_model_instance, "llm", "test_query", use_cache=False
            )

            assert response == "fresh_response"
            mock_model_instance.model.call.assert_called_once()

    def test_cached_model_call_with_trace(self, mock_model_instance):
        """Test that trace is recorded for cached calls."""
        mock_cache = Mock()
        mock_cache.get.return_value = "cached_response"

        with patch("ssa.vlm.get_global_cache", return_value=mock_cache):
            response = _cached_model_call(
                mock_model_instance,
                "llm",
                "test_query",
                use_cache=True,
                scorer_name="test_scorer",
                step_description="test_step",
            )

            mock_model_instance._record_trace.assert_called_once_with(
                "llm", "test_query", "cached_response", "test_scorer", "test_step"
            )
