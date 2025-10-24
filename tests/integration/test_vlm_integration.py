"""
Integration tests for VLM and LLM components.

These tests verify the integration between VLM/LLM classes and their providers,
caching systems, and trace collectors.
"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from ssa.vlm import Llm, Vlm


@pytest.fixture(autouse=True)
def mock_secrets():
    """Mock secrets retrieval to prevent API key lookups."""
    with patch("ssa.utils.secrets.get_secret", return_value="mock_api_key"):
        yield


@pytest.fixture(autouse=True)
def mock_genai_client():
    """Mock Google GenAI client to prevent real API connections."""
    mock_client = Mock()
    mock_client._api_client = Mock()  # Prevent __del__ AttributeError
    mock_client.close = Mock()
    with patch("google.genai.Client", return_value=mock_client):
        yield


@pytest.fixture
def mock_provider_factory():
    """Create a mock provider factory for testing."""

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
class TestVlmLlmIntegration:
    """Integration tests for VLM and LLM classes."""

    def test_llm_initialization_and_call(self, mock_provider_factory):
        """Test LLM can be initialized and called."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")

        # Patch the LLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create LLM instance
            llm = Llm("gemini/2.5-flash", warmup=False, enable_caching=False)

            # Call the model
            with patch("ssa.vlm.get_global_cache", return_value=None):
                response = llm.call("What is 2+2?", use_cache=False)

            assert response == "test response"
            mock_provider.return_value.call.assert_called_once()

    def test_vlm_initialization_and_call_with_image(self, mock_provider_factory):
        """Test VLM can be initialized and called with an image."""
        # Create mock provider with custom response
        mock_provider = mock_provider_factory("gemini-2.5-flash")
        mock_provider.return_value.call = Mock(return_value="A cat sitting on a table")

        # Patch the VLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.VLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create VLM instance
            vlm = Vlm("gemini/2.5-flash", warmup=False, enable_caching=False)

            # Create a simple test image
            test_image = Image.new("RGB", (100, 100), color="red")

            # Call the model
            with patch("ssa.vlm.get_global_cache", return_value=None):
                response = vlm.call(
                    "Describe this image", image=test_image, use_cache=False
                )

            assert response == "A cat sitting on a table"
            mock_provider.return_value.call.assert_called_once()

    def test_llm_with_caching_enabled(self, mock_provider_factory):
        """Test LLM with caching enabled."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")
        mock_provider.return_value.call = Mock(return_value="cached response")

        # Patch the LLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create LLM with caching enabled
            llm = Llm("gemini/2.5-flash", warmup=False, enable_caching=True)

            # Setup mock cache
            mock_cache = Mock()
            mock_cache.get = Mock(return_value=None)  # First call - cache miss
            mock_cache.put = Mock()

            # Call the model twice
            with patch("ssa.vlm.get_global_cache", return_value=mock_cache):
                response1 = llm.call("What is AI?", use_cache=True)

                # Second call - should hit cache
                mock_cache.get = Mock(return_value="cached response")
                response2 = llm.call("What is AI?", use_cache=True)

            assert response1 == "cached response"
            assert response2 == "cached response"
            # First call should store in cache
            mock_cache.put.assert_called_once()

    def test_llm_with_trace_collector(self, mock_provider_factory):
        """Test LLM with trace collector."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")
        mock_provider.return_value.call = Mock(return_value="traced response")

        # Create mock trace collector
        mock_trace_collector = Mock()

        # Patch the LLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create LLM with trace collector
            llm = Llm(
                "gemini/2.5-flash",
                warmup=False,
                enable_caching=False,
                trace_collector=mock_trace_collector,
            )

            # Call the model
            with patch("ssa.vlm.get_global_cache", return_value=None):
                llm.call(
                    "Test query",
                    use_cache=False,
                    scorer_name="test_scorer",
                    step_description="test_step",
                )

            # Verify trace was recorded
            mock_trace_collector.record_call.assert_called_once()
            call_args = mock_trace_collector.record_call.call_args
            assert call_args[1]["model_type"] == "llm"
            assert call_args[1]["query"] == "Test query"
            assert call_args[1]["scorer_name"] == "test_scorer"

    def test_vlm_with_schema(self, mock_provider_factory):
        """Test VLM with structured output schema."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")
        mock_provider.return_value.call = Mock(
            return_value='{"objects": ["cat", "dog"]}'
        )

        # Patch the VLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.VLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create VLM instance
            vlm = Vlm("gemini/2.5-flash", warmup=False, enable_caching=False)

            # Define schema
            schema = {
                "type": "object",
                "properties": {
                    "objects": {"type": "array", "items": {"type": "string"}}
                },
            }

            test_image = Image.new("RGB", (100, 100), color="blue")

            # Call with schema
            with patch("ssa.vlm.get_global_cache", return_value=None):
                response = vlm.call(
                    "List objects in image",
                    image=test_image,
                    schema=schema,
                    use_cache=False,
                )

            assert "cat" in response or "dog" in response
            mock_provider.return_value.call.assert_called_once()

    def test_llm_with_temperature_and_seed(self, mock_provider_factory):
        """Test LLM with temperature and seed parameters."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")
        mock_provider.return_value.call = Mock(return_value="deterministic response")

        # Patch the LLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create LLM instance
            llm = Llm("gemini/2.5-flash", warmup=False, enable_caching=False)

            # Call with temperature and seed
            with patch("ssa.vlm.get_global_cache", return_value=None):
                response = llm.call(
                    "Generate text", temperature=0.7, seed=42, use_cache=False
                )

            assert response == "deterministic response"
            # Verify parameters were passed
            call_args = mock_provider.return_value.call.call_args
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["seed"] == 42

    def test_llm_config_override(self, mock_provider_factory):
        """Test LLM with configuration overrides."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")

        # Patch the LLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create LLM with config overrides
            config_overrides = {"max_tokens": 500}
            Llm("gemini/2.5-flash", warmup=False, config_overrides=config_overrides)

            # Verify provider was called with overrides
            mock_provider.assert_called_once_with(
                "gemini/2.5-flash", config_overrides=config_overrides
            )

    def test_vlm_inherits_llm_functionality(self, mock_provider_factory):
        """Test that VLM inherits all LLM functionality."""
        # Create mock provider
        mock_provider = mock_provider_factory("gemini-2.5-flash")

        # Patch the VLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.VLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create VLM instance
            vlm = Vlm("gemini/2.5-flash", warmup=False)

            # Verify VLM has LLM methods
            assert hasattr(vlm, "config")
            assert hasattr(vlm, "get_cache_stats")
            assert hasattr(vlm, "setup_cache")
            assert isinstance(vlm, Llm)

    def test_error_handling_in_cached_call(self, mock_provider_factory):
        """Test error handling in cached model calls."""
        # Create mock provider that raises an error
        mock_provider = mock_provider_factory("gemini-2.5-flash")
        mock_provider.return_value.call = Mock(side_effect=Exception("API Error"))

        # Patch the LLM_VERSIONS dictionary
        with patch.dict("ssa.vlm.LLM_VERSIONS", {"gemini/2.5-flash": mock_provider}):
            # Create LLM instance
            llm = Llm("gemini/2.5-flash", warmup=False, enable_caching=False)

            # Verify exception is raised
            with patch("ssa.vlm.get_global_cache", return_value=None):
                with pytest.raises(Exception, match="API Error"):
                    llm.call("Test query", use_cache=False)
