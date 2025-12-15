"""Integration tests for Bedrock provider with saneval architecture."""

import json
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from ssa.providers.bedrock import LLAMA_4_MAVERICK
from ssa.vlm import Llm, Vlm


@pytest.fixture
def mock_bedrock_response():
    """Fixture for mocked Bedrock API response."""
    return {
        "body": Mock(
            read=lambda: json.dumps(
                {
                    "generation": "This is a test response",
                    "prompt_token_count": 50,
                    "generation_token_count": 20,
                }
            ).encode()
        )
    }


@pytest.fixture
def mock_bedrock_json_response():
    """Fixture for mocked Bedrock JSON response."""
    return {
        "body": Mock(
            read=lambda: json.dumps(
                {
                    "generation": '{"conformity": true, "explanation": "Test explanation"}',
                    "prompt_token_count": 100,
                    "generation_token_count": 30,
                }
            ).encode()
        )
    }


class TestLlmIntegration:
    """Tests for Llm wrapper with Bedrock provider."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_initialization(self, mock_report_cost, mock_session, mock_setup_aws):
        """Test Llm initialization with Bedrock model."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock warmup call
        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Hello",
                        "prompt_token_count": 5,
                        "generation_token_count": 2,
                    }
                ).encode()
            )
        }

        llm = Llm(LLAMA_4_MAVERICK)

        assert llm.key == LLAMA_4_MAVERICK
        assert llm.model is not None

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_call_basic(
        self, mock_report_cost, mock_session, mock_setup_aws, mock_bedrock_response
    ):
        """Test basic Llm call."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response

        llm = Llm(LLAMA_4_MAVERICK, warmup=False)
        response = llm.call("What is 2+2?")

        assert "test response" in response.lower()
        mock_client.invoke_model.assert_called()

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_call_with_temperature(
        self, mock_report_cost, mock_session, mock_setup_aws, mock_bedrock_response
    ):
        """Test Llm call with temperature override."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response

        llm = Llm(LLAMA_4_MAVERICK, warmup=False)
        response = llm.call("Tell me a story", temperature=0.8)

        # Verify temperature was passed through
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert body["temperature"] == 0.8

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_call_with_schema(
        self,
        mock_report_cost,
        mock_session,
        mock_setup_aws,
        mock_bedrock_json_response,
    ):
        """Test Llm call with structured output schema."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_json_response

        llm = Llm(LLAMA_4_MAVERICK, warmup=False)

        schema = {
            "type": "object",
            "properties": {
                "conformity": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
        }

        with patch("ssa.providers.bedrock.validate_response_against_schema"):
            response = llm.call("Check this", schema=schema)

        assert isinstance(response, dict)
        assert "conformity" in response
        assert "explanation" in response

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_llm_config_overrides(self, mock_session, mock_setup_aws):
        """Test Llm with config overrides."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Hi",
                        "prompt_token_count": 3,
                        "generation_token_count": 1,
                    }
                ).encode()
            )
        }

        llm = Llm(
            LLAMA_4_MAVERICK,
            config_overrides={"max_gen_len": 512, "system_prompt": "Be brief"},
            warmup=False,
        )

        assert llm.model.config.max_gen_len == 512
        assert llm.model.config.system_prompt == "Be brief"


class TestVlmIntegration:
    """Tests for Vlm wrapper with Bedrock provider."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_vlm_initialization(self, mock_report_cost, mock_session, mock_setup_aws):
        """Test Vlm initialization with Bedrock model."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock warmup
        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Hello",
                        "prompt_token_count": 5,
                        "generation_token_count": 2,
                    }
                ).encode()
            )
        }

        vlm = Vlm(LLAMA_4_MAVERICK)

        assert vlm.key == LLAMA_4_MAVERICK
        assert vlm.model is not None

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    @patch("ssa.providers.bedrock.encode_image")
    def test_vlm_call_with_image(
        self,
        mock_encode,
        mock_report_cost,
        mock_session,
        mock_setup_aws,
        mock_bedrock_response,
    ):
        """Test Vlm call with image input."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response
        mock_encode.return_value = "base64imagedata"

        vlm = Vlm(LLAMA_4_MAVERICK, warmup=False)
        image = Image.new("RGB", (100, 100))
        response = vlm.call("Describe this image", image=image)

        assert response is not None
        mock_encode.assert_called_once()

        # Verify image was included in request
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert "images" in body
        assert body["images"] == ["base64imagedata"]

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_vlm_call_text_only(
        self, mock_report_cost, mock_session, mock_setup_aws, mock_bedrock_response
    ):
        """Test Vlm call without image (text-only)."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response

        vlm = Vlm(LLAMA_4_MAVERICK, warmup=False)
        response = vlm.call("What is AI?")

        assert response is not None

        # Verify no images in request
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert "images" not in body


class TestCachingIntegration:
    """Tests for caching integration with Bedrock."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_with_caching_enabled(
        self, mock_report_cost, mock_session, mock_setup_aws, mock_bedrock_response
    ):
        """Test Llm with caching enabled."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response

        llm = Llm(LLAMA_4_MAVERICK, warmup=False, enable_caching=True)

        # First call should hit the API
        response1 = llm.call("Test query", use_cache=True)
        assert response1 is not None

        # Note: In a real integration test with actual cache, we'd verify
        # that the second call doesn't hit the API. Here we just verify
        # the interface works.

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_with_caching_disabled(
        self, mock_report_cost, mock_session, mock_setup_aws, mock_bedrock_response
    ):
        """Test Llm with caching disabled."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response

        llm = Llm(LLAMA_4_MAVERICK, warmup=False, enable_caching=False)

        # Call should work without caching
        response = llm.call("Test query", use_cache=False)
        assert response is not None


class TestTraceCollectionIntegration:
    """Tests for trace collection integration."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_with_trace_collector(
        self, mock_report_cost, mock_session, mock_setup_aws, mock_bedrock_response
    ):
        """Test Llm with trace collector."""
        from ssa.utils.reasoning_trace import ReasoningTraceCollector

        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.return_value = mock_bedrock_response

        trace_collector = ReasoningTraceCollector()
        llm = Llm(LLAMA_4_MAVERICK, warmup=False, trace_collector=trace_collector)

        response = llm.call(
            "Test query", scorer_name="test_scorer", step_description="test_step"
        )

        assert response is not None
        # Verify trace was recorded
        traces = trace_collector.get_traces()
        assert len(traces) > 0
        assert traces[0]["model_key"] == LLAMA_4_MAVERICK
        assert traces[0]["scorer_name"] == "test_scorer"


class TestErrorHandlingIntegration:
    """Tests for error handling in integration."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_llm_with_invalid_model_key(self, mock_session, mock_setup_aws):
        """Test that invalid model key raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported.*version"):
            Llm("bedrock/invalid-model", warmup=False)

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_llm_api_error_propagates(
        self, mock_report_cost, mock_session, mock_setup_aws
    ):
        """Test that API errors propagate correctly."""
        from botocore.exceptions import ClientError

        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock API error
        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "invoke_model",
        )
        mock_client.invoke_model.side_effect = error

        llm = Llm(LLAMA_4_MAVERICK, warmup=False)

        with pytest.raises(Exception, match="Bedrock Llama API error"):
            llm.call("Test query")


class TestConformitySchema:
    """Tests for ConformityWithExplanation schema (common use case)."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_conformity_schema_usage(
        self, mock_report_cost, mock_session, mock_setup_aws
    ):
        """Test using ConformityWithExplanation schema."""
        from ssa.schemas import ConformityWithExplanation

        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock response with conformity JSON
        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": '{"conformity": true, "explanation": "Object matches prompt description"}',
                        "prompt_token_count": 150,
                        "generation_token_count": 40,
                    }
                ).encode()
            )
        }

        vlm = Vlm(LLAMA_4_MAVERICK, warmup=False)

        with patch("ssa.providers.bedrock.validate_response_against_schema"):
            response = vlm.call(
                "Is there a dog in this image?",
                image=Image.new("RGB", (100, 100)),
                schema=ConformityWithExplanation,
            )

        assert isinstance(response, dict)
        assert "conformity" in response
        assert "explanation" in response
        assert response["conformity"] is True


class TestMultipleModels:
    """Tests for using multiple models in same session."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.gemini.genai.Client")
    @patch("ssa.providers.gemini.get_secret")
    @patch("ssa.providers.bedrock.report_cost")
    def test_bedrock_and_gemini_together(
        self,
        mock_report_cost,
        mock_get_secret,
        mock_gemini_client,
        mock_session,
        mock_setup_aws,
    ):
        """Test using both Bedrock and Gemini models in same session."""
        from ssa.providers.gemini import GEMINI_2_5_FLASH

        # Setup Bedrock mock
        mock_bedrock_client = Mock()
        mock_session.return_value.client.return_value = mock_bedrock_client
        mock_bedrock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Bedrock response",
                        "prompt_token_count": 10,
                        "generation_token_count": 5,
                    }
                ).encode()
            )
        }

        # Setup Gemini mock
        mock_get_secret.return_value = "fake_key"
        mock_gemini_instance = Mock()
        mock_gemini_client.return_value = mock_gemini_instance
        mock_gemini_response = Mock()
        mock_gemini_response.text = "Gemini response"
        mock_gemini_instance.models.generate_content.return_value = mock_gemini_response

        # Create both models
        llm_bedrock = Llm(LLAMA_4_MAVERICK, warmup=False)
        llm_gemini = Llm(GEMINI_2_5_FLASH, warmup=False)

        # Both should work independently
        response_bedrock = llm_bedrock.call("Test Bedrock")
        response_gemini = llm_gemini.call("Test Gemini")

        assert "Bedrock" in response_bedrock
        assert "Gemini" in response_gemini
