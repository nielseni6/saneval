"""Unit tests for Bedrock provider."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from ssa.providers.bedrock import LLAMA_4_MAVERICK, BedrockProvider, get_llama_cost


class TestBedrockProviderInitialization:
    """Tests for BedrockProvider initialization."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_init_with_valid_version(self, mock_session, mock_setup_aws):
        """Test initialization with valid Llama version."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        provider = BedrockProvider(version=LLAMA_4_MAVERICK)

        assert provider.version == LLAMA_4_MAVERICK
        assert provider.config is not None
        assert provider.client == mock_client
        mock_setup_aws.assert_called_once()

    @patch("ssa.providers.bedrock.setup_aws")
    def test_init_with_invalid_version(self, mock_setup_aws):
        """Test initialization with unsupported version raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported Bedrock version"):
            BedrockProvider(version="bedrock/invalid-model")

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_init_with_config_overrides(self, mock_session, mock_setup_aws):
        """Test initialization with config overrides."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        overrides = {
            "temperature": 0.7,
            "max_gen_len": 1024,
            "system_prompt": "You are helpful",
        }

        provider = BedrockProvider(version=LLAMA_4_MAVERICK, config_overrides=overrides)

        assert provider.config.temperature == 0.7
        assert provider.config.max_gen_len == 1024
        assert provider.config.system_prompt == "You are helpful"


class TestPromptFormatting:
    """Tests for Llama prompt formatting."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_format_prompt_simple_text(self, mock_session, mock_setup_aws):
        """Test formatting a simple text prompt."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        prompt = provider._format_prompt_llama("Hello, world!")

        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "Hello, world!" in prompt
        assert "<|eot_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_format_prompt_with_system(self, mock_session, mock_setup_aws):
        """Test formatting prompt with system message."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        prompt = provider._format_prompt_llama(
            "Hello", system_prompt="You are a helpful assistant"
        )

        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "You are a helpful assistant" in prompt

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_format_prompt_with_image(self, mock_session, mock_setup_aws):
        """Test formatting prompt with image token."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        prompt = provider._format_prompt_llama("Describe this", has_image=True)

        assert prompt.startswith("<|image|>")
        assert "<|begin_of_text|>" in prompt

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_format_prompt_conversation(self, mock_session, mock_setup_aws):
        """Test formatting multi-turn conversation."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What about 3+3?"},
        ]

        prompt = provider._format_prompt_llama(conversation)

        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "What about 3+3?" in prompt
        assert prompt.count("<|start_header_id|>user<|end_header_id|>") == 2
        assert prompt.count("<|start_header_id|>assistant<|end_header_id|>") >= 1


class TestStructuredOutput:
    """Tests for structured output handling."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_add_schema_instructions(self, mock_session, mock_setup_aws):
        """Test adding schema instructions to prompt."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        prompt = provider._format_prompt_llama("Extract name")
        prompt_with_schema = provider._add_schema_instructions_llama(prompt, schema)

        assert "JSON" in prompt_with_schema
        # Schema is pretty-printed in the prompt
        assert '"type"' in prompt_with_schema
        assert '"properties"' in prompt_with_schema
        assert "CRITICAL REQUIREMENTS" in prompt_with_schema

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_parse_json_from_markdown_block(self, mock_session, mock_setup_aws):
        """Test parsing JSON from markdown code block."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        generation = '```json\n{"name": "Alice", "age": 30}\n```'
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        with patch("ssa.providers.bedrock.validate_response_against_schema"):
            result = provider._parse_structured_output_llama(generation, schema)

        assert result == {"name": "Alice", "age": 30}

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_parse_json_from_generic_block(self, mock_session, mock_setup_aws):
        """Test parsing JSON from generic code block."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        generation = '```\n{"status": "success"}\n```'
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}

        with patch("ssa.providers.bedrock.validate_response_against_schema"):
            result = provider._parse_structured_output_llama(generation, schema)

        assert result == {"status": "success"}

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_parse_raw_json(self, mock_session, mock_setup_aws):
        """Test parsing raw JSON without code blocks."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        generation = '{"result": true}'
        schema = {"type": "object", "properties": {"result": {"type": "boolean"}}}

        with patch("ssa.providers.bedrock.validate_response_against_schema"):
            result = provider._parse_structured_output_llama(generation, schema)

        assert result == {"result": True}

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_parse_invalid_json_raises(self, mock_session, mock_setup_aws):
        """Test that invalid JSON raises ValueError."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        generation = "This is not JSON"
        schema = {"type": "object"}

        with pytest.raises(ValueError, match="did not return valid JSON"):
            provider._parse_structured_output_llama(generation, schema)


class TestResponseCleaning:
    """Tests for response cleaning."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_clean_response_removes_tokens(self, mock_session, mock_setup_aws):
        """Test that control tokens are removed from response."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        generation = (
            "<|begin_of_text|>Hello world<|eot_id|><|start_header_id|>assistant"
        )
        cleaned = provider._clean_response_llama(generation)

        assert "<|begin_of_text|>" not in cleaned
        assert "<|eot_id|>" not in cleaned
        assert "<|start_header_id|>" not in cleaned
        assert "Hello world" in cleaned


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_get_llama_cost(self):
        """Test Llama cost calculation."""
        from ssa.providers.bedrock import LlamaBedrockConfig

        config = LlamaBedrockConfig(
            model_id="test", cost_mm_input=0.0008, cost_mm_output=0.001
        )

        response_body = {"prompt_token_count": 1000, "generation_token_count": 500}

        cost = get_llama_cost(config, response_body)

        # Expected: (1000 * 0.0008 / 1M) + (500 * 0.001 / 1M)
        expected = (1000 * 0.0008 / 1_000_000) + (500 * 0.001 / 1_000_000)
        assert abs(cost - expected) < 1e-10


class TestBedrockProviderCall:
    """Tests for the main call method."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_call_text_only(self, mock_report_cost, mock_session, mock_setup_aws):
        """Test simple text-only call."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        # Mock Bedrock response
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Hello! How can I help?",
                        "prompt_token_count": 10,
                        "generation_token_count": 20,
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        provider = BedrockProvider()
        response = provider.call("Hello")

        assert "Hello! How can I help?" in response
        mock_client.invoke_model.assert_called_once()
        mock_report_cost.assert_called_once()

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    @patch("ssa.providers.bedrock.encode_image")
    def test_call_with_image(
        self, mock_encode, mock_report_cost, mock_session, mock_setup_aws
    ):
        """Test call with image input."""
        from PIL import Image

        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_encode.return_value = "base64encodedimage"

        # Mock Bedrock response
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "I see a cat",
                        "prompt_token_count": 100,
                        "generation_token_count": 50,
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        provider = BedrockProvider()
        image = Image.new("RGB", (100, 100))
        response = provider.call("Describe this", image=image)

        assert "I see a cat" in response
        mock_encode.assert_called_once()

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_call_with_video_raises(self, mock_session, mock_setup_aws):
        """Test that video input raises ValueError."""
        mock_session.return_value.client.return_value = MagicMock()
        provider = BedrockProvider()

        with pytest.raises(ValueError, match="Video input is not supported"):
            provider.call("Analyze this", video="video.mp4")

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_call_with_schema(self, mock_report_cost, mock_session, mock_setup_aws):
        """Test call with structured output schema."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        # Mock Bedrock response with JSON
        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": '{"result": "success"}',
                        "prompt_token_count": 50,
                        "generation_token_count": 10,
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        provider = BedrockProvider()
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        with patch("ssa.providers.bedrock.validate_response_against_schema"):
            response = provider.call("Test", schema=schema)

        assert isinstance(response, dict)
        assert response["result"] == "success"

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_call_with_client_error(self, mock_session, mock_setup_aws):
        """Test handling of Bedrock API errors."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        # Mock ClientError
        error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "invoke_model",
        )
        mock_client.invoke_model.side_effect = error

        provider = BedrockProvider()

        with pytest.raises(Exception, match="Bedrock Llama API error"):
            provider.call("Test")

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_call_ignores_seed_parameter(self, mock_session, mock_setup_aws):
        """Test that seed parameter is ignored (no error raised)."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Response",
                        "prompt_token_count": 10,
                        "generation_token_count": 5,
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        provider = BedrockProvider()

        # Should not raise an error even with seed parameter
        with patch("ssa.providers.bedrock.report_cost"):
            response = provider.call("Test", seed=42)

        # Verify the call succeeded
        assert response is not None


class TestWarmup:
    """Tests for warmup method."""

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    @patch("ssa.providers.bedrock.report_cost")
    def test_warmup_success(self, mock_report_cost, mock_session, mock_setup_aws):
        """Test successful warmup."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_response = {
            "body": Mock(
                read=lambda: json.dumps(
                    {
                        "generation": "Hello",
                        "prompt_token_count": 5,
                        "generation_token_count": 3,
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        provider = BedrockProvider()
        provider.warmup()  # Should not raise

    @patch("ssa.providers.bedrock.setup_aws")
    @patch("ssa.providers.bedrock.boto3.Session")
    def test_warmup_failure_handled(self, mock_session, mock_setup_aws):
        """Test that warmup failures are handled gracefully (no exception raised)."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("Network error")

        provider = BedrockProvider()

        # Should not raise even if warmup fails
        try:
            provider.warmup()
        except Exception:
            pytest.fail("Warmup should handle errors gracefully and not raise")
