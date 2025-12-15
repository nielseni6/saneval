"""AWS Bedrock provider for Llama 4 models.

This module provides access to Llama 4 models via AWS Bedrock. Currently
supports Llama 4 Maverick (17B instruct model) with vision capabilities.

Example usage:
    from ssa.providers.bedrock import BedrockProvider, LLAMA_4_MAVERICK

    provider = BedrockProvider(version=LLAMA_4_MAVERICK)
    response = provider.call("What is machine learning?")

    # With vision
    from PIL import Image
    image = Image.open("photo.jpg")
    response = provider.call("Describe this image", image=image)
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image
from pydantic import BaseModel

from ssa.aws import setup_aws
from ssa.schemas import validate_response_against_schema
from ssa.utils.costs import report_cost
from ssa.utils.images import encode_image
from ssa.utils.logging import get_log
from ssa.utils.system import apply_overrides

log = get_log(__file__)

# Llama model versions available on Bedrock
LLAMA_4_MAVERICK = "bedrock/llama-4-maverick-17b-instruct"

MILLION = 1000000


class LlamaBedrockConfig(BaseModel):
    """Configuration for Llama models on Bedrock."""

    model_id: str = ""
    max_retries: int = 7
    temperature: float = 0.0
    max_gen_len: int = 2048  # Max response tokens for Llama
    top_p: float = 0.9
    system_prompt: str = ""
    # Pricing per million tokens - TODO: Verify from AWS Bedrock pricing page
    cost_mm_input: float = 0.0008  # Placeholder pricing
    cost_mm_output: float = 0.001  # Placeholder pricing
    supports_vision: bool = False


# Llama model configurations with verified Bedrock model IDs
# Note: Pricing is PLACEHOLDER and should be verified from AWS Bedrock pricing page
SUPPORTED_VERSIONS = {
    LLAMA_4_MAVERICK: LlamaBedrockConfig(
        model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
        cost_mm_input=0.0008,  # TODO: Verify actual Bedrock pricing
        cost_mm_output=0.001,  # TODO: Verify actual Bedrock pricing
        max_gen_len=2048,
        supports_vision=True,
    ),
}

DEFAULT_VERSION = LLAMA_4_MAVERICK


def get_llama_cost(config: LlamaBedrockConfig, response_body: dict) -> float:
    """
    Calculate cost based on token usage for Llama models.

    Args:
        config: Llama configuration with pricing info
        response_body: API response containing token counts

    Returns:
        Cost in dollars
    """
    input_tokens = response_body.get("prompt_token_count", 0)
    output_tokens = response_body.get("generation_token_count", 0)
    cost = (input_tokens * config.cost_mm_input / MILLION) + (
        output_tokens * config.cost_mm_output / MILLION
    )
    log.debug(
        f"Llama cost calculation: input={input_tokens} tokens, "
        f"output={output_tokens} tokens, cost=${cost:.6f}"
    )
    return cost


class BedrockProvider:
    """Provider for accessing Llama 4 models via AWS Bedrock."""

    def __init__(
        self, version: str = DEFAULT_VERSION, config_overrides: Optional[Dict] = None
    ):
        """
        Initialize Bedrock provider for Llama 4 models.

        Args:
            version: Model version key (e.g., "bedrock/llama-4-maverick-17b-instruct")
            config_overrides: Optional dict of config overrides

        Raises:
            ValueError: If unsupported version is specified
        """
        self.version = version
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported Bedrock version: {version}. "
                f"Supported: {list(SUPPORTED_VERSIONS.keys())}"
            )

        self.config = apply_overrides(SUPPORTED_VERSIONS[version], config_overrides)

        # Ensure AWS is configured
        setup_aws()

        # Initialize Bedrock client with retry configuration
        boto_config = Config(
            retries={
                "max_attempts": self.config.max_retries,
                "mode": "adaptive",
            }
        )

        profile = os.environ.get("AWS_PROFILE")
        if profile:
            session = boto3.Session(profile_name=profile)
        else:
            session = boto3.Session()

        self.client = session.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            config=boto_config,
        )

        log.debug(
            f"Initialized Bedrock Llama provider: {self.version} "
            f"(max_retries={self.config.max_retries})"
        )

    def _format_prompt_llama(
        self,
        query: Union[str, List[Dict]],
        system_prompt: str = "",
        has_image: bool = False,
    ) -> str:
        """
        Format prompt using Llama's special token syntax.

        Llama uses specific control tokens for chat formatting:
        - <|begin_of_text|>: Start of text
        - <|start_header_id|>role<|end_header_id|>: Role headers
        - <|eot_id|>: End of turn
        - <|image|>: Image placeholder (for vision models)

        Args:
            query: User query as string or conversation history as list of dicts
            system_prompt: Optional system prompt
            has_image: Whether an image is included (adds <|image|> token)

        Returns:
            Formatted prompt string with Llama control tokens
        """
        prompt_parts = []

        # For vision models, add <|image|> token before <|begin_of_text|>
        if has_image:
            prompt_parts.append("<|image|>")
            log.debug("Added <|image|> token for vision model")

        prompt_parts.append("<|begin_of_text|>")

        # Add system prompt if provided
        if system_prompt:
            prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n")
            prompt_parts.append(f"{system_prompt}<|eot_id|>")

        # Handle query as string or conversation history
        if isinstance(query, str):
            prompt_parts.append("<|start_header_id|>user<|end_header_id|>\n")
            prompt_parts.append(f"{query}<|eot_id|>")
        elif isinstance(query, list):
            # Multi-turn conversation
            for message in query:
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n")
                prompt_parts.append(f"{content}<|eot_id|>")
        else:
            raise ValueError("Query must be string or list of message dicts")

        # Add assistant header to trigger response
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")

        return "".join(prompt_parts)

    def _add_schema_instructions_llama(self, prompt: str, schema: dict) -> str:
        """
        Add schema instructions to Llama prompt for structured output.

        Since Llama doesn't have native JSON mode like some other models,
        we use prompt engineering to instruct it to return valid JSON.

        Args:
            prompt: Formatted prompt
            schema: JSON schema dict or Pydantic model class

        Returns:
            Prompt with schema instructions inserted
        """
        # Handle Pydantic models - convert to JSON schema
        if hasattr(schema, "model_json_schema"):
            # Pydantic v2
            schema = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            # Pydantic v1
            schema = schema.schema()

        schema_str = json.dumps(schema, indent=2)
        structured_instruction = (
            f"\n\nYou must respond with valid JSON matching this exact schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- Respond ONLY with the JSON object\n"
            f"- Do not include any explanations, reasoning, or additional text\n"
            f"- Do not include any special tokens like <|eot_id|> or <|start_header_id|>\n"
            f"- Do not add markdown code blocks around the JSON\n"
            f"- Output must be pure, valid JSON that can be parsed directly\n"
        )

        # Insert instruction before assistant header
        parts = prompt.rsplit("<|start_header_id|>assistant<|end_header_id|>", 1)
        if len(parts) == 2:
            return (
                parts[0]
                + structured_instruction
                + "<|start_header_id|>assistant<|end_header_id|>"
                + parts[1]
            )
        else:
            return prompt + structured_instruction

    def _parse_structured_output_llama(self, generation: str, schema: dict) -> dict:
        """
        Parse structured output from Llama generation.

        Attempts to extract JSON from the generation, handling common cases:
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON wrapped in generic code blocks (``` ... ```)
        - Raw JSON text

        Args:
            generation: Raw generation from Llama
            schema: JSON schema for validation

        Returns:
            Parsed and validated JSON dict

        Raises:
            ValueError: If JSON parsing or validation fails
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in generation:
            start = generation.find("```json") + 7
            end = generation.find("```", start)
            json_text = generation[start:end].strip()
        elif "```" in generation:
            start = generation.find("```") + 3
            end = generation.find("```", start)
            json_text = generation[start:end].strip()
        else:
            json_text = generation.strip()

        try:
            parsed = json.loads(json_text)
            validate_response_against_schema(parsed, schema)
            return parsed
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse structured output: {e}")
            log.debug(f"Raw generation: {generation}")
            raise ValueError(f"Model did not return valid JSON: {e}")

    def _clean_response_llama(self, generation: str) -> str:
        """
        Remove Llama control tokens from response.

        Args:
            generation: Raw generation from Llama

        Returns:
            Cleaned text with control tokens removed
        """
        tokens_to_strip = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "assistant<|end_header_id|>",
            "<|begin_of_text|>",
        ]
        cleaned = generation
        for token in tokens_to_strip:
            cleaned = cleaned.replace(token, "")
        return cleaned.strip()

    def _prepare_request_body_llama(
        self,
        query: Union[str, List[Dict]],
        image: Optional[Any] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Prepare request body for Llama Bedrock API.

        Args:
            query: User query or conversation history
            image: Optional image (PIL Image or base64 string)
            temperature: Optional temperature override

        Returns:
            Request body dict for Bedrock API

        Raises:
            ValueError: If image provided for non-vision model
        """
        # Check if image provided for vision models
        if image is not None:
            if not self.config.supports_vision:
                raise ValueError(
                    f"Model {self.version} does not support vision input. "
                    f"Vision-capable models: {[k for k, v in SUPPORTED_VERSIONS.items() if v.supports_vision]}"
                )

        # Format prompt with special tokens
        system_prompt = self.config.system_prompt
        formatted_prompt = self._format_prompt_llama(
            query, system_prompt, has_image=(image is not None)
        )

        body = {
            "prompt": formatted_prompt,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_gen_len": self.config.max_gen_len,
            "top_p": self.config.top_p,
        }

        # Add images for vision-capable models
        if image is not None:
            if isinstance(image, str):
                image_data = image
            elif isinstance(image, Image.Image):
                image_data = encode_image(image)
            else:
                raise ValueError("Image must be base64 string or PIL Image")
            body["images"] = [image_data]

        return body

    def call(
        self,
        query: Union[str, List[Dict]],
        image: Optional[Union[Image.Image, str]] = None,
        video: Optional[Any] = None,
        schema: Optional[dict] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[str, dict]:
        """
        Make a call to Llama via AWS Bedrock.

        Args:
            query: Text query or conversation history
            image: Optional image (PIL Image or base64 string)
            video: Video input (not supported, will raise error if provided)
            schema: Optional JSON schema for structured output
            seed: Random seed (not supported by Bedrock Llama, will be ignored)
            temperature: Optional temperature override (0.0-1.0)

        Returns:
            String response or dict if schema provided

        Raises:
            ValueError: If video input provided (not supported)
            Exception: If API call fails

        Note:
            - Seed parameter is not supported by Bedrock Llama and will be ignored
            - Video input is not supported
            - Structured output uses prompt engineering, not native JSON mode
        """
        # Note: Bedrock Llama doesn't support seed parameter
        if seed is not None:
            log.warning("Seed parameter not supported by Bedrock Llama, ignoring")

        if video is not None:
            raise ValueError(
                f"Video input is not supported by Bedrock Llama models. "
                f"Model: {self.version}"
            )

        try:
            # Prepare request body
            body = self._prepare_request_body_llama(
                query=query, image=image, temperature=temperature
            )

            # Add structured output instructions if schema provided
            if schema:
                body["prompt"] = self._add_schema_instructions_llama(
                    body["prompt"], schema
                )

            # Make API call
            log.debug(f"Calling Bedrock API: model={self.config.model_id}")
            response = self.client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            # Calculate and report cost
            cost = get_llama_cost(self.config, response_body)
            report_cost(cost, name=f"bedrock_llama_{self.version}")

            # Extract generation
            generation = response_body.get("generation", "").strip()

            # Clean control tokens from response
            generation = self._clean_response_llama(generation)

            # Handle structured output if schema provided
            if schema:
                return self._parse_structured_output_llama(generation, schema)

            return generation

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            log.error(f"Bedrock Llama error {error_code}: {error_message}")
            raise Exception(f"Bedrock Llama API error: {error_message}") from e
        except Exception as e:
            log.error(f"Unexpected error calling Bedrock Llama: {e}")
            raise

    def warmup(self, wait: bool = False) -> None:
        """
        Warm up the provider by making a simple test call.

        Args:
            wait: Unused parameter for interface compatibility
        """
        _ = wait  # Unused parameter for compatibility with interface
        try:
            log.debug(f"Warming up Bedrock Llama {self.version}")
            self.call("Hello", temperature=0)
            log.debug(f"Bedrock Llama {self.version} warmed up successfully")
        except Exception as e:
            log.warning(f"Bedrock Llama warmup failed: {e}")
