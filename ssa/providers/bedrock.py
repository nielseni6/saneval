import json
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError
from PIL import Image
from pydantic import BaseModel

from ssa.schemas import validate_response_against_schema
from ssa.utils.images import encode_image
from ssa.utils.logging import get_log
from ssa.utils.system import apply_overrides

log = get_log(__file__)

# Claude model versions available on Bedrock
CLAUDE_4_OPUS = "bedrock/claude-4-opus"
CLAUDE_4_SONNET = "bedrock/claude-4-sonnet"
CLAUDE_37_SONNET = "bedrock/claude-3-7-sonnet"


class BedrockConfig(BaseModel):
    model_id: str = ""
    max_retries: int = 7
    temperature: float = 0.0
    max_tokens: int = 4096
    system_prompt: str = ""


# See https://docs.aws.amazon.com/bedrock/ for model configurations
SUPPORTED_VERSIONS = {
    CLAUDE_4_OPUS: BedrockConfig(
        model_id="us.anthropic.claude-opus-4-20250514-v1:0",  # Inference profile ID
        max_tokens=8192,
    ),
    CLAUDE_4_SONNET: BedrockConfig(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Inference profile ID
        max_tokens=8192,
    ),
    CLAUDE_37_SONNET: BedrockConfig(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # Inference profile ID
        max_tokens=8192,
    ),
}

DEFAULT_VERSION = CLAUDE_4_SONNET


def fix_json_schema(schema):
    """Fix JSON schema for Bedrock compatibility"""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        for _k, v in schema.get("properties", {}).items():
            fix_json_schema(v)


class BedrockProvider:
    def __init__(self, version=DEFAULT_VERSION, config_overrides=None):
        self.version = version
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported Bedrock Claude version: {version}")

        self.config = apply_overrides(SUPPORTED_VERSIONS[version], config_overrides)
        # Use boto3 directly - it will use AWS credentials from environment, config, or IAM role
        self.client = boto3.client("bedrock-runtime")

    def _prepare_messages(self, query, image=None):
        """Prepare messages in Claude's format"""
        if isinstance(query, str):
            content: List[Dict[str, Any]] = [{"type": "text", "text": query}]
        else:
            content = query

        # Add image if provided
        if image is not None:
            if isinstance(image, str):
                # Assume it's a base64 encoded image or path
                if image.startswith(("data:", "http")):
                    # Extract base64 from data URL or handle URL
                    if image.startswith("data:"):
                        image_data = image.split(",")[1]
                    else:
                        raise ValueError(
                            "URL images not supported, please provide base64 or PIL Image"
                        )
                else:
                    image_data = image
            elif isinstance(image, Image.Image):
                # Convert PIL image to base64
                image_data = encode_image(image)
            else:
                raise ValueError("Image must be base64 string or PIL Image")

            # Add image to content
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                }
            )

        return [{"role": "user", "content": content}]

    def _prepare_tool_config(self, schema):
        """Prepare tool configuration for structured output"""
        if not schema:
            return None

        # Fix schema for Bedrock compatibility
        schema_copy = json.loads(json.dumps(schema))
        fix_json_schema(schema_copy)

        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": "structured_output",
                        "description": "Generate structured output according to the provided schema",
                        "inputSchema": {"json": schema_copy},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "structured_output"}},
        }

    def call(self, query, image=None, schema=None, seed=None, temperature=None):
        """Make a call to Claude via Bedrock"""
        # Note: seed parameter not currently supported by Bedrock Claude API
        _ = seed  # Unused parameter
        try:
            messages = self._prepare_messages(query, image)

            # Build request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "messages": messages,
                "temperature": (
                    temperature if temperature is not None else self.config.temperature
                ),
            }

            # Add system message if needed
            if hasattr(self.config, "system_prompt") and self.config.system_prompt:
                body["system"] = self.config.system_prompt

            # Handle structured output via tools
            tool_config = None
            if schema:
                tool_config = self._prepare_tool_config(schema)
                if tool_config:
                    body.update(tool_config)

            # Make the request
            response = self.client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            # Extract content
            if schema and tool_config:
                # Handle structured output from tool use
                content = response_body.get("content", [])
                for item in content:
                    if item.get("type") == "tool_use":
                        structured_data = item.get("input", {})
                        if schema:
                            validate_response_against_schema(structured_data, schema)
                        return structured_data

                # Fallback if no tool use found
                text_content = ""
                for item in content:
                    if item.get("type") == "text":
                        text_content += item.get("text", "")

                # Try to parse as JSON if schema provided
                if text_content.strip():
                    try:
                        parsed_json = json.loads(text_content.strip())
                        if schema:
                            validate_response_against_schema(parsed_json, schema)
                        return parsed_json
                    except json.JSONDecodeError:
                        log.warning("Failed to parse response as JSON, returning text")
                        return text_content

                return text_content
            else:
                # Handle regular text response
                content = response_body.get("content", [])
                text_response = ""
                for item in content:
                    if item.get("type") == "text":
                        text_response += item.get("text", "")

                return text_response.strip()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            log.error(f"Bedrock Claude error {error_code}: {error_message}")
            raise Exception(f"Bedrock Claude API error: {error_message}")
        except Exception as e:
            log.error(f"Unexpected error calling Bedrock Claude: {e}")
            raise

    def warmup(self, wait=False):
        """Warm up the provider by making a simple test call"""
        _ = wait  # Unused parameter for compatibility with interface
        try:
            log.debug(f"Warming up Bedrock Claude {self.version}")
            self.call("Hello", temperature=0)
            log.debug(f"Bedrock Claude {self.version} warmed up successfully")
        except Exception as e:
            log.warning(f"Bedrock Claude warmup failed: {e}")
