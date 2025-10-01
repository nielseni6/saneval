"""Azure OpenAI provider for SSA image generation and vision language models.

This module provides a comprehensive, production-ready interface to Azure OpenAI services,
featuring optimized performance, robust error handling, and complete cost tracking.

Features:
- GPT-4o for vision-language tasks and multimodal chat completions
- GPT-Image-1 for advanced image generation and editing capabilities with multi-image support
- FLUX.1-Kontext-pro for advanced image generation with context understanding and native editing
- Memory-efficient image processing with lazy base64 encoding
- Comprehensive cost tracking with detailed token usage breakdown
- Structured output with JSON schema validation and Azure compatibility fixes
- Input validation for all parameters with descriptive error messages
- Flexible authentication (environment variables, secret management systems)

The provider uses:
- Efficient iterative schema processing to avoid performance issues
- Thread-safe cost calculation for both text and image generation
- Proper circular reference detection in JSON schemas
- Standardized error handling with actionable error messages

API Documentation:
- Azure OpenAI: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- GPT-Image-1: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/gpt-image-1
- FLUX.1-Kontext-pro: Native image-to-image generation with context understanding
- Pricing: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

Example usage:
    # Text completion with vision
    provider = AzureProvider(version="azure/gpt-4o")
    response = provider.call("Describe this image", image=my_image)

    # Image generation
    img_provider = AzureProvider(version="azure/gpt-image-1")
    image = img_provider.generate_image("A sunset over mountains", width=1024, height=1024)

    # FLUX.1-Kontext-pro for advanced image generation
    flux_provider = AzureProvider(version="azure/flux-kontext-pro")
    image = flux_provider.generate_image("A sunset over mountains", width=1024, height=1024)

    # Use specific API version for testing or compatibility
    provider = AzureProvider(version="azure/gpt-4o", api_version="2024-12-01")
"""

import base64
import copy
import json
import os
from collections import deque
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlunparse

import openai
import requests
from openai import AzureOpenAI
from PIL import Image
from pydantic import BaseModel, Field

from ssa.interfaces import BaseImageEditingMixin
from ssa.utils.aspect_ratios import resolve_aspect_ratio_and_dimensions
from ssa.utils.costs import report_cost
from ssa.utils.images import encode_image
from ssa.utils.logging import get_log
from ssa.utils.secrets import get_secret
from ssa.utils.system import apply_overrides

log = get_log(__file__)

# Azure OpenAI model versions
AZURE_GPT_4O = "azure/gpt-4o"
AZURE_GPT_IMAGE_1 = "azure/gpt-image-1"
AZURE_FLUX_KONTEXT_PRO = "azure/flux-kontext-pro"

# Azure deployment names (used in API calls)
GPT_IMAGE_1_DEPLOYMENT_NAME = "gpt-image-1"
FLUX_KONTEXT_PRO_DEPLOYMENT_NAME = "FLUX.1-Kontext-pro"

# Azure OpenAI API version configuration
# Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"  # API version for image generation models (GPT-Image-1 and FLUX.1-Kontext-pro)

# Mathematical constants
MILLION = 1000000

# API validation constants
# Based on Azure OpenAI API documentation: https://learn.microsoft.com/en-us/azure/ai-services/openai/
MIN_API_KEY_LENGTH = 10  # Minimum expected length for a valid Azure OpenAI API key
MAX_SCHEMA_SIZE_BYTES = 1024 * 1024  # 1MB limit for JSON schemas in Azure OpenAI API

# Image generation parameter limits
# Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/gpt-image-1
MIN_IMAGE_DIMENSION = 64  # Minimum width/height in pixels
MAX_IMAGE_DIMENSION = 2048  # Maximum width/height in pixels
MIN_IMAGE_COUNT = 1  # Minimum number of images to generate
MAX_IMAGE_COUNT = 10  # Maximum number of images per request
MIN_TEMPERATURE = 0.0  # Minimum temperature for chat completions
MAX_TEMPERATURE = 2.0  # Maximum temperature for chat completions

# Cost calculation constants
# Reference: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
IMAGE_TOKEN_ESTIMATION_RATIO = 0.1  # ~10% of input tokens typically used for image processing in multimodal requests
MAX_ESTIMATED_IMAGE_TOKENS = (
    1000  # Conservative upper bound for image tokens per request
)
FALLBACK_IMAGE_COST_ESTIMATE = (
    0.04  # Conservative estimate: $0.04 per 1024x1024 image when usage data unavailable
)

# HTTP request timeout constants
AZURE_IMAGE_EDIT_TIMEOUT = 300  # Timeout in seconds for Azure image edit API calls


class _AzureImageData:
    """Simple wrapper for Azure image response data."""

    def __init__(self, data_dict: Dict[str, Any]):
        # Azure returns b64_json in the data dict
        self.b64_json = data_dict.get("b64_json")
        self.url = data_dict.get("url")


class _AzureImageResponse:
    """Simple wrapper for Azure image response to match OpenAI SDK format."""

    def __init__(self, response_dict: Dict[str, Any]):
        # Wrap each data item to have proper attributes
        self.data = [_AzureImageData(item) for item in response_dict.get("data", [])]
        # Preserve usage data for accurate cost tracking
        usage_data = response_dict.get("usage")
        self.usage = SimpleNamespace(**usage_data) if usage_data else None


class AzureBaseConfig(BaseModel):
    """Base configuration for Azure OpenAI services.

    Contains common settings shared across all Azure OpenAI model types.
    """

    deployment_name: str = ""  # Azure deployment name for the model
    api_version: str = (
        DEFAULT_AZURE_API_VERSION  # Azure OpenAI API version (configurable)
    )
    max_retries: int = 7  # Maximum retry attempts for failed requests
    azure_endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", "")
    )  # Azure OpenAI service endpoint URL

    # Base supported parameters that all Azure models support
    @classmethod
    def get_supported_params(cls) -> set[str]:
        """Get the set of supported configuration parameters for this model."""
        return {"deployment_name", "api_version", "max_retries", "azure_endpoint"}


class AzureConfig(AzureBaseConfig):
    """Configuration for Azure Vision Language Models (VLM) like GPT-4o.

    Used for chat completions and vision analysis tasks.
    """

    temperature: float = 0.5  # Sampling temperature for response generation
    cost_mm_input: float = (
        5.0  # Cost per million input tokens (USD) - overridden in AZURE_SUPPORTED_VERSIONS
    )
    cost_mm_output: float = (
        15.0  # Cost per million output tokens (USD) - overridden in AZURE_SUPPORTED_VERSIONS
    )

    # Supported parameters for VLM models (extends base parameters)
    @classmethod
    def get_supported_params(cls) -> set[str]:
        """Get the set of supported configuration parameters for this model."""
        return super().get_supported_params() | {
            "temperature",
            "cost_mm_input",
            "cost_mm_output",
        }


class AzureImageConfig(AzureBaseConfig):
    """Configuration for Azure Image Generation models (GPT-Image-1).

    Supports text-to-image and image-to-image generation with token-based pricing.
    """

    # GPT-Image-1 pricing per 1M tokens
    cost_mm_input_text: float = 5.0  # Input text tokens: $5 per 1M tokens
    cost_mm_input_image: float = 10.0  # Input image tokens: $10 per 1M tokens
    cost_mm_output_image: float = 40.0  # Output image tokens: $40 per 1M tokens
    azure_image_endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_IMAGE_ENDPOINT", "")
    )  # Dedicated endpoint for image generation services

    # Supported parameters for image generation models (extends base parameters)
    @classmethod
    def get_supported_params(cls) -> set[str]:
        """Get the set of supported configuration parameters for this model."""
        return super().get_supported_params() | {
            "cost_mm_input_text",
            "cost_mm_input_image",
            "cost_mm_output_image",
            "azure_image_endpoint",
        }


class AzureFluxKontextConfig(AzureBaseConfig):
    """Configuration for Azure FLUX.1-Kontext-pro model.

    Supports text-to-image and image-to-image generation with per-image pricing.
    """

    # FLUX.1-Kontext-pro pricing
    cost_per_img: float = 0.04  # Cost per generated image (USD)
    azure_flux_endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_FLUX_ENDPOINT", "")
    )  # Dedicated endpoint for FLUX model

    # Supported parameters for FLUX model (extends base parameters, excludes unsupported ones)
    @classmethod
    def get_supported_params(cls) -> set[str]:
        """Get the set of supported configuration parameters for this model."""
        return super().get_supported_params() | {
            "cost_per_img",
            "azure_flux_endpoint",
            # Note: temperature and aspect_ratio are NOT supported by FLUX model
        }


# Supported Azure OpenAI model versions with their default configurations
# Pricing reference: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
AZURE_SUPPORTED_VERSIONS = {
    AZURE_GPT_4O: AzureConfig(
        deployment_name="gpt-4o",
        cost_mm_input=5.0,  # $5 per 1M input tokens
        cost_mm_output=15.0,  # $15 per 1M output tokens
    ),
    AZURE_GPT_IMAGE_1: AzureImageConfig(
        deployment_name="gpt-image-1",
        cost_mm_input_text=5.0,  # $5 per 1M input text tokens
        cost_mm_input_image=10.0,  # $10 per 1M input image tokens
        cost_mm_output_image=40.0,  # $40 per 1M output image tokens
    ),
    AZURE_FLUX_KONTEXT_PRO: AzureFluxKontextConfig(
        deployment_name="FLUX.1-Kontext-pro",
        cost_per_img=0.04,  # $0.04 per generated image
    ),
}

# Models that support native image-to-image editing capabilities
# GPT-Image-1: Native image-to-image generation and inpainting
# FLUX.1-Kontext-pro: Native image-to-image generation with context
# GPT-4o: Vision analysis only (text output), no direct image editing
AZURE_I2I_SUPPORTED_MODELS = [AZURE_GPT_IMAGE_1, AZURE_FLUX_KONTEXT_PRO]

# Vision Language Models (VLM) - models that analyze images and generate text responses
AZURE_VLM_VERSIONS = {AZURE_GPT_4O: AZURE_SUPPORTED_VERSIONS[AZURE_GPT_4O]}

# Image Generation (IG) models - models that create/edit actual image content
AZURE_IG_VERSIONS = {
    AZURE_GPT_IMAGE_1: AZURE_SUPPORTED_VERSIONS[AZURE_GPT_IMAGE_1],
    AZURE_FLUX_KONTEXT_PRO: AZURE_SUPPORTED_VERSIONS[AZURE_FLUX_KONTEXT_PRO],
}

DEFAULT_AZURE_VERSION = AZURE_GPT_4O


def _validate_image_generation_params(
    width: int, height: int, number_of_images: int, temperature: Optional[float] = None
) -> None:
    """
    Validate parameters for Azure OpenAI image generation.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        number_of_images: Number of images to generate
        temperature: Optional temperature for chat completions

    Raises:
        ValueError: If any parameter is outside valid ranges
    """
    if not (MIN_IMAGE_DIMENSION <= width <= MAX_IMAGE_DIMENSION):
        raise ValueError(
            f"Width must be between {MIN_IMAGE_DIMENSION} and {MAX_IMAGE_DIMENSION} pixels, got {width}"
        )

    if not (MIN_IMAGE_DIMENSION <= height <= MAX_IMAGE_DIMENSION):
        raise ValueError(
            f"Height must be between {MIN_IMAGE_DIMENSION} and {MAX_IMAGE_DIMENSION} pixels, got {height}"
        )

    if not (MIN_IMAGE_COUNT <= number_of_images <= MAX_IMAGE_COUNT):
        raise ValueError(
            f"Number of images must be between {MIN_IMAGE_COUNT} and {MAX_IMAGE_COUNT}, got {number_of_images}"
        )

    if temperature is not None and not (
        MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE
    ):
        raise ValueError(
            f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {temperature}"
        )


def _extract_token_counts(usage: Any) -> tuple[int, int]:
    """Extract input and output token counts from usage object."""
    input_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
    output_tokens = getattr(
        usage, "completion_tokens", getattr(usage, "output_tokens", 0)
    )

    return input_tokens, output_tokens


def _log_ignored_parameters(
    guidance_scale: Optional[float] = None,
    image_prompt_strength: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> None:
    """Log parameters that are not supported by Azure API but are ignored gracefully."""
    if guidance_scale is not None:
        log.debug(
            f"guidance_scale parameter ({guidance_scale}) not supported by Azure API, ignoring"
        )
    if image_prompt_strength is not None:
        log.debug(
            f"image_prompt_strength parameter ({image_prompt_strength}) not supported by Azure API, ignoring"
        )
    if aspect_ratio is not None:
        log.debug(
            f"aspect_ratio parameter ({aspect_ratio}) not supported by Azure API, ignoring"
        )
    if seed is not None:
        log.debug(f"seed parameter ({seed}) not supported by Azure API, ignoring")
    if temperature is not None:
        log.debug(
            f"temperature parameter ({temperature}) not supported by Azure API, ignoring"
        )
    if kwargs:
        log.debug(f"Additional kwargs not used: {list(kwargs.keys())}")


def _calculate_image_generation_cost(
    config: AzureImageConfig,
    input_tokens: int,
    output_tokens: int,
    has_input_images: bool,
) -> float:
    """Calculate cost for image generation models (GPT-Image-1)."""
    # Calculate text input cost
    input_text_cost = input_tokens * config.cost_mm_input_text / MILLION

    # Calculate image input cost if applicable
    input_image_cost = 0
    estimated_image_tokens = 0  # Initialize for logging

    if has_input_images:
        estimated_image_tokens = min(
            input_tokens * IMAGE_TOKEN_ESTIMATION_RATIO, MAX_ESTIMATED_IMAGE_TOKENS
        )
        input_image_cost = estimated_image_tokens * config.cost_mm_input_image / MILLION
        input_text_cost = (
            (input_tokens - estimated_image_tokens)
            * config.cost_mm_input_text
            / MILLION
        )

    # Calculate output cost (for generated images)
    output_image_cost = output_tokens * config.cost_mm_output_image / MILLION

    total_cost = input_text_cost + input_image_cost + output_image_cost

    # Log cost breakdown with proper variable scoping
    text_tokens_used = input_tokens - estimated_image_tokens
    log.debug(
        f"Azure GPT-Image-1 cost breakdown: "
        f"input_text_tokens={text_tokens_used:.0f} "
        f"input_image_tokens={estimated_image_tokens:.0f} "
        f"output_tokens={output_tokens} "
        f"total_cost=${total_cost:.6f}"
    )

    return total_cost


def _calculate_chat_completion_cost(
    config: AzureConfig, input_tokens: int, output_tokens: int
) -> float:
    """Calculate cost for chat completion models (GPT-4o)."""
    cost = (input_tokens * config.cost_mm_input / MILLION) + (
        output_tokens * config.cost_mm_output / MILLION
    )
    log.debug(
        f"Azure cost calculation: input_tokens={input_tokens} output_tokens={output_tokens} cost={cost}"
    )
    return cost


def get_cost(
    config: Union[AzureConfig, AzureImageConfig, AzureFluxKontextConfig],
    response: Optional[Any] = None,
    num_images: int = 1,
    has_input_images: bool = False,
) -> float:
    """Calculate cost based on response usage or image generation."""
    # For FLUX.1-Kontext-pro - per-image pricing
    if isinstance(config, AzureFluxKontextConfig):
        cost = config.cost_per_img * num_images
        log.debug(f"Azure FLUX.1-Kontext-pro cost: num_images={num_images} cost={cost}")
        return cost

    # For image generation (GPT-Image-1) - token-based pricing
    if isinstance(config, AzureImageConfig):
        if response and hasattr(response, "usage"):
            input_tokens, output_tokens = _extract_token_counts(response.usage)
            return _calculate_image_generation_cost(
                config, input_tokens, output_tokens, has_input_images
            )
        else:
            # Fallback: estimate cost for image generation without usage data
            estimated_cost = FALLBACK_IMAGE_COST_ESTIMATE * num_images
            log.debug(
                f"Azure GPT-Image-1 estimated cost: num_images={num_images} cost={estimated_cost}"
            )
            return estimated_cost

    # For chat completions (GPT-4o)
    if isinstance(config, AzureConfig) and response and hasattr(response, "usage"):
        input_tokens, output_tokens = _extract_token_counts(response.usage)
        return _calculate_chat_completion_cost(config, input_tokens, output_tokens)

    return 0


def fix_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix JSON schema for Azure OpenAI compatibility.

    This function modifies object schemas to set additionalProperties=False,
    which is required for Azure OpenAI's strict JSON schema validation.

    Uses an efficient iterative approach with circular reference protection
    to handle deeply nested schemas without performance issues or stack overflow.

    Args:
        schema: The JSON schema dictionary to fix

    Returns:
        A modified copy of the schema with Azure OpenAI compatibility fixes applied

    Raises:
        ValueError: If the schema is too large (>1MB) or contains circular references
    """
    # Use JSON serialization for efficient copying when possible
    # This is much faster than copy.deepcopy() for large nested structures
    try:
        schema_json = json.dumps(schema)
        # Validate schema size to prevent memory issues
        schema_size = len(schema_json.encode("utf-8"))
        if schema_size > MAX_SCHEMA_SIZE_BYTES:
            raise ValueError(
                f"Schema too large ({schema_size} bytes). Azure OpenAI supports schemas up to {MAX_SCHEMA_SIZE_BYTES // (1024*1024)}MB."
            )
        fixed_schema = json.loads(schema_json)
    except (TypeError, ValueError) as e:
        # Fallback to deepcopy for schemas with non-serializable objects
        log.debug(f"Using deepcopy fallback for schema copying: {e}")
        fixed_schema = copy.deepcopy(schema)

    # Use deque for better performance than list.pop() on large collections
    # Track visited objects by id() to prevent infinite loops from circular references
    queue = deque([fixed_schema])
    visited = set()

    while queue:
        current = queue.popleft()

        # Prevent circular reference infinite loops
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if isinstance(current, dict):
            # Apply Azure OpenAI compatibility fix: set additionalProperties=False for objects
            if current.get("type") == "object":
                current["additionalProperties"] = False

            # Queue all nested dictionaries and lists for processing
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)

        elif isinstance(current, list):
            # Queue all list items that are dictionaries or nested lists
            for item in current:
                if isinstance(item, (dict, list)):
                    queue.append(item)

    return fixed_schema


class AzureProvider(BaseImageEditingMixin):
    def __init__(
        self,
        version=DEFAULT_AZURE_VERSION,
        max_retries=7,
        config_overrides=None,
        api_key=None,
        azure_endpoint=None,
        api_version=None,
    ):
        """
        Initialize Azure OpenAI provider with configurable parameters.

        Args:
            version: Azure OpenAI model version (e.g., "azure/gpt-4o", "azure/gpt-image-1", "azure/flux-kontext-pro")
            max_retries: Maximum retry attempts for failed requests (default: 7)
            config_overrides: Dictionary of configuration overrides for model settings
            api_key: Optional API key override (uses secret management system if not provided)
            azure_endpoint: Optional endpoint URL override (uses secret management system if not provided)
            api_version: Optional API version override (default: latest stable version)

        Example:
            # Use default configuration
            provider = AzureProvider(version="azure/gpt-4o")

            # FLUX.1-Kontext-pro for image generation
            provider = AzureProvider(version="azure/flux-kontext-pro")

            # Override API version for testing
            provider = AzureProvider(
                version="azure/gpt-4o",
                api_version="2024-12-01-preview"
            )
        """
        self.version = version
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version

        # Create configuration overrides while preserving caller's original dictionary
        overrides = config_overrides.copy() if config_overrides else {}
        # Set default max_retries only if not explicitly configured
        overrides.setdefault("max_retries", max_retries)
        # Set api_version if provided as parameter
        if api_version is not None:
            overrides["api_version"] = api_version

        self.config = self._configure_provider(overrides)
        self.client = self._initialize_client()

    def _configure_provider(self, config_overrides):
        """
        Apply configuration overrides to the base model configuration.

        Validates the model version and configuration parameters to ensure
        compatibility with Azure OpenAI API requirements.

        Args:
            config_overrides: Dictionary of configuration values to override

        Returns:
            Configured model instance with applied overrides

        Raises:
            ValueError: If the model version is not supported or configuration is invalid
        """
        if self.version not in AZURE_SUPPORTED_VERSIONS:
            supported_versions = list(AZURE_SUPPORTED_VERSIONS.keys())
            raise ValueError(
                f"Unsupported Azure OpenAI version: {self.version}. "
                f"Supported versions: {supported_versions}"
            )

        base_config = AZURE_SUPPORTED_VERSIONS[self.version]

        # Data-driven parameter filtering based on model's supported parameters
        filtered_overrides = {}
        ignored_params = []

        if config_overrides:
            supported_params = base_config.get_supported_params()

            for param, value in config_overrides.items():
                if param in supported_params:
                    filtered_overrides[param] = value
                else:
                    ignored_params.append((param, value))

            # Log ignored parameters with model-specific context
            for param, value in ignored_params:
                log.debug(
                    f"{param} parameter ({value}) not supported by {self.version}, ignoring"
                )

        config = apply_overrides(base_config, filtered_overrides)

        # Apply instance-level endpoint override if provided
        if self._azure_endpoint:
            config.azure_endpoint = self._azure_endpoint

        # Validate critical configuration fields with descriptive error messages
        if hasattr(config, "azure_endpoint") and config.azure_endpoint:
            if not config.azure_endpoint.startswith(("https://", "http://")):
                raise ValueError(
                    f"Invalid Azure endpoint URL: {config.azure_endpoint}. "
                    f"URL must start with 'https://' or 'http://'"
                )

        if hasattr(config, "deployment_name") and not config.deployment_name:
            raise ValueError(
                f"deployment_name is required for {self.version}. "
                f"Please specify the Azure deployment name in your configuration."
            )

        return config

    def _initialize_client(self):
        """Initialize and configure the appropriate Azure OpenAI client.

        Returns:
            Configured AzureOpenAI client instance for the model type
        """
        # Initialize appropriate client based on model type
        if self.version == AZURE_GPT_IMAGE_1:
            return self._initialize_image_client()
        elif self.version == AZURE_FLUX_KONTEXT_PRO:
            return self._initialize_flux_client()
        else:
            return self._initialize_chat_client()

    def _initialize_client_helper(
        self, api_key_secret: str, endpoint_secret: str, endpoint_config_attr: str
    ):
        """Helper to initialize AzureOpenAI client with specific credentials.

        Args:
            api_key_secret: Name of the secret containing the API key
            endpoint_secret: Name of the secret containing the endpoint URL
            endpoint_config_attr: Name of the config attribute containing the endpoint

        Returns:
            Configured AzureOpenAI client instance

        Raises:
            ValueError: If required API key or endpoint is missing or invalid
        """
        api_key_to_use = self._api_key or get_secret(api_key_secret)
        if not api_key_to_use or len(api_key_to_use.strip()) < MIN_API_KEY_LENGTH:
            service_suffix = " for GPT-Image-1" if "IMAGE" in api_key_secret else ""
            raise ValueError(
                f"{api_key_secret} must be set and valid{service_suffix}. "
                "Please configure it as an environment variable or in your secret management system."
            )

        azure_endpoint = getattr(self.config, endpoint_config_attr, None) or get_secret(
            endpoint_secret
        )
        if not azure_endpoint:
            service_suffix = " for GPT-Image-1" if "IMAGE" in endpoint_secret else ""
            raise ValueError(
                f"{endpoint_secret} must be set{service_suffix}. "
                "Please configure your Azure OpenAI endpoint URL as an environment variable or in your secret management system."
            )

        try:
            return AzureOpenAI(
                api_key=api_key_to_use,
                azure_endpoint=azure_endpoint,
                api_version=self.config.api_version,
                max_retries=self.config.max_retries,
            )
        except Exception as e:
            client_type = "image client" if "IMAGE" in api_key_secret else "client"
            raise ValueError(
                f"Failed to initialize Azure OpenAI {client_type}: {e}"
            ) from e

    def _initialize_chat_client(self):
        """
        Initialize Azure OpenAI client for chat and vision-language models.

        Used for GPT-4o and similar models that process text and images
        to generate text responses.

        Returns:
            Configured AzureOpenAI client for chat completions

        Raises:
            ValueError: If required API key or endpoint is missing or invalid
        """
        return self._initialize_client_helper(
            "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "azure_endpoint"
        )

    def _initialize_image_client(self):
        """
        Initialize Azure OpenAI client for image generation models.

        Used for GPT-Image-1 and similar models that generate or edit images
        based on text prompts and optional input images.

        Returns:
            Configured AzureOpenAI client for image operations

        Raises:
            ValueError: If required API key or endpoint is missing or invalid
        """
        return self._initialize_client_helper(
            "AZURE_IMAGE_API_KEY", "AZURE_IMAGE_ENDPOINT", "azure_image_endpoint"
        )

    def _initialize_flux_client(self):
        """
        Initialize Azure OpenAI client for FLUX.1-Kontext-pro model.

        Used for FLUX.1-Kontext-pro model that generates or edits images
        based on text prompts and optional input images with context understanding.

        Returns:
            Configured AzureOpenAI client for FLUX operations

        Raises:
            ValueError: If required API key or endpoint is missing or invalid
        """
        return self._initialize_client_helper(
            "AZURE_FLUX_API_KEY", "AZURE_FLUX_ENDPOINT", "azure_flux_endpoint"
        )

    def call(
        self,
        query,
        temperature=None,
        seed=None,
        image=None,
        schema=None,
    ):
        if not query:
            raise ValueError("Query cannot be empty")

        # Validate temperature parameter if provided
        if temperature is not None and not (
            MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE
        ):
            raise ValueError(
                f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {temperature}"
            )

        try:
            # Prepare request content and parameters
            content = self._prepare_call_content(query, image)
            common_params = self._prepare_call_params(content, temperature, seed)

            # Execute the completion
            completion = self._execute_chat_completion(common_params, schema)

            if not completion or not completion.choices:
                log.error("No completion choices returned from Azure OpenAI")
                return None

            # Report cost
            self._report_completion_cost(completion)

            # Extract and return result
            return self._extract_completion_result(completion, schema)

        except openai.BadRequestError as e:
            log.error(f"Invalid request to Azure OpenAI: {e}")
            raise ValueError(f"Invalid Azure OpenAI request: {e}") from e
        except Exception as e:
            log.error(f"Unexpected error during Azure OpenAI API call: {e}")
            raise RuntimeError(f"Azure OpenAI API call failed: {e}") from e

    def _prepare_call_content(
        self, query: str, image: Optional[Union[Image.Image, List[Image.Image]]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare the content array for the chat completion request.

        Uses memory-efficient lazy encoding to avoid holding multiple large
        base64 strings in memory simultaneously.

        Args:
            query: Text query for the chat completion
            image: Optional image(s) to include in the request

        Returns:
            List of content dictionaries for the API request

        Raises:
            ValueError: If image encoding fails
        """
        content = [{"type": "text", "text": query}]

        if image is not None:
            images = image if isinstance(image, list) else [image]

            # Process images one at a time to reduce peak memory usage
            # Instead of creating all base64 strings at once
            try:
                for img in images:
                    # Encode each image individually and immediately append to content
                    # This reduces memory pressure compared to batch encoding
                    encoded_image = encode_image(img)
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    }
                    content.append(image_content)
                    # encoded_image goes out of scope here, allowing garbage collection

            except Exception as e:
                log.error(f"Failed to encode image: {e}")
                raise ValueError(f"Image encoding failed: {e}") from e

        return content

    def _prepare_call_params(
        self,
        content: List[Dict[str, Any]],
        temperature: Optional[float],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Prepare the common parameters for the chat completion request.

        Args:
            content: The prepared content array with text and images
            temperature: Optional temperature override
            seed: Optional seed for reproducible results

        Returns:
            Dictionary of parameters for the API call
        """
        params = {
            "model": self.config.deployment_name,
            "messages": [{"role": "user", "content": content}],
        }

        if temperature is not None:
            params["temperature"] = temperature
        else:
            params["temperature"] = self.config.temperature

        if seed is not None:
            params["seed"] = seed

        return params

    def _execute_chat_completion(
        self,
        common_params: Dict[str, Any],
        schema: Optional[Union[Dict[str, Any], Any]],
    ) -> Any:
        """Execute the chat completion with optional structured output.

        Args:
            common_params: Common parameters for the API call
            schema: Optional schema for structured output

        Returns:
            The completion response from Azure OpenAI
        """
        if schema:
            api_response_format = schema
            if isinstance(schema, dict):
                fixed_schema = fix_json_schema(schema)
                api_response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "azure_schema",
                        "strict": True,
                        "schema": fixed_schema,
                    },
                }
            return self.client.beta.chat.completions.parse(
                **common_params, response_format=api_response_format
            )
        else:
            return self.client.chat.completions.create(**common_params)

    def _report_completion_cost(self, completion):
        """Report the cost of the completion."""
        try:
            cost = get_cost(self.config, completion)
            report_cost(cost, name=self.version)
        except Exception as e:
            log.warning(f"Failed to get Azure OpenAI cost from response metadata: {e}")
            # Cost calculation failure should not break the flow

    def _extract_completion_result(self, completion, schema):
        """Extract the result from the completion response."""
        message = completion.choices[0].message

        if schema:
            # Check if we have a parsed result from the beta API
            parsed_obj = getattr(message, "parsed", None)
            if parsed_obj is not None:
                # Try Pydantic v2 first (more recent)
                if hasattr(parsed_obj, "model_dump"):
                    return parsed_obj.model_dump()
                # Fallback to Pydantic v1
                elif hasattr(parsed_obj, "dict"):
                    return parsed_obj.dict()

            # Fallback: parse JSON from message content
            if message.content:
                try:
                    return json.loads(message.content)
                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse structured output JSON: {e}")
                    log.error(f"Raw content: {message.content}")
                    raise
            else:
                log.error("No content available for structured output parsing")
                return None
        else:
            return message.content.strip() if message.content else None

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        number_of_images: int = 1,
        image_prompt: Optional[Union[Image.Image, List[Image.Image]]] = None,
        guidance_scale: Optional[float] = None,
        image_prompt_strength: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """
        Generate an image using Azure OpenAI.

        Supports GPT-Image-1 and FLUX.1-Kontext-pro for image generation. Other models will raise NotImplementedError.
        """
        # Handle aspect_ratio parameter using unified resolution system
        width, height, _ = resolve_aspect_ratio_and_dimensions(
            aspect_ratio, self.version, width, height
        )

        if self.version == AZURE_GPT_IMAGE_1:
            return self._generate_image_with_gpt_image_1(
                prompt=prompt,
                width=width,
                height=height,
                _seed=seed,
                number_of_images=number_of_images,
                image_prompt=image_prompt,
                _guidance_scale=guidance_scale,
                _image_prompt_strength=image_prompt_strength,
                _aspect_ratio=aspect_ratio,
                **kwargs,
            )
        elif self.version == AZURE_FLUX_KONTEXT_PRO:
            return self._generate_image_with_flux_kontext_pro(
                prompt=prompt,
                width=width,
                height=height,
                _seed=seed,
                number_of_images=number_of_images,
                image_prompt=image_prompt,
                _guidance_scale=guidance_scale,
                _image_prompt_strength=image_prompt_strength,
                _aspect_ratio=aspect_ratio,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Image generation is not supported for {self.version}. "
                f"Only {AZURE_GPT_IMAGE_1} and {AZURE_FLUX_KONTEXT_PRO} support image generation."
            )

    def _process_image_generation_response(
        self,
        result: Any,
        number_of_images: int,
        has_input_images: bool,
        model_name: str,
    ) -> Optional[Image.Image]:
        """Process the response from an image generation API call.

        Args:
            result: The API response object
            number_of_images: Number of images requested
            has_input_images: Whether input images were provided
            model_name: Name of the model for logging (e.g., "GPT-Image-1", "FLUX.1-Kontext-pro")

        Returns:
            Generated PIL Image or None if processing failed
        """
        if not result:
            log.error(f"No response received from Azure {model_name} API")
            return None
        if not result.data:
            log.error(f"Empty data in Azure {model_name} response: {result}")
            return None
        if len(result.data) == 0:
            log.error(f"No images in Azure {model_name} response data array")
            return None

        # Report cost for image generation
        try:
            cost = get_cost(
                self.config,
                response=result,
                num_images=number_of_images,
                has_input_images=has_input_images,
            )
            report_cost(cost, name=self.version)
        except Exception as e:
            log.warning(f"Failed to calculate Azure {model_name} cost: {e}")
            # Cost calculation failure should not break image generation flow

        # Get the first image from the response
        image_data = result.data[0]

        # Handle both base64 and URL responses
        if hasattr(image_data, "b64_json") and image_data.b64_json:
            image_bytes = base64.b64decode(image_data.b64_json)
            return Image.open(BytesIO(image_bytes))
        elif hasattr(image_data, "url") and image_data.url:
            # If URL is provided, we'd need to fetch it
            log.warning("URL-based image response not yet implemented")
            return None
        else:
            log.error("No usable image data in response")
            return None

    def _generate_image_internal(
        self,
        prompt: str,
        width: int,
        height: int,
        number_of_images: int,
        image_prompt: Optional[Union[Image.Image, List[Image.Image]]],
        deployment_name: str,
        model_display_name: str,
        i2i_api_call_func,
        _seed: Optional[int] = None,
        _guidance_scale: Optional[float] = None,
        _image_prompt_strength: Optional[float] = None,
        _aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Internal method for image generation shared by all Azure models.

        Args:
            prompt: Text description of the desired image
            width: Image width in pixels
            height: Image height in pixels
            number_of_images: Number of images to generate
            image_prompt: Input image(s) for image-to-image generation
            deployment_name: Azure deployment name for the model
            model_display_name: Display name for logging (e.g., "GPT-Image-1", "FLUX.1-Kontext-pro")
            i2i_api_call_func: Function to call for image-to-image generation
            _seed: Random seed for reproducible results (currently unused by Azure API)
            _guidance_scale: Control adherence to prompt (currently unused by Azure API)
            _image_prompt_strength: Influence of input image (currently unused by Azure API)
            _aspect_ratio: Aspect ratio (currently unused by Azure API)

        Returns:
            Generated PIL Image or None if generation failed

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate input parameters
        _validate_image_generation_params(width, height, number_of_images)

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Log unused parameters for debugging but don't error
        _log_ignored_parameters(
            guidance_scale=_guidance_scale,
            image_prompt_strength=_image_prompt_strength,
            aspect_ratio=_aspect_ratio,
            seed=_seed,
            **kwargs,
        )

        try:
            if image_prompt:
                # Image-to-image generation
                images = (
                    image_prompt if isinstance(image_prompt, list) else [image_prompt]
                )

                # Use direct HTTP API call for image-to-image generation
                result = i2i_api_call_func(
                    images=images,
                    prompt=prompt,
                    number_of_images=number_of_images,
                    width=width,
                    height=height,
                )
            else:
                # Text-to-image generation
                result = self.client.images.generate(
                    model=deployment_name,
                    prompt=prompt,
                    n=number_of_images,
                    size=f"{width}x{height}",
                )

            # Process and return the generated image
            return self._process_image_generation_response(
                result=result,
                number_of_images=number_of_images,
                has_input_images=(image_prompt is not None),
                model_name=model_display_name,
            )

        except Exception as e:
            log.error(
                f"Azure {model_display_name} generation failed, prompt '{prompt}': {e}"
            )
            raise RuntimeError(f"Image generation failed: {e}") from e

    def _generate_image_with_gpt_image_1(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        _seed: Optional[int] = None,  # Currently unused by Azure API
        number_of_images: int = 1,
        image_prompt: Optional[Union[Image.Image, List[Image.Image]]] = None,
        _guidance_scale: Optional[float] = None,  # Currently unused by Azure API
        _image_prompt_strength: Optional[float] = None,  # Currently unused by Azure API
        _aspect_ratio: Optional[str] = None,  # Currently unused by Azure API
        **kwargs,
    ) -> Optional[Image.Image]:
        """Generate an image using Azure GPT-Image-1 dedicated image generation API.

        Args:
            prompt: Text description of the desired image
            width: Image width in pixels (64-2048)
            height: Image height in pixels (64-2048)
            _seed: Random seed for reproducible results (currently unused by Azure API)
            number_of_images: Number of images to generate (1-10)
            image_prompt: Input image(s) for image-to-image generation.
                          Multiple images can be provided as references for generating new images.
            _guidance_scale: Control adherence to prompt (currently unused by Azure API)
            _image_prompt_strength: Influence of input image (currently unused by Azure API)

        Returns:
            Generated PIL Image or None if generation failed

        Raises:
            ValueError: If parameters are outside valid ranges

        Note:
            Some parameters (seed, guidance_scale, image_prompt_strength) are not yet
            supported by the Azure GPT-Image-1 API but are included for interface compatibility.
        """
        return self._generate_image_internal(
            prompt=prompt,
            width=width,
            height=height,
            number_of_images=number_of_images,
            image_prompt=image_prompt,
            deployment_name=GPT_IMAGE_1_DEPLOYMENT_NAME,
            model_display_name="GPT-Image-1",
            i2i_api_call_func=self._call_azure_image_edit_api,
            _seed=_seed,
            _guidance_scale=_guidance_scale,
            _image_prompt_strength=_image_prompt_strength,
            _aspect_ratio=_aspect_ratio,
            **kwargs,
        )

    def _generate_image_with_flux_kontext_pro(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        _seed: Optional[int] = None,  # Currently unused by Azure API
        number_of_images: int = 1,
        image_prompt: Optional[Union[Image.Image, List[Image.Image]]] = None,
        _guidance_scale: Optional[float] = None,  # Currently unused by Azure API
        _image_prompt_strength: Optional[float] = None,  # Currently unused by Azure API
        _aspect_ratio: Optional[str] = None,  # Currently unused by Azure API
        **kwargs,
    ) -> Optional[Image.Image]:
        """Generate an image using Azure FLUX.1-Kontext-pro model.

        Args:
            prompt: Text description of the desired image
            width: Image width in pixels
            height: Image height in pixels
            _seed: Random seed for reproducible results (currently unused by Azure API)
            number_of_images: Number of images to generate (1-10)
            image_prompt: Input image(s) for image-to-image generation.
                          Multiple images can be provided as references for generating new images.
            _guidance_scale: Control adherence to prompt (currently unused by Azure API)
            _image_prompt_strength: Influence of input image (currently unused by Azure API)

        Returns:
            Generated PIL Image or None if generation failed

        Raises:
            ValueError: If parameters are outside valid ranges

        Note:
            Some parameters (seed, guidance_scale, image_prompt_strength) are not yet
            supported by the Azure FLUX.1-Kontext-pro API but are included for interface compatibility.
        """
        return self._generate_image_internal(
            prompt=prompt,
            width=width,
            height=height,
            number_of_images=number_of_images,
            image_prompt=image_prompt,
            deployment_name=FLUX_KONTEXT_PRO_DEPLOYMENT_NAME,
            model_display_name="FLUX.1-Kontext-pro",
            i2i_api_call_func=self._call_azure_flux_image_api,
            _seed=_seed,
            _guidance_scale=_guidance_scale,
            _image_prompt_strength=_image_prompt_strength,
            _aspect_ratio=_aspect_ratio,
            **kwargs,
        )

    def supports_image_editing(self) -> bool:
        """Check if this Azure OpenAI version supports image editing.

        GPT-Image-1: Has native image-to-image and inpainting capabilities
        FLUX.1-Kontext-pro: Has native image-to-image generation with context understanding
        GPT-4o: Does not support image generation or editing in this provider.
        """
        return self.version in AZURE_I2I_SUPPORTED_MODELS

    def _has_native_edit_image(self) -> bool:
        """Indicates whether this provider has native image editing capabilities.

        GPT-Image-1: True - uses native _native_edit_image method
        FLUX.1-Kontext-pro: True - uses native _native_edit_image method
        GPT-4o: False - uses chat completion fallback
        """
        return self.version in [AZURE_GPT_IMAGE_1, AZURE_FLUX_KONTEXT_PRO]

    def _native_edit_image(
        self, prompt: str, image: Image.Image, **kwargs
    ) -> Optional[Image.Image]:
        """
        Native image editing implementation called by BaseImageEditingMixin.

        This method is called by the base class when _has_native_edit_image() returns True.
        """
        # Extract common parameters from kwargs, providing defaults
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        _guidance_scale = kwargs.get("guidance_scale", None)
        _image_prompt_strength = kwargs.get("image_prompt_strength", None)
        _aspect_ratio = kwargs.get("aspect_ratio", None)

        if not self.supports_image_editing():
            raise NotImplementedError(
                f"Image editing is not supported for {self.version}. "
                f"Only {AZURE_GPT_IMAGE_1} and {AZURE_FLUX_KONTEXT_PRO} support image editing."
            )

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if image is None:
            raise ValueError("Input image cannot be None")

        # Call the appropriate image editing implementation based on model
        # Both supported models use the same parameter set for image editing
        common_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "image_prompt": [image],
            "number_of_images": 1,
            "_guidance_scale": _guidance_scale,
            "_image_prompt_strength": _image_prompt_strength,
            "_aspect_ratio": _aspect_ratio,
        }

        if self.version == AZURE_GPT_IMAGE_1:
            return self._generate_image_with_gpt_image_1(**common_params)
        elif self.version == AZURE_FLUX_KONTEXT_PRO:
            return self._generate_image_with_flux_kontext_pro(**common_params)
        else:
            raise NotImplementedError(
                f"Image editing is not supported for {self.version}."
            )

    def _build_azure_endpoint(
        self, deployment_name: str, endpoint_path: str
    ) -> tuple[str, dict[str, str]]:
        """Build Azure API endpoint URL and headers for any deployment.

        Args:
            deployment_name: Azure deployment name
            endpoint_path: API endpoint path (e.g., 'images/edits', 'images/generations')

        Returns:
            tuple: (endpoint_url, headers_dict)
        """
        # Get API credentials from the initialized client
        api_key_to_use = self.client.api_key

        # The client's base_url is an httpx.URL object. The scheme and netloc attributes can be
        # either str or bytes, so we defensively handle both cases.
        scheme = self.client.base_url.scheme
        if isinstance(scheme, bytes):
            scheme = scheme.decode("utf-8")

        netloc = self.client.base_url.netloc
        if isinstance(netloc, bytes):
            netloc = netloc.decode("utf-8")

        base_url = urlunparse(
            (
                scheme or "",
                netloc or "",
                "",
                "",
                "",
                "",
            )
        )

        # Construct the API endpoint
        endpoint = f"{base_url}/openai/deployments/{deployment_name}/{endpoint_path}"

        headers = {"api-key": api_key_to_use}

        return endpoint, headers

    def _prepare_multipart_form_data(
        self,
        images: List[Image.Image],
        prompt: str,
        number_of_images: int,
        width: int,
        height: int,
        deployment_name: str,
        image_field_name: str = "image",
    ) -> tuple[list, dict[str, str], list[BytesIO]]:
        """Prepare multipart form data for Azure image APIs.

        Args:
            images: List of PIL images
            prompt: Text prompt
            number_of_images: Number of images to generate
            width: Image width in pixels
            height: Image height in pixels
            deployment_name: Azure deployment name
            image_field_name: Form field name for images ('image' for FLUX, 'image[]' for GPT-Image-1)

        Returns:
            tuple: (files_data, form_data, image_buffers)
        """
        files_data = []  # Use list to allow multiple entries with same key
        image_buffers = []

        # Add images to multipart form data with proper format
        for i, img in enumerate(images):
            # Convert PIL image directly to bytes without unnecessary base64 round-trip
            img_buffer = BytesIO()
            try:
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                # Store buffer reference for cleanup
                image_buffers.append(img_buffer)

                # Use appropriate field name for each model
                files_data.append(
                    (image_field_name, (f"image_{i}.png", img_buffer, "image/png"))
                )
            except Exception:
                # If image processing fails, close this buffer immediately
                img_buffer.close()
                raise

        # Prepare form data fields
        form_data = {
            "prompt": prompt,
            "model": deployment_name,
            "n": str(number_of_images),
            "size": f"{width}x{height}",
        }

        return files_data, form_data, image_buffers

    def _handle_azure_api_response_errors(self, response: requests.Response) -> None:
        """Handle Azure API response errors with specific status code handling.

        Args:
            response: The HTTP response from Azure API

        Raises:
            ValueError: For 400 and 401 errors
            RuntimeError: For 429, 500+, and other errors
        """
        if response.status_code == 400:
            raise ValueError(
                f"Azure API request error (400): {response.text}. "
                "Check image format, prompt, or API parameters."
            )
        elif response.status_code == 401:
            raise ValueError(
                f"Azure API authentication error (401): {response.text}. "
                "Check your API key configuration."
            )
        elif response.status_code == 429:
            raise RuntimeError(
                f"Azure API rate limit exceeded (429): {response.text}. "
                "Please retry after some time."
            )
        elif response.status_code >= 500:
            raise RuntimeError(
                f"Azure API server error ({response.status_code}): {response.text}. "
                "This is a temporary issue, please retry."
            )
        elif response.status_code != 200:
            raise RuntimeError(
                f"Azure API error ({response.status_code}): {response.text}"
            )

    def _parse_azure_image_response(
        self, response: requests.Response
    ) -> _AzureImageResponse:
        """Parse Azure API response and return structured response object.

        Args:
            response: The HTTP response from Azure API

        Returns:
            _AzureImageResponse: Parsed response object

        Raises:
            RuntimeError: If JSON parsing fails
        """
        try:
            json_response = response.json()
        except ValueError as e:
            raise RuntimeError(
                f"Failed to parse Azure API response as JSON: {e}. "
                f"Response content: {response.text[:500]}"
            ) from e

        # Create response object to mimic OpenAI SDK response structure
        return _AzureImageResponse(json_response)

    def _call_azure_multipart_api(
        self,
        images: List[Image.Image],
        prompt: str,
        number_of_images: int,
        width: int,
        height: int,
        deployment_name: str,
        image_field_name: str,
    ) -> _AzureImageResponse:
        """Call Azure's multipart image API directly using HTTP requests.

        This bypasses the OpenAI SDK which has compatibility issues with Azure's implementation.
        Uses in-memory multipart form data for better performance and proper resource management.
        """
        # Build endpoint and headers
        endpoint, headers = self._build_azure_endpoint(deployment_name, "images/edits")

        # Prepare multipart form data using in-memory BytesIO objects
        files_data, form_data, image_buffers = self._prepare_multipart_form_data(
            images,
            prompt,
            number_of_images,
            width,
            height,
            deployment_name,
            image_field_name,
        )

        try:
            # Make the HTTP request - requests will read from buffers during this call
            try:
                response = requests.post(
                    f"{endpoint}?api-version={self.config.api_version}",
                    headers=headers,
                    data=form_data,
                    files=files_data,
                    timeout=AZURE_IMAGE_EDIT_TIMEOUT,
                )
            except requests.exceptions.RequestException as req_error:
                log.error(f"Request failed with error: {req_error}")
                log.error(f"Error type: {type(req_error)}")
                raise

            # Handle response errors
            self._handle_azure_api_response_errors(response)

            # Parse and return response
            return self._parse_azure_image_response(response)

        finally:
            # Clean up in-memory image buffers
            for buffer in image_buffers:
                try:
                    buffer.close()
                except (OSError, AttributeError):
                    pass

    def _call_azure_image_edit_api(
        self,
        images: List[Image.Image],
        prompt: str,
        number_of_images: int,
        width: int,
        height: int,
    ) -> _AzureImageResponse:
        """Call Azure's GPT-Image-1 edit API directly using HTTP requests."""
        return self._call_azure_multipart_api(
            images,
            prompt,
            number_of_images,
            width,
            height,
            GPT_IMAGE_1_DEPLOYMENT_NAME,
            "image[]",
        )

    def _call_azure_flux_image_api(
        self,
        images: List[Image.Image],
        prompt: str,
        number_of_images: int,
        width: int,
        height: int,
    ) -> _AzureImageResponse:
        """Call Azure's FLUX.1-Kontext-pro image API directly using HTTP requests."""
        return self._call_azure_multipart_api(
            images,
            prompt,
            number_of_images,
            width,
            height,
            FLUX_KONTEXT_PRO_DEPLOYMENT_NAME,
            "image",
        )
