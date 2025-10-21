"""
VLM (Vision-Language Model) and LLM abstractions.

This module provides unified interfaces for interacting with different
vision-language models and language models across multiple providers
(Bedrock, Gemini, OpenAI).

Main classes:
    - Vlm: Vision-language model interface for image-text tasks
    - Llm: Language model interface for text-only tasks
"""
from typing import Any, Dict, Optional, Union

from PIL import Image

from ssa.caching import (
    get_cache_statistics,
    get_global_cache,
    setup_caching,
)
from ssa.providers.bedrock import (
    SUPPORTED_VERSIONS as BEDROCK_SUPPORTED_VERSIONS,
)
from ssa.providers.bedrock import (
    BedrockProvider,
)
from ssa.providers.gemini import (
    GEMINI_2_5_FLASH,
    GeminiProvider,
)
from ssa.providers.gemini import (
    SUPPORTED_VERSIONS as GEMINI_SUPPORTED_VERSIONS,
)
from ssa.providers.openai import OPENAI_SUPPORTED_VERSIONS, OpenAIProvider
from ssa.utils.base import flatten_cfg
from ssa.utils.logging import get_log

VLM_VERSIONS = {}
VLM_VERSIONS.update({k: BedrockProvider for k in BEDROCK_SUPPORTED_VERSIONS})
VLM_VERSIONS.update({k: GeminiProvider for k in GEMINI_SUPPORTED_VERSIONS})
VLM_VERSIONS.update({k: OpenAIProvider for k in OPENAI_SUPPORTED_VERSIONS})

DEFAULT_VLM_VERSION = GEMINI_2_5_FLASH
DEFAULT_LLM_VERSION = GEMINI_2_5_FLASH

LLM_VERSIONS = VLM_VERSIONS | {}


log = get_log(__file__)


def _format_unsupported_version_error(
    model_type: str, key: str, available_versions: dict
) -> str:
    """
    Format a standardized error message for unsupported model versions.

    Args:
        model_type: Type of model (e.g., "LLM", "VLM")
        key: The unsupported version key provided
        available_versions: Dictionary of available versions

    Returns:
        Formatted error message string
    """
    versions_list = ", ".join(f"'{v}'" for v in sorted(available_versions.keys()))
    return f"Unsupported {model_type} version: '{key}'. Available versions: {versions_list}"


def _cached_model_call(
    model_instance: Union["Llm", "Vlm"],
    model_type: str,
    query: str,
    image: Optional[Union[Image.Image, str]] = None,
    schema: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    use_cache: Optional[bool] = False,
    scorer_name: Optional[str] = None,
    step_description: Optional[str] = None,
) -> str:
    """
    Shared cached call implementation for both Llm and Vlm classes.

    Args:
        model_instance: The model instance (Llm or Vlm)
        model_type: Type identifier ("llm" or "vlm")
        query: The query to send to the model
        image: Optional image data (for VLMs)
        schema: Optional schema for structured output
        seed: Optional random seed
        temperature: Optional temperature parameter
        use_cache: Whether to use caching
        scorer_name: Optional scorer name for tracing
        step_description: Optional step description for tracing

    Returns:
        The model response (from cache or fresh call)
    """
    # Determine caching behavior
    if use_cache or (use_cache is None):
        cache = get_global_cache()
    if use_cache is None:
        # Use instance setting if not specified, but respect global cache availability
        use_cache = model_instance.enable_caching and (cache is not None)

    # If caching is disabled, make direct call
    if not use_cache:
        response = model_instance.model.call(
            query, image=image, schema=schema, seed=seed, temperature=temperature
        )
        model_instance._record_trace(
            model_type, query, response, scorer_name, step_description
        )
        return response

    # Try to get from cache first (only if caching is enabled)
    if cache is not None and use_cache:
        try:
            cached_response = cache.get(
                query=query,
                model_key=model_instance.key,
                image=image,
                schema=schema,
                seed=seed,
                temperature=temperature,
            )
            if cached_response is not None:
                # Only log cache hits at debug level with truncated query for readability
                query_preview = (
                    str(query)[:50] + "..." if len(str(query)) > 50 else str(query)
                )
                log.debug(f"Cache HIT: {model_instance.key} | {query_preview}")
                model_instance._record_trace(
                    model_type, query, cached_response, scorer_name, step_description
                )
                return cached_response
        except (ValueError, TypeError, UnicodeError, KeyError, AttributeError) as e:
            log.warning(
                f"Cache lookup failed for {model_instance.key}: {e}. Proceeding without cache."
            )

    # Make the actual call
    response = model_instance.model.call(
        query, image=image, schema=schema, seed=seed, temperature=temperature
    )

    # Cache the response (only if caching is enabled)
    if cache is not None and use_cache:
        try:
            cache.put(
                query=query,
                model_key=model_instance.key,
                response=response,
                image=image,
                schema=schema,
                seed=seed,
                temperature=temperature,
                metadata={"instance_class": model_instance.__class__.__name__},
            )
        except (ValueError, TypeError, UnicodeError, KeyError, AttributeError) as e:
            log.warning(
                f"Failed to cache response for {model_instance.key}: {e}. Response not cached."
            )

    model_instance._record_trace(
        model_type, query, response, scorer_name, step_description
    )
    return response


class Llm:
    """A Large-Language-Model which can respond to text-only queries"""

    def __init__(
        self,
        key,
        config_overrides=None,
        warmup=True,
        enable_caching=False,
        trace_collector=None,
    ):
        self.key = key
        if key not in LLM_VERSIONS:
            raise ValueError(
                _format_unsupported_version_error("LLM", key, LLM_VERSIONS)
            )
        self.model = LLM_VERSIONS[key](key, config_overrides=config_overrides)
        self.version = self.model.version
        self.enable_caching = enable_caching
        self.trace_collector = trace_collector
        if warmup:
            self._warmup(wait=False)

    def config(self):
        res = {
            "key": self.key,
            "version": self.model.version,
            "class": self.model.__class__.__name__,
        }
        if hasattr(self.model, "config"):
            res.update({f"config.{k}": flatten_cfg(v) for k, v in self.model.config})
        return res

    def _warmup(self, wait=False):
        if hasattr(self.model, "warmup"):
            log.debug(f"Warm-up: {self.key}")
            self.model.warmup(wait=wait)
        else:
            log.debug(f"No warmup for {self.key}")

    def __str__(self):
        return f"<LLM:{self.key}>"

    def _record_trace(self, model_type, query, response, scorer_name, step_description):
        """Record a trace for this model call."""
        if self.trace_collector:
            self.trace_collector.record_call(
                model_type=model_type,
                model_key=self.key,
                query=query,
                response=response,
                scorer_name=scorer_name,
                step_description=step_description,
            )

    def call(
        self,
        query,
        schema=None,
        seed=None,
        temperature=None,
        use_cache=False,
        scorer_name=None,
        step_description=None,
    ):
        """
        Call the LLM with optional caching and tracing.

        Args:
            query: The query to send to the model
            schema: Optional schema for structured output
            seed: Optional random seed
            temperature: Optional temperature parameter
            use_cache: Whether to use caching (defaults to instance setting)
            scorer_name: Optional scorer name for tracing
            step_description: Optional step description for tracing

        Returns:
            The model response (from cache or fresh call)
        """
        return _cached_model_call(
            self,
            "llm",
            query,
            None,
            schema,
            seed,
            temperature,
            use_cache,
            scorer_name,
            step_description,
        )

    def get_cache_stats(self):
        """Get cache statistics."""
        return get_cache_statistics()

    def setup_cache(self, capacity=None, ttl=None):
        """Setup or reconfigure the cache."""
        return setup_caching(capacity=capacity, ttl=ttl)


class Vlm(Llm):
    """A Vision-Language-Model which can respond to text/image queries"""

    def __init__(
        self,
        key,
        config_overrides=None,
        warmup=True,
        enable_caching=False,
        trace_collector=None,
    ):
        self.key = key
        if key not in VLM_VERSIONS:
            raise ValueError(
                _format_unsupported_version_error("VLM", key, VLM_VERSIONS)
            )
        self.model = VLM_VERSIONS[key](key, config_overrides=config_overrides)
        self.version = self.model.version
        self.enable_caching = enable_caching
        self.trace_collector = trace_collector
        if warmup:
            self._warmup(wait=False)

    def __str__(self):
        return f"<VLM:{self.key}>"

    def call(
        self,
        query,
        image=None,
        schema=None,
        seed=None,
        temperature=None,
        use_cache=False,
        scorer_name=None,
        step_description=None,
    ):
        """
        Call the VLM with optional caching and tracing.

        Args:
            query: The query to send to the model
            image: Optional image data
            schema: Optional schema for structured output
            seed: Optional random seed
            temperature: Optional temperature parameter
            use_cache: Whether to use caching (defaults to instance setting)
            scorer_name: Optional scorer name for tracing
            step_description: Optional step description for tracing

        Returns:
            The model response (from cache or fresh call)
        """
        return _cached_model_call(
            self,
            "vlm",
            query,
            image,
            schema,
            seed,
            temperature,
            use_cache,
            scorer_name,
            step_description,
        )
