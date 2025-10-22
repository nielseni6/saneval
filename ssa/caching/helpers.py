"""
Simple caching utilities for LLM/VLM integration.

This module provides easy-to-use functions for integrating prompt caching
with existing LLM/VLM classes.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .config import get_config
from .prompt_cache import PromptCache, get_global_cache, init_global_cache

logger = logging.getLogger(__name__)


def setup_caching(
    capacity: Optional[int] = None, ttl: Optional[float] = None
) -> PromptCache:
    """
    Setup prompt caching with configurable parameters.

    Args:
        capacity: Maximum number of cached responses (default: from config)
        ttl: Time to live in seconds (default: from config)

    Returns:
        PromptCache: The initialized cache instance
    """
    config = get_config()

    # Use config defaults if not provided
    if capacity is None:
        capacity = config.capacity

    if ttl is None:
        ttl = config.ttl

    return init_global_cache(capacity=capacity, ttl=ttl)


def cached_call(
    model_instance: Any,
    query: Union[str, List],
    image: Any = None,
    schema: Optional[Dict] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    use_cache: bool = True,
) -> Any:
    """
    Make a cached call to any model instance.

    This function can be used with any model instance that has a `call` method.
    It will automatically cache the response based on the input parameters.

    Args:
        model_instance: The model instance to call
        query: The query to send to the model
        image: Optional image data (for VLMs)
        schema: Optional schema for structured output
        seed: Optional random seed
        temperature: Optional temperature parameter
        use_cache: Whether to use caching for this call

    Returns:
        The model response (from cache or fresh call)
    """
    cache = get_global_cache()

    # If no cache or caching disabled, make direct call
    if not use_cache or cache is None:
        return model_instance.call(
            query, image=image, schema=schema, seed=seed, temperature=temperature
        )

    # Get model key for caching
    model_key = getattr(
        model_instance, "key", getattr(model_instance, "version", "unknown")
    )

    # Check cache first
    cached_response = cache.get(
        query=query,
        model_key=model_key,
        image=image,
        schema=schema,
        seed=seed,
        temperature=temperature,
    )

    if cached_response is not None:
        return cached_response

    # Make the actual call
    response = model_instance.call(
        query, image=image, schema=schema, seed=seed, temperature=temperature
    )

    # Cache the response
    cache.put(
        query=query,
        model_key=model_key,
        response=response,
        image=image,
        schema=schema,
        seed=seed,
        temperature=temperature,
        metadata={"instance_class": model_instance.__class__.__name__},
    )

    return response


def invalidate_cache_entry(
    model_key: str,
    query: Union[str, List],
    image: Any = None,
    schema: Optional[Dict] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
) -> bool:
    """
    Invalidate a specific cache entry.

    Args:
        model_key: The model key
        query: The query to invalidate
        image: Optional image data
        schema: Optional schema
        seed: Optional seed
        temperature: Optional temperature

    Returns:
        bool: True if entry was found and invalidated
    """
    cache = get_global_cache()
    if cache is not None:
        return cache.invalidate(
            query,
            model_key,
            image=image,
            schema=schema,
            seed=seed,
            temperature=temperature,
        )
    return False


def clear_cache() -> None:
    """Clear all cached entries."""
    cache = get_global_cache()
    if cache is not None:
        cache.clear()


def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics."""
    cache = get_global_cache()
    if cache is not None:
        return cache.get_stats()
    return {"error": "No global cache available or caching is disabled"}


def destroy_cache() -> bool:
    """
    Destroy the global cache and set all cache-related flags to disabled.

    This function:
    1. Clears the global cache
    2. Sets the global cache to None
    3. Updates configuration to disabled state
    4. Sets environment variables to indicate cache is disabled

    Returns:
        bool: True if cache was successfully destroyed, False if no cache existed
    """
    import os

    from .config import set_config
    from .prompt_cache import _cache_lock, clear_global_cache

    logger.info("Destroying cache due to disable_cache flag")

    cache_existed = False

    # Clear and destroy the global cache
    with _cache_lock:
        # Import the global variable properly
        import ssa.caching.prompt_cache as prompt_cache_module

        if prompt_cache_module._global_cache is not None:
            cache_existed = True
            logger.info(
                f"Clearing cache with {len(prompt_cache_module._global_cache)} entries"
            )
            clear_global_cache()
            # Set global cache to None to completely destroy it
            prompt_cache_module._global_cache = None

    # Update configuration to disabled state
    set_config({"enabled": False})

    # Set environment variable to indicate cache is disabled
    os.environ["PROMPT_CACHE_DISABLED"] = "true"

    if cache_existed:
        logger.info("Cache successfully destroyed and disabled")
    else:
        logger.info("No cache existed to destroy, but cache disabled flags set")

    return cache_existed


def disable_all_cache_flags() -> None:
    """
    Set all cache-related variable flags to disabled state.

    This ensures that even if cache initialization is attempted later,
    it will be disabled based on the configuration and environment variables.
    """
    import os

    from .config import set_config

    # Set environment variables
    os.environ["PROMPT_CACHE_DISABLED"] = "true"
    os.environ["PROMPT_CACHE_ENABLED"] = "false"

    # Update global configuration
    set_config({"enabled": False, "auto_init": False})

    logger.info("All cache flags set to disabled state")


def configure_cache_from_env() -> Optional[PromptCache]:
    """
    Configure cache from configuration system.

    Returns:
        PromptCache instance if enabled, None otherwise
    """
    config = get_config()

    if not config.enabled:
        logger.info("Prompt caching is disabled")
        return None

    logger.info("Prompt caching is enabled")
    return setup_caching(capacity=config.capacity, ttl=config.ttl)


def initialize_cache_with_status_log() -> Optional[PromptCache]:
    """
    Initialize cache and log status at the beginning of the application.

    This should be called once at application startup to show cache status.

    Returns:
        PromptCache instance if enabled, None otherwise
    """
    config = get_config()

    if not config.enabled:
        logger.info("Not using cache (disabled)")
        return None

    cache = setup_caching(capacity=config.capacity, ttl=config.ttl)
    logger.info(f"Using cache (capacity={config.capacity}, ttl={config.ttl}s)")
    return cache
