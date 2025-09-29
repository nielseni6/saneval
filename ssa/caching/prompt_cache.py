"""
Prompt Caching System for LLM/VLM responses.

This module provides a simple and efficient caching system for LLM and VLM responses
that integrates seamlessly with the existing provider architecture.
"""

import enum
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def issubclass_safe(obj: Any, class_or_tuple) -> bool:
    """Safe version of issubclass that doesn't raise TypeError."""
    try:
        return isinstance(obj, type) and issubclass(obj, class_or_tuple)
    except TypeError:
        return False


def _is_json_serializable(obj: Any) -> bool:
    """
    Efficiently check if an object is JSON serializable without actual serialization.

    This is more efficient than calling json.dumps() for every value.
    """
    # Check basic JSON-serializable types first (most common case)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return True

    # Handle enum types - they are serializable via their value
    if isinstance(obj, enum.Enum):
        return _is_json_serializable(obj.value)

    # Handle enum classes - they are serializable as class representations
    if issubclass_safe(obj, enum.Enum):
        return True

    # Check collections of basic types
    if isinstance(obj, (list, tuple)):
        # For performance, only check first few elements for large collections
        sample_size = min(5, len(obj))
        return all(_is_json_serializable(item) for item in obj[:sample_size])

    if isinstance(obj, dict):
        # Check if keys are strings and values are serializable (sample for large dicts)
        items = list(obj.items())
        sample_size = min(5, len(items))
        for k, v in items[:sample_size]:
            if not isinstance(k, str):
                return False
            # Don't consider dicts with enums as "simply serializable" - they need processing
            if isinstance(v, enum.Enum) or (
                isinstance(v, type) and issubclass_safe(v, enum.Enum)
            ):
                return False
            if not _is_json_serializable(v):
                return False
        return True

    # For other types, they're likely not JSON serializable
    return False


@dataclass
class CachedResponse:
    """
    Represents a cached LLM/VLM response with metadata.
    """

    response: Any
    timestamp: float
    model_key: str
    query_hash: str
    schema: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    has_image: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def is_expired(self, ttl: float) -> bool:
        """Check if the cached response has expired."""
        return time.time() - self.timestamp > ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

        def serialize_value(value):
            """Helper to serialize potentially non-JSON-serializable values."""
            if _is_json_serializable(value):
                return value
            elif hasattr(value, "model_dump") and callable(
                getattr(value, "model_dump")
            ):  # Pydantic v2
                try:
                    return value.model_dump()
                except Exception:
                    pass
            elif hasattr(value, "dict") and callable(
                getattr(value, "dict")
            ):  # Pydantic v1
                try:
                    return value.dict()
                except Exception:
                    pass
            # For all other non-serializable objects, fallback to string representation
            # This includes custom objects that don't have model_dump() or dict() methods
            return str(value)

        return {
            "response": serialize_value(self.response),
            "timestamp": self.timestamp,
            "model_key": self.model_key,
            "query_hash": self.query_hash,
            "schema": serialize_value(self.schema),
            "seed": self.seed,
            "temperature": self.temperature,
            "has_image": self.has_image,
            "metadata": serialize_value(self.metadata),
        }


class PromptCache:
    """
    Thread-safe LRU cache for prompt responses with TTL support.

    This cache is designed to store LLM/VLM responses with automatic expiration
    and least-recently-used eviction when capacity is reached.
    """

    def __init__(self, capacity: int = 1000, ttl: float = 3600):
        """
        Initialize the prompt cache.

        Args:
            capacity (int): Maximum number of cached responses (default: 1000)
            ttl (float): Time to live in seconds (default: 3600 = 1 hour)
        """
        if capacity < 1:
            raise ValueError("Cache capacity must be at least 1")
        if ttl <= 0:
            raise ValueError("TTL must be greater than 0")

        self.capacity = capacity
        self.ttl = ttl
        self.size = 0
        self.hit_count = 0
        self.miss_count = 0

        # OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()

        # Thread safety
        self._lock = threading.Lock()

    def _serialize_object_deterministically(
        self, obj: Any, visited: Optional[set] = None
    ) -> Any:
        """
        Serialize an object deterministically for cache key generation.

        This ensures that identical objects produce identical cache keys,
        regardless of memory addresses or other non-deterministic factors.

        Args:
            obj: The object to serialize
            visited: Set of object IDs to detect circular references

        Returns:
            A deterministic representation suitable for JSON serialization
        """
        # Initialize visited set for circular reference detection
        if visited is None:
            visited = set()

        # Check for circular references
        obj_id = id(obj)
        if obj_id in visited:
            # Return a placeholder for circular references
            return {
                "__circular_ref__": True,
                "__class__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, "__module__", "unknown"),
            }

        # Handle None
        if obj is None:
            return None

        # Handle enum types - convert to their values (must be before _is_json_serializable check)
        if isinstance(obj, enum.Enum):
            return obj.value

        # Handle enum classes (EnumType) - convert to class name for deterministic representation
        if issubclass_safe(obj, enum.Enum):
            return {
                "__enum_class__": True,
                "__name__": obj.__name__,
                "__module__": getattr(obj, "__module__", "unknown"),
                "__members__": [member.value for member in obj],
            }

        # Handle Pydantic model classes - use model_json_schema() for clean serialization
        try:
            # Check if it's a Pydantic model class (not instance)
            if (
                isinstance(obj, type)
                and hasattr(obj, "model_json_schema")
                and callable(getattr(obj, "model_json_schema"))
            ):
                return {
                    "__pydantic_model_class__": True,
                    "__name__": obj.__name__,
                    "__module__": getattr(obj, "__module__", "unknown"),
                    "__schema__": obj.model_json_schema(),
                }
        except Exception:
            # If model_json_schema() fails, fall through to other handlers
            pass

        # Handle basic JSON-serializable types
        if _is_json_serializable(obj):
            return obj

        # Add to visited set for objects we're about to traverse
        visited.add(obj_id)

        try:
            # Handle Pydantic models (v2)
            if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
                try:
                    # Use mode='json' to ensure proper enum serialization
                    result = obj.model_dump(mode="json")
                    visited.remove(obj_id)
                    return result
                except Exception:
                    # Fallback to default model_dump if mode='json' fails
                    try:
                        result = obj.model_dump()
                        visited.remove(obj_id)
                        return result
                    except Exception:
                        pass

            # Handle Pydantic models (v1)
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                try:
                    result = obj.dict()
                    visited.remove(obj_id)
                    return result
                except Exception:
                    pass

            # Handle objects with __dict__ (most custom classes)
            if hasattr(obj, "__dict__"):
                try:
                    result = {}
                    for k, v in sorted(obj.__dict__.items()):  # Sort for determinism
                        # Recursively serialize nested objects with visited tracking
                        result[k] = self._serialize_object_deterministically(v, visited)
                    return {
                        "__class__": obj.__class__.__name__,
                        "__module__": getattr(obj.__class__, "__module__", "unknown"),
                        "__dict__": result,
                    }
                except Exception:
                    pass

            # Handle iterables (lists, tuples, sets)
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                try:
                    if isinstance(obj, dict):
                        # Handle dictionaries with potentially non-serializable values
                        result = {}
                        for k, v in sorted(obj.items()):
                            serialized_key = str(k) if not isinstance(k, str) else k
                            result[serialized_key] = (
                                self._serialize_object_deterministically(v, visited)
                            )
                        return result
                    elif isinstance(obj, set):
                        # Convert sets to sorted lists for determinism
                        items = [
                            self._serialize_object_deterministically(item, visited)
                            for item in obj
                        ]
                        return sorted(
                            items, key=str
                        )  # Sort by string representation for determinism
                    else:
                        # Handle lists, tuples, and other iterables
                        return [
                            self._serialize_object_deterministically(item, visited)
                            for item in obj
                        ]
                except Exception:
                    pass

            # Handle callables by their name and module
            if callable(obj):
                try:
                    return {
                        "__callable__": True,
                        "__name__": getattr(obj, "__name__", "unknown"),
                        "__module__": getattr(obj, "__module__", "unknown"),
                        "__qualname__": getattr(obj, "__qualname__", "unknown"),
                    }
                except Exception:
                    pass

            # Final fallback: use class info instead of str() to avoid memory addresses
            return {
                "__class__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, "__module__", "unknown"),
                "__repr_fallback__": True,
            }

        finally:
            # Always remove from visited set when done with this object
            visited.discard(obj_id)

    def _generate_cache_key(
        self, query: Union[str, List], model_key: str, **kwargs
    ) -> str:
        """
        Generate a unique cache key for the given parameters.

        Args:
            query: The query (string or list of messages)
            model_key: The model identifier
            **kwargs: Additional parameters (image, schema, seed, temperature, etc.)

        Returns:
            str: A unique hash key

        Raises:
            ValueError: If required parameters are invalid
        """
        if not query:
            raise ValueError("Query cannot be empty")
        if not model_key:
            raise ValueError("Model key cannot be empty")

        try:
            # Convert query to string if it's a list/dict
            if isinstance(query, (list, dict)):
                query_str = json.dumps(query, sort_keys=True, separators=(",", ":"))
            else:
                query_str = str(query).strip()

            # Validate query after conversion
            if not query_str:
                raise ValueError("Query cannot be empty after conversion")

            # Create base parameters dict
            params = {
                "query": query_str,
                "model_key": str(model_key).strip(),
            }

            # Handle image parameter specially for hashing
            image = kwargs.get("image")
            if image is not None:
                try:
                    # Create a simple hash of the image
                    if hasattr(image, "tobytes"):
                        # For numpy arrays or PIL images
                        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
                    elif isinstance(image, bytes):
                        image_hash = hashlib.md5(image).hexdigest()[:16]
                    elif isinstance(image, str):
                        # For string representations (file paths, base64, etc.)
                        image_hash = hashlib.md5(image.encode("utf-8")).hexdigest()[:16]
                    else:
                        # For other types, convert to string
                        image_hash = hashlib.md5(
                            str(image).encode("utf-8")
                        ).hexdigest()[:16]
                    params["image_hash"] = image_hash
                except Exception as e:
                    logger.warning(f"Failed to hash image data: {e}. Using fallback.")
                    # Fallback: use type and str representation
                    fallback_str = f"{type(image).__name__}:{str(image)[:100]}"
                    params["image_hash"] = hashlib.md5(
                        fallback_str.encode("utf-8")
                    ).hexdigest()[:16]

                # Mark that image was present
                params["has_image"] = True
            else:
                params["has_image"] = False

            # Add all other kwargs to params, excluding image since we handled it specially
            for key, value in kwargs.items():
                if key != "image":  # Skip image since we handled it above
                    # Handle non-serializable objects (like Pydantic models)
                    if _is_json_serializable(value):
                        params[key] = value
                    else:
                        # For non-serializable objects, create a deterministic representation
                        params[key] = self._serialize_object_deterministically(value)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Create deterministic string
            params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))

            # Generate hash
            return hashlib.sha256(params_str.encode("utf-8")).hexdigest()

        except (TypeError, UnicodeError) as e:
            logger.error(f"Failed to generate cache key: {e}")
            raise ValueError(f"Cannot generate cache key: {e}") from e

    def get(self, query: Union[str, List], model_key: str, **kwargs) -> Optional[Any]:
        """
        Get cached response if it exists and hasn't expired.

        Args:
            query: The query (string or list of messages)
            model_key: The model identifier
            **kwargs: Additional parameters for cache key generation
                     (image, schema, seed, temperature, etc.)

        Returns:
            The cached response or None if not found/expired
        """
        cache_key = self._generate_cache_key(query, model_key, **kwargs)

        with self._lock:
            if cache_key in self._cache:
                cached_response = self._cache[cache_key]

                # Check if expired
                if cached_response.is_expired(self.ttl):
                    self._remove_key(cache_key)
                    self.miss_count += 1
                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                self.hit_count += 1

                # Cache hit logging removed - now handled by caller (VLM/IG layers)
                # to reduce log noise and avoid duplicate messages

                return cached_response.response
            else:
                self.miss_count += 1
                return None

    def put(
        self,
        query: Union[str, List],
        model_key: str,
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Store a response in the cache.

        Args:
            query: The query (string or list of messages)
            model_key: The model identifier
            response: The response to cache
            metadata: Optional additional metadata
            **kwargs: Additional parameters for cache key generation
                     (image, schema, seed, temperature, etc.)
        """
        cache_key = self._generate_cache_key(query, model_key, **kwargs)
        current_time = time.time()

        # Create cached response
        cached_response = CachedResponse(
            response=response,
            timestamp=current_time,
            model_key=model_key,
            query_hash=cache_key,
            schema=kwargs.get("schema"),
            seed=kwargs.get("seed"),
            temperature=kwargs.get("temperature"),
            has_image=kwargs.get("image") is not None,
            metadata=metadata,
        )

        with self._lock:
            # Clean expired items first
            self._clean_expired()

            if cache_key in self._cache:
                # Update existing entry
                self._cache[cache_key] = cached_response
                self._cache.move_to_end(cache_key)
            else:
                # Check if we need to evict
                if self.size >= self.capacity:
                    self._evict_lru()

                # Add new entry
                self._cache[cache_key] = cached_response
                self.size += 1

    def invalidate(self, query: Union[str, List], model_key: str, **kwargs) -> bool:
        """
        Invalidate a specific cached entry.

        Args:
            query: The query (string or list of messages)
            model_key: The model identifier
            **kwargs: Additional parameters for cache key generation
                     (image, schema, seed, temperature, etc.)

        Returns:
            bool: True if entry was found and removed
        """
        cache_key = self._generate_cache_key(query, model_key, **kwargs)

        with self._lock:
            if cache_key in self._cache:
                self._remove_key(cache_key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self.size = 0
            self.hit_count = 0
            self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            miss_rate = self.miss_count / total_requests if total_requests > 0 else 0.0
            utilization = self.size / self.capacity if self.capacity > 0 else 0.0

            # Calculate memory efficiency metrics
            avg_entries_per_request = (
                self.size / total_requests if total_requests > 0 else 0.0
            )

            # Time-based metrics
            current_time = time.time()
            expired_count = 0
            oldest_entry_age = 0.0
            newest_entry_age = 0.0

            if self._cache:
                timestamps = [
                    cached_resp.timestamp for cached_resp in self._cache.values()
                ]
                if timestamps:
                    oldest_timestamp = min(timestamps)
                    newest_timestamp = max(timestamps)
                    oldest_entry_age = current_time - oldest_timestamp
                    newest_entry_age = current_time - newest_timestamp

                    # Count expired entries (without cleaning them)
                    expired_count = sum(
                        1 for ts in timestamps if current_time - ts > self.ttl
                    )

            return {
                # Basic metrics
                "capacity": self.capacity,
                "size": self.size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "total_requests": total_requests,
                # Rate metrics
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "utilization": utilization,
                # Efficiency metrics
                "avg_entries_per_request": avg_entries_per_request,
                "expired_entries_count": expired_count,
                "expired_entries_ratio": (
                    expired_count / self.size if self.size > 0 else 0.0
                ),
                # Time metrics
                "ttl": self.ttl,
                "oldest_entry_age_seconds": oldest_entry_age,
                "newest_entry_age_seconds": newest_entry_age,
                # Health indicators
                "is_healthy": hit_rate > 0.1
                and utilization < 0.9,  # Basic health check
                "cache_efficiency_score": hit_rate
                * (1 - expired_count / max(self.size, 1)),
            }

    def resize(self, new_capacity: int) -> None:
        """Resize the cache capacity."""
        if new_capacity < 1:
            raise ValueError("Cache capacity must be at least 1")

        with self._lock:
            self.capacity = new_capacity
            while self.size > self.capacity:
                self._evict_lru()

    def set_ttl(self, new_ttl: float) -> None:
        """Set a new TTL value."""
        if new_ttl <= 0:
            raise ValueError("TTL must be greater than 0")

        with self._lock:
            self.ttl = new_ttl
            self._clean_expired()

    def _clean_expired(self) -> None:
        """Remove expired entries from the cache."""
        expired_keys = []

        for key, cached_response in self._cache.items():
            if cached_response.is_expired(self.ttl):
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_key(key)

    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_key(key)

    def _remove_key(self, key: str) -> None:
        """Remove a key from the cache."""
        if key in self._cache:
            del self._cache[key]
            self.size -= 1

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return self.size

    def __repr__(self) -> str:
        """Return string representation."""
        with self._lock:
            return f"PromptCache(capacity={self.capacity}, size={self.size}, ttl={self.ttl})"


# Global cache instance with thread safety
_global_cache: Optional[PromptCache] = None
_cache_lock = threading.RLock()


def get_global_cache() -> Optional[PromptCache]:
    """
    Get the global prompt cache instance.

    Returns None if caching is disabled via configuration,
    even if a cache instance exists.
    """
    with _cache_lock:
        if _global_cache is None:
            return None

        # Check if caching is disabled via configuration
        try:
            from .config import get_config

            config = get_config()
            if not config.enabled:
                return None
        except Exception:
            # If config check fails, return the cache anyway
            # This prevents breaking functionality due to config issues
            pass

        return _global_cache


def init_global_cache(capacity: int = 1000, ttl: float = 3600) -> PromptCache:
    """
    Initialize the global prompt cache.

    Args:
        capacity (int): Maximum number of cached responses
        ttl (float): Time to live in seconds

    Returns:
        PromptCache: The initialized cache instance

    Raises:
        ValueError: If capacity or ttl are invalid
    """
    if capacity < 1:
        raise ValueError("Cache capacity must be at least 1")
    if ttl <= 0:
        raise ValueError("TTL must be greater than 0")

    global _global_cache
    with _cache_lock:
        _global_cache = PromptCache(capacity=capacity, ttl=ttl)
        # Only log detailed initialization info at debug level to avoid noise
        logger.debug(
            f"Initialized global prompt cache (capacity={capacity}, ttl={ttl})"
        )
        return _global_cache


def clear_global_cache() -> None:
    """Clear the global prompt cache."""
    global _global_cache
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
            logger.info("Cleared global prompt cache")


def _reset_global_cache() -> None:
    """Reset the global prompt cache instance to None. For testing only."""
    global _global_cache
    with _cache_lock:
        _global_cache = None


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    cache = get_global_cache()
    if cache is not None:
        return cache.get_stats()
    return {"error": "No global cache available or caching is disabled"}
