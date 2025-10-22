"""
Caching package for prompt caching system.

This package provides caching functionality for LLM/VLM responses with features
including configuration, monitoring, and the core caching implementation.
"""

from .config import (PRESETS, CacheConfig, get_config, load_preset,
                     print_config, reset_config, set_config, validate_config)
from .helpers import (cached_call, clear_cache, configure_cache_from_env,
                      destroy_cache, disable_all_cache_flags,
                      get_cache_statistics, initialize_cache_with_status_log,
                      invalidate_cache_entry, setup_caching)
from .monitoring import (CacheAlert, CacheMonitor, check_cache_health,
                         get_cache_report, get_global_monitor,
                         init_cache_monitoring)
from .prompt_cache import (CachedResponse, PromptCache, clear_global_cache,
                           get_cache_stats, get_global_cache,
                           init_global_cache)

__all__ = [
    # Config
    "CacheConfig",
    "get_config",
    "set_config",
    "reset_config",
    "load_preset",
    "validate_config",
    "print_config",
    "PRESETS",
    # Monitoring
    "CacheAlert",
    "CacheMonitor",
    "get_global_monitor",
    "init_cache_monitoring",
    "check_cache_health",
    "get_cache_report",
    # Core Cache
    "CachedResponse",
    "PromptCache",
    "get_global_cache",
    "init_global_cache",
    "clear_global_cache",
    "get_cache_stats",
    # Helpers
    "setup_caching",
    "cached_call",
    "invalidate_cache_entry",
    "clear_cache",
    "get_cache_statistics",
    "destroy_cache",
    "disable_all_cache_flags",
    "configure_cache_from_env",
    "initialize_cache_with_status_log",
]
