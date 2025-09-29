"""
Configuration for prompt caching.

This module provides configuration options for the prompt caching system.
"""

import os
import threading
from typing import Any, Dict, List, Optional

from ssa.utils.logging import get_log

# Default configuration
DEFAULT_CONFIG = {
    "enabled": False,
    "capacity": 1000,
    "ttl": 3600,  # 1 hour in seconds
    "log_level": "INFO",
    "auto_init": True,
}

# Environment variable mappings
ENV_MAPPINGS = {
    "PROMPT_CACHE_DISABLED": (
        "enabled",
        lambda x: x.lower() != "true",
    ),  # Inverted logic: disabled=true means enabled=false
    "PROMPT_CACHE_ENABLED": (
        "enabled",
        lambda x: x.lower() == "true",
    ),  # Kept for backward compatibility
    "PROMPT_CACHE_CAPACITY": ("capacity", int),
    "PROMPT_CACHE_TTL": ("ttl", float),
    "PROMPT_CACHE_LOG_LEVEL": ("log_level", str),
    "PROMPT_CACHE_AUTO_INIT": ("auto_init", lambda x: x.lower() == "true"),
}

log = get_log(__file__)


class CacheConfig:
    """Configuration class for prompt caching."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cache configuration.

        Args:
            config: Optional configuration dictionary
        """
        self._config = DEFAULT_CONFIG.copy()

        # Update with environment variables
        self._load_from_env()

        # Update with provided config
        if config:
            self._config.update(config)

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Handle enabled/disabled logic with priority: DISABLED > ENABLED
        disabled_value = os.getenv("PROMPT_CACHE_DISABLED", "false")
        enabled_value = os.getenv("PROMPT_CACHE_ENABLED", "false")

        # Enable the prompt cache only if the user sets prompt_cache_enabled in their environmental variable, and prompt_cache_disabled is not already set.
        try:
            is_disabled = disabled_value.lower() == "true"
            is_enabled = enabled_value.lower() == "true"
            self._config["enabled"] = not is_disabled and is_enabled
            # Only log debug level to avoid noise
            if self._config["enabled"]:
                log.debug(
                    f"Cache {self._config['enabled']} via PROMPT_CACHE_DISABLED={disabled_value} and PROMPT_CACHE_ENABLED={enabled_value}"
                )
        except Exception as e:
            log.warning(
                f"Invalid value for PROMPT_CACHE_DISABLED: {disabled_value} or PROMPT_CACHE_ENABLED: {enabled_value} ({e})"
            )

        # Process other environment variables
        for env_var, (key, converter) in ENV_MAPPINGS.items():
            if env_var in ["PROMPT_CACHE_DISABLED", "PROMPT_CACHE_ENABLED"]:
                continue  # Already handled above

            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    # Validate the converted value
                    self._validate_config_value(key, converted_value)
                    self._config[key] = converted_value
                except (ValueError, TypeError) as e:
                    log.warning(f"Warning: Invalid value for {env_var}: {value} ({e})")
                    log.info(
                        f"Using default value for {key}: {DEFAULT_CONFIG.get(key)}"
                    )

    def _validate_config_value(self, key: str, value: Any) -> None:
        """Validate a configuration value."""
        if key == "capacity":
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Capacity must be a positive integer, got: {value}")
            if value > 100000:
                log.warning(
                    f"Very large cache capacity ({value}). This may use significant memory."
                )

        elif key == "ttl":
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"TTL must be a positive number, got: {value}")
            if value > 86400 * 7:  # 7 days
                log.warning(
                    f"Very long TTL ({value} seconds). Consider shorter TTL for fresher data."
                )
            elif value < 60:  # 1 minute
                log.warning(
                    f"Very short TTL ({value} seconds). This may reduce cache effectiveness."
                )

        elif key == "log_level":
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if value.upper() not in valid_levels:
                raise ValueError(
                    f"Log level must be one of {valid_levels}, got: {value}"
                )

        elif key == "enabled":
            if not isinstance(value, bool):
                raise ValueError(f"Enabled must be a boolean, got: {value}")

        elif key == "auto_init":
            if not isinstance(value, bool):
                raise ValueError(f"Auto init must be a boolean, got: {value}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with validation."""
        self._validate_config_value(key, value)
        self._config[key] = value

    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with dictionary, validating all values."""
        for key, value in config.items():
            if key in DEFAULT_CONFIG:  # Only validate known keys
                self._validate_config_value(key, value)
        self._config.update(config)

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get("enabled", False)

    @property
    def capacity(self) -> int:
        """Get cache capacity."""
        return self.get("capacity", 1000)

    @property
    def ttl(self) -> float:
        """Get cache TTL."""
        return self.get("ttl", 3600)

    @property
    def log_level(self) -> str:
        """Get log level."""
        return self.get("log_level", "INFO")

    @property
    def auto_init(self) -> bool:
        """Check if auto-initialization is enabled."""
        return self.get("auto_init", True)


# Global configuration instance with thread safety
_global_config: Optional[CacheConfig] = None
_config_lock = threading.RLock()


def get_config() -> CacheConfig:
    """Get the global configuration instance."""
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = CacheConfig()
        return _global_config


def set_config(config: Dict[str, Any]) -> None:
    """Set the global configuration."""
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = CacheConfig(config)
        else:
            _global_config.update(config)


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    with _config_lock:
        _global_config = CacheConfig()


def _reset_global_config() -> None:
    """Reset the global configuration instance to None. For testing only."""
    global _global_config
    with _config_lock:
        _global_config = None


# Configuration presets
PRESETS = {
    "development": {
        "enabled": True,
        "capacity": 100,
        "ttl": 300,  # 5 minutes
        "log_level": "DEBUG",
    },
    "production": {
        "enabled": True,
        "capacity": 5000,
        "ttl": 7200,  # 2 hours
        "log_level": "INFO",
    },
    "testing": {
        "enabled": True,
        "capacity": 50,
        "ttl": 60,  # 1 minute
        "log_level": "WARNING",
    },
    "disabled": {
        "enabled": False,
        "capacity": 1,
        "ttl": 1,
        "log_level": "ERROR",
    },
}


def load_preset(preset_name: str) -> None:
    """
    Load a configuration preset.

    Args:
        preset_name: Name of the preset to load

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"
        )

    preset_config = PRESETS[preset_name]

    # Validate preset configuration
    temp_config = CacheConfig()
    for key, value in preset_config.items():
        temp_config._validate_config_value(key, value)

    set_config(preset_config)


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary and return any warnings.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        List of warning messages (empty if no issues)

    Raises:
        ValueError: If configuration has critical errors
    """
    warnings = []
    temp_config = CacheConfig()

    for key, value in config_dict.items():
        if key in DEFAULT_CONFIG:
            try:
                temp_config._validate_config_value(key, value)
            except ValueError as e:
                raise ValueError(f"Invalid configuration for {key}: {e}")
        else:
            warnings.append(f"Unknown configuration key: {key}")

    # Cross-validation checks
    capacity = config_dict.get("capacity", DEFAULT_CONFIG["capacity"])
    ttl = config_dict.get("ttl", DEFAULT_CONFIG["ttl"])

    # Check if TTL is reasonable for the capacity
    if capacity > 10000 and ttl < 300:  # Large cache with short TTL
        warnings.append(
            f"Large cache capacity ({capacity}) with short TTL ({ttl}s) may lead to inefficient memory usage"
        )

    # Check if enabled but no capacity
    if config_dict.get("enabled", True) and capacity < 10:
        warnings.append(
            f"Cache is enabled but capacity is very small ({capacity}). Consider increasing capacity or disabling cache."
        )

    return warnings


def print_config() -> None:
    """Print current configuration with validation."""
    config = get_config()
    config_dict = config.to_dict()

    log.info("Prompt Cache Configuration:")
    log.info("=" * 30)
    for key, value in config_dict.items():
        log.info(f"{key}: {value}")
    log.info("=" * 30)

    # Show validation warnings
    try:
        warnings = validate_config(config_dict)
        if warnings:
            log.info("\nConfiguration Warnings:")
            for warning in warnings:
                log.warning(f"  ⚠️  {warning}")
        else:
            log.info("\n✅ Configuration looks good!")
    except ValueError as e:
        log.error(f"\n❌ Configuration Error: {e}")

    log.info("=" * 30)
