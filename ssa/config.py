"""
Central configuration for SSA benchmarking.

This module provides comprehensive configuration management with:
- Centralized default values
- Pydantic-based validation
- Environment variable support
- Type-safe configuration access

Configuration Structure:
    - PathConfig: File and path-related settings
    - ModelConfig: Default model versions and providers
    - SecurityConfig: Security and validation settings
    - BenchmarkConfig: Benchmark execution settings
    - SSAConfig: Main configuration combining all subsystems

Example:
    >>> from ssa.config import get_config
    >>> config = get_config()
    >>> print(config.models.default_vlm)
    'gemini/2.5-flash'

Environment Variables:
    SSA_OUTPUT_BASE: Base output directory for results
    SSA_DEFAULT_VLM: Default vision-language model
    SSA_DEFAULT_LLM: Default language model
    SSA_DEFAULT_OD: Default object detection model
    SSA_MAX_PATH_LENGTH: Maximum allowed path length
    SSA_CIRCULAR_REF_DEPTH: Maximum depth for circular reference removal
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Path and File Configuration
# =============================================================================


class PathConfig(BaseModel):
    """
    Configuration for file paths and formats.

    Attributes:
        supported_image_formats: Set of allowed image file extensions
        default_output_base: Default directory for benchmark outputs
        default_random_chars: Number of random characters for unique IDs
    """

    supported_image_formats: Set[str] = Field(
        default={".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
        description="Supported image file extensions for benchmark evaluation",
    )

    default_output_base: Path = Field(
        default=Path("results/unknown"),
        description="Base directory for debug outputs when output_dir is not specified",
    )

    default_random_chars: int = Field(
        default=12,
        ge=4,
        le=32,
        description="Number of random characters for generating unique IDs",
    )

    @field_validator("default_output_base")
    @classmethod
    def ensure_path(cls, v: Any) -> Path:
        """Ensure the value is a Path object."""
        return Path(v) if not isinstance(v, Path) else v

    @field_validator("supported_image_formats")
    @classmethod
    def validate_extensions(cls, v: Set[str]) -> Set[str]:
        """Ensure all extensions start with a dot."""
        validated = set()
        for ext in v:
            if not ext.startswith("."):
                validated.add(f".{ext}")
            else:
                validated.add(ext.lower())
        return validated

    class Config:
        frozen = False  # Allow updates after creation


# =============================================================================
# Model Configuration
# =============================================================================


class ModelConfig(BaseModel):
    """
    Configuration for model versions and providers.

    Attributes:
        default_vlm: Default vision-language model identifier
        default_llm: Default language model identifier
        default_od: Default object detection model identifier
        gemini_max_retries: Maximum retries for Gemini API calls
        gemini_temperature: Default temperature for Gemini models
    """

    default_vlm: str = Field(
        default="gemini/2.5-flash",
        description="Default vision-language model (e.g., 'gemini/2.5-flash')",
    )

    default_llm: str = Field(
        default="gemini/2.5-flash",
        description="Default language model (e.g., 'gemini/2.5-flash')",
    )

    default_od: str = Field(
        default="yolo/e",
        description="Default object detection model (e.g., 'yolo/e', 'yolo/v11')",
    )

    gemini_max_retries: int = Field(
        default=7,
        ge=1,
        le=20,
        description="Maximum number of retry attempts for Gemini API calls",
    )

    gemini_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Default temperature for Gemini model generation (0.0-2.0)",
    )

    @field_validator("default_od")
    @classmethod
    def validate_od_model(cls, v: str) -> str:
        """Validate object detection model identifier."""
        valid_models = {"yolo/v11", "yolo/v12", "yolo/world", "yolo/e"}
        if v not in valid_models:
            raise ValueError(
                f"Invalid object detection model '{v}'. "
                f"Valid options: {', '.join(sorted(valid_models))}"
            )
        return v

    class Config:
        frozen = False


# =============================================================================
# Security Configuration
# =============================================================================


class SecurityConfig(BaseModel):
    """
    Configuration for security and validation.

    Attributes:
        max_path_length: Maximum allowed file path length
        allowed_read_dirs: Set of directories allowed for reading
        allowed_write_dirs: Set of directories allowed for writing
        enable_path_validation: Whether to validate file paths before access
    """

    max_path_length: int = Field(
        default=4096, ge=256, le=32768, description="Maximum allowed file path length"
    )

    allowed_read_dirs: Optional[Set[str]] = Field(
        default=None,
        description="Directories allowed for reading (None = use defaults)",
    )

    allowed_write_dirs: Optional[Set[str]] = Field(
        default=None,
        description="Directories allowed for writing (None = use defaults)",
    )

    enable_path_validation: bool = Field(
        default=True, description="Enable path validation checks before file access"
    )

    @field_validator("max_path_length")
    @classmethod
    def validate_path_length(cls, v: int) -> int:
        """Warn if path length is unusually large."""
        if v > 8192:
            import warnings

            warnings.warn(
                f"Very large max_path_length ({v}) configured. "
                "This may allow unusually long paths."
            )
        return v

    class Config:
        frozen = False


# =============================================================================
# Benchmark Configuration
# =============================================================================


class BenchmarkConfig(BaseModel):
    """
    Configuration for benchmark execution.

    Attributes:
        circular_ref_depth_limit: Maximum depth when removing circular references
        enable_caching: Enable caching for VLM/LLM calls
        save_reasoning_traces: Save detailed reasoning traces to JSON
        debug_mode: Enable debug outputs and verbose logging
    """

    circular_ref_depth_limit: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum depth when removing circular references from results",
    )

    enable_caching: bool = Field(
        default=False, description="Enable caching for VLM/LLM API calls"
    )

    save_reasoning_traces: bool = Field(
        default=True, description="Save detailed reasoning traces to JSON files"
    )

    debug_mode: bool = Field(
        default=False, description="Enable debug mode with verbose logging and outputs"
    )

    @field_validator("circular_ref_depth_limit")
    @classmethod
    def validate_depth_limit(cls, v: int) -> int:
        """Warn if depth limit is too high."""
        if v > 500:
            import warnings

            warnings.warn(
                f"Very large circular_ref_depth_limit ({v}). "
                "This may allow deeply nested structures."
            )
        return v

    class Config:
        frozen = False


# =============================================================================
# Main Configuration
# =============================================================================


class SSAConfig(BaseModel):
    """
    Main configuration for SSA benchmarking system.

    This is the root configuration object that combines all subsystem configs.

    Attributes:
        paths: Path and file configuration
        models: Model version configuration
        security: Security settings
        benchmark: Benchmark execution settings

    Example:
        >>> config = SSAConfig()
        >>> print(config.models.default_vlm)
        'gemini/2.5-flash'
        >>> config.benchmark.enable_caching = True
        >>> config.validate_configuration()
    """

    paths: PathConfig = Field(
        default_factory=PathConfig, description="Path and file configuration"
    )

    models: ModelConfig = Field(
        default_factory=ModelConfig, description="Model version configuration"
    )

    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and validation configuration",
    )

    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig, description="Benchmark execution configuration"
    )

    def validate_configuration(self) -> None:
        """
        Validate the entire configuration and raise errors for critical issues.

        Raises:
            ValueError: If configuration has critical validation errors
        """
        # Validate output base exists or can be created
        try:
            self.paths.default_output_base.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(
                f"Cannot create output directory {self.paths.default_output_base}: {e}"
            )

        # Validate model configurations
        if not self.models.default_vlm:
            raise ValueError("default_vlm must be specified")
        if not self.models.default_llm:
            raise ValueError("default_llm must be specified")
        if not self.models.default_od:
            raise ValueError("default_od must be specified")

    def _update_path_config_from_env(self) -> None:
        """Update path configuration from environment variables."""
        if output_base := os.getenv("SSA_OUTPUT_BASE"):
            self.paths.default_output_base = Path(output_base)

        if random_chars := os.getenv("SSA_RANDOM_CHARS"):
            try:
                self.paths.default_random_chars = int(random_chars)
            except ValueError:
                pass  # Silently ignore invalid values

    def _update_model_versions_from_env(self) -> None:
        """Update model version settings from environment variables."""
        if vlm := os.getenv("SSA_DEFAULT_VLM"):
            self.models.default_vlm = vlm

        if llm := os.getenv("SSA_DEFAULT_LLM"):
            self.models.default_llm = llm

        if od := os.getenv("SSA_DEFAULT_OD"):
            try:
                self.models.default_od = od
            except ValueError as e:
                import warnings

                warnings.warn(f"Invalid SSA_DEFAULT_OD value: {e}")

    def _update_gemini_settings_from_env(self) -> None:
        """Update Gemini-specific settings from environment variables."""
        if retries := os.getenv("SSA_GEMINI_MAX_RETRIES"):
            try:
                self.models.gemini_max_retries = int(retries)
            except ValueError:
                pass

        if temp := os.getenv("SSA_GEMINI_TEMPERATURE"):
            try:
                self.models.gemini_temperature = float(temp)
            except ValueError:
                pass

    def _update_model_config_from_env(self) -> None:
        """Update model configuration from environment variables."""
        self._update_model_versions_from_env()
        self._update_gemini_settings_from_env()

    def _update_security_config_from_env(self) -> None:
        """Update security configuration from environment variables."""
        if max_path := os.getenv("SSA_MAX_PATH_LENGTH"):
            try:
                self.security.max_path_length = int(max_path)
            except ValueError:
                pass

        if enable_validation := os.getenv("SSA_ENABLE_PATH_VALIDATION"):
            self.security.enable_path_validation = enable_validation.lower() == "true"

    def _update_benchmark_config_from_env(self) -> None:
        """Update benchmark configuration from environment variables."""
        if depth := os.getenv("SSA_CIRCULAR_REF_DEPTH"):
            try:
                self.benchmark.circular_ref_depth_limit = int(depth)
            except ValueError:
                pass

        if caching := os.getenv("SSA_ENABLE_CACHING"):
            self.benchmark.enable_caching = caching.lower() == "true"

        if debug := os.getenv("SSA_DEBUG_MODE"):
            self.benchmark.debug_mode = debug.lower() == "true"

        if traces := os.getenv("SSA_SAVE_REASONING_TRACES"):
            self.benchmark.save_reasoning_traces = traces.lower() == "true"

    def update_from_env(self) -> None:
        """
        Update configuration from environment variables.

        Environment variables:
            SSA_OUTPUT_BASE: Override paths.default_output_base
            SSA_DEFAULT_VLM: Override models.default_vlm
            SSA_DEFAULT_LLM: Override models.default_llm
            SSA_DEFAULT_OD: Override models.default_od
            SSA_MAX_PATH_LENGTH: Override security.max_path_length
            SSA_CIRCULAR_REF_DEPTH: Override benchmark.circular_ref_depth_limit
            SSA_ENABLE_CACHING: Override benchmark.enable_caching
            SSA_DEBUG_MODE: Override benchmark.debug_mode
        """
        self._update_path_config_from_env()
        self._update_model_config_from_env()
        self._update_security_config_from_env()
        self._update_benchmark_config_from_env()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "paths": self.paths.model_dump(),
            "models": self.models.model_dump(),
            "security": self.security.model_dump(),
            "benchmark": self.benchmark.model_dump(),
        }

    def print_config(self) -> None:
        """Print current configuration in a readable format."""
        print("=" * 60)
        print("SSA Configuration")
        print("=" * 60)

        print("\n[Path Configuration]")
        print(f"  Output Base:          {self.paths.default_output_base}")
        print(f"  Random ID Length:     {self.paths.default_random_chars}")
        print(
            f"  Image Formats:        {', '.join(sorted(self.paths.supported_image_formats))}"
        )

        print("\n[Model Configuration]")
        print(f"  Default VLM:          {self.models.default_vlm}")
        print(f"  Default LLM:          {self.models.default_llm}")
        print(f"  Default OD:           {self.models.default_od}")
        print(f"  Gemini Max Retries:   {self.models.gemini_max_retries}")
        print(f"  Gemini Temperature:   {self.models.gemini_temperature}")

        print("\n[Security Configuration]")
        print(f"  Max Path Length:      {self.security.max_path_length}")
        print(f"  Path Validation:      {self.security.enable_path_validation}")

        print("\n[Benchmark Configuration]")
        print(f"  Circular Ref Depth:   {self.benchmark.circular_ref_depth_limit}")
        print(f"  Caching Enabled:      {self.benchmark.enable_caching}")
        print(f"  Save Traces:          {self.benchmark.save_reasoning_traces}")
        print(f"  Debug Mode:           {self.benchmark.debug_mode}")

        print("=" * 60)

    class Config:
        frozen = False
        validate_assignment = True


# =============================================================================
# Global Configuration Instance
# =============================================================================

_global_config: Optional[SSAConfig] = None


def get_config() -> SSAConfig:
    """
    Get the global configuration instance.

    Loads configuration from environment variables on first access.

    Returns:
        The global SSAConfig instance

    Example:
        >>> config = get_config()
        >>> print(config.models.default_vlm)
    """
    global _global_config
    if _global_config is None:
        _global_config = SSAConfig()
        _global_config.update_from_env()
    return _global_config


def set_config(config: SSAConfig) -> None:
    """
    Set the global configuration instance.

    Args:
        config: SSAConfig instance to set as global

    Example:
        >>> custom_config = SSAConfig()
        >>> custom_config.benchmark.debug_mode = True
        >>> set_config(custom_config)
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """
    Reset configuration to defaults.

    This clears the global configuration and forces reinitialization
    on next access.
    """
    global _global_config
    _global_config = None


# =============================================================================
# Legacy Exports (for backward compatibility)
# =============================================================================

# These module-level constants are maintained for backward compatibility
# with existing code that imports them directly

SUPPORTED_IMAGE_FORMATS = get_config().paths.supported_image_formats
DEFAULT_OUTPUT_BASE = get_config().paths.default_output_base


def update_legacy_exports() -> None:
    """Update module-level constants to match current config."""
    global SUPPORTED_IMAGE_FORMATS, DEFAULT_OUTPUT_BASE
    config = get_config()
    SUPPORTED_IMAGE_FORMATS = config.paths.supported_image_formats
    DEFAULT_OUTPUT_BASE = config.paths.default_output_base


# =============================================================================
# Validation and Testing Utilities
# =============================================================================


def validate_startup_config() -> None:
    """
    Validate configuration at startup and log any issues.

    This should be called during application initialization to catch
    configuration errors early.

    Raises:
        ValueError: If configuration has critical errors
    """
    config = get_config()

    try:
        config.validate_configuration()
        print("✅ Configuration validation successful")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


def print_config_help() -> None:
    """Print help information about configuration options."""
    print(
        """
SSA Configuration Help
======================

The SSA benchmarking system can be configured via environment variables:

Path Configuration:
  SSA_OUTPUT_BASE              Base directory for benchmark outputs
  SSA_RANDOM_CHARS             Length of random IDs (4-32, default: 12)

Model Configuration:
  SSA_DEFAULT_VLM              Default vision-language model
  SSA_DEFAULT_LLM              Default language model
  SSA_DEFAULT_OD               Default object detection model
  SSA_GEMINI_MAX_RETRIES       Gemini API retry limit (1-20, default: 7)
  SSA_GEMINI_TEMPERATURE       Gemini temperature (0.0-2.0, default: 0.5)

Security Configuration:
  SSA_MAX_PATH_LENGTH          Maximum path length (default: 4096)
  SSA_ENABLE_PATH_VALIDATION   Enable path validation (true/false)

Benchmark Configuration:
  SSA_CIRCULAR_REF_DEPTH       Circular reference depth limit (10-1000, default: 100)
  SSA_ENABLE_CACHING           Enable VLM/LLM caching (true/false)
  SSA_SAVE_REASONING_TRACES    Save reasoning traces (true/false, default: true)
  SSA_DEBUG_MODE               Enable debug mode (true/false)

Example:
  export SSA_DEFAULT_VLM="gemini/2.5-flash"
  export SSA_ENABLE_CACHING="true"
  export SSA_DEBUG_MODE="true"

For programmatic access:
  from ssa.config import get_config
  config = get_config()
  config.print_config()
"""
    )
