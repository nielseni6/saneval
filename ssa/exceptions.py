"""
Custom exception classes for SSA benchmarking.

This module defines domain-specific exceptions to provide clearer error handling
and more informative error messages throughout the codebase.
"""


class SSAError(Exception):
    """Base exception class for all SSA-related errors."""

    pass


class ScorerError(SSAError):
    """Base exception for scorer-related errors."""

    pass


class MetricAggregationError(ScorerError):
    """
    Exception raised when metric aggregation fails.

    This can occur when:
    - Required attributes are missing from scorer objects
    - Metric calculations encounter invalid data
    - Aggregation operations fail due to type mismatches
    """

    pass


class InvalidConfigurationError(SSAError):
    """
    Exception raised for invalid configuration parameters.

    This covers:
    - Invalid model keys or configuration values
    - Missing required configuration fields
    - Configuration values outside allowed ranges
    """

    pass


class ModelError(SSAError):
    """
    Exception raised for model initialization or execution errors.

    This includes:
    - Model loading failures
    - Invalid model parameters
    - Runtime errors during model inference
    """

    pass


class ValidationError(SSAError):
    """
    Exception raised for input validation failures.

    Used when:
    - Input parameters are of incorrect type
    - Input values are outside valid ranges
    - Required inputs are missing
    """

    pass
