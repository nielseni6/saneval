"""
Standardized scoring infrastructure package.
"""

# Scoring method constants
SPATIAL = "spatial"
NUMERACY = "numeracy"
OD_ATTR_BINDING = "od-attr-binding"  # Object Detection-Based Attribute Binding (ODBAB)
SCORING_METHODS = [
    SPATIAL,
    NUMERACY,
    OD_ATTR_BINDING,
]

SPATIAL_QUERIES = [
    "on side of",
    "next to",
    "near",
    "on the left of",
    "on the right of",
    "on the bottom of",
    "on the top of",
    "on top of",
]

NUMERIC_QUERIES = [
    "a",
    "an",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
]

# Set for efficient membership testing
VALID_SCORING_METHODS = set(SCORING_METHODS)

from .adapters import (
    create_scorer_adapter,
)
from .interfaces import (
    AggregatedResults,
    BenchmarkScoreResult,
    ScoreResult,
    ScorerInterface,
)
from .runner import (
    PromptContext,
    StandardizedBenchmarkRunner,
    create_prompt_context_from_legacy,
)

__all__ = [
    # Interfaces and results
    "ScoreResult",
    "BenchmarkScoreResult",
    "ScorerInterface",
    "AggregatedResults",
    # Runner components
    "StandardizedBenchmarkRunner",
    "PromptContext",
    "create_prompt_context_from_legacy",
    # Scorer adapters
    "create_scorer_adapter",
    # Scoring method constants
    "SPATIAL",
    "NUMERACY",
    "OD_ATTR_BINDING",
    "SCORING_METHODS",
    "VALID_SCORING_METHODS",
]
