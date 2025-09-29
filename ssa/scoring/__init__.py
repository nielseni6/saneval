"""
Standardized scoring infrastructure package.
"""

# Scoring method constants
DSG_VLM = "dsg-vlm"
BOC_VLM = "boc-vlm"
GEN_EVAL = "gen-eval"
HPSV2 = "hps-v2"
PICK_SCORE = "pick-score"
COMPBENCH = "spatial-numeracy"  # Deprecated, use SPATIAL or NUMERACY
SPATIAL = "spatial"
NUMERACY = "numeracy"
OD_ATTR_BINDING = "od-attr-binding"  # Object Detection-Based Attribute Binding (ODBAB)
FACE_ANALYSIS = "face-analysis"
FACE_ID = "face-id"
ROUNDTRIP = "roundtrip"
MOCK_I2I = "mock-i2i"
I2I_CRITERIA_VLM = "i2i-criteria-vlm"
SCORING_METHODS = [
    DSG_VLM,
    GEN_EVAL,
    HPSV2,
    PICK_SCORE,
    COMPBENCH,
    SPATIAL,
    NUMERACY,
    OD_ATTR_BINDING,
    ROUNDTRIP,
    FACE_ANALYSIS,
    FACE_ID,
    BOC_VLM,
    MOCK_I2I,
    I2I_CRITERIA_VLM,
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
    DSGVLMScorerAdapter,
    FaceIdScorerAdapter,
    GenEvalScorerAdapter,
    HPSv2ScorerAdapter,
    PickScoreScorerAdapter,
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
    "DSGVLMScorerAdapter",
    "BOCVLMScorerAdapter",
    "FaceIdScorerAdapter",
    "GenEvalScorerAdapter",
    "HPSv2ScorerAdapter",
    "PickScoreScorerAdapter",
    "create_scorer_adapter",
    # Scoring method constants
    "DSG_VLM",
    "BOC_VLM",
    "GEN_EVAL",
    "HPSV2",
    "PICK_SCORE",
    "COMPBENCH",
    "SPATIAL",
    "NUMERACY",
    "OD_ATTR_BINDING",
    "FACE_ANALYSIS",
    "FACE_ID",
    "ROUNDTRIP",
    "MOCK_I2I",
    "SCORING_METHODS",
    "VALID_SCORING_METHODS",
]
