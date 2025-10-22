"""
Central configuration for SSA benchmarking.

This module contains global constants and configuration settings used throughout
the SSA benchmarking system. Centralizing these values improves maintainability
and makes it easier to modify behavior across the codebase.
"""

from pathlib import Path

# Supported image formats for benchmark evaluation
# These extensions are used when scanning directories for images to evaluate
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# Default output directory base path
# When output_dir is not specified, debug outputs are saved here
DEFAULT_OUTPUT_BASE = Path("results/unknown")
