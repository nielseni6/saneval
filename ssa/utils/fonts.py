"""
Font configuration and utilities for the SSA project.

This module provides centralized font path management to avoid hardcoded
font paths throughout the codebase and reduce coupling between modules.
"""

from pathlib import Path

# Base font directory - relative to project root
_FONT_DIR = Path("ssa/assets/fonts")

# Font path constants
ROBOTO_REGULAR = str(_FONT_DIR / "roboto.ttf")
ROBOTO_BOLD = str(_FONT_DIR / "robotobold.ttf")
ARIAL_REGULAR = str(_FONT_DIR / "arial.ttf")
ARIAL_BOLD = str(_FONT_DIR / "arialbold.ttf")
MONTSERRAT_REGULAR = str(_FONT_DIR / "montserrat.ttf")
MONTSERRAT_BOLD = str(_FONT_DIR / "montserratbold.ttf")
POPPINS_REGULAR = str(_FONT_DIR / "Poppins-Regular.ttf")

# Default fonts for different use cases
DEFAULT_BODY_FONT = ROBOTO_REGULAR
DEFAULT_BOLD_BODY_FONT = ROBOTO_BOLD
DEFAULT_HEADING_FONT = MONTSERRAT_REGULAR
DEFAULT_BOLD_HEADING_FONT = MONTSERRAT_BOLD
DEFAULT_WATERMARK_FONT = POPPINS_REGULAR


def get_font_path(font_name: str) -> str:
    """
    Get the full path to a font file.

    Args:
        font_name: Name of the font (e.g., 'roboto.ttf' or 'Poppins-Regular.ttf')

    Returns:
        Full path to the font file

    Raises:
        FileNotFoundError: If the font file doesn't exist
    """
    font_path = _FONT_DIR / font_name
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    return str(font_path)


def font_exists(font_name: str) -> bool:
    """
    Check if a font file exists.

    Args:
        font_name: Name of the font file

    Returns:
        True if the font file exists, False otherwise
    """
    return (_FONT_DIR / font_name).exists()
