"""
Aspect ratio utilities for mapping requested ratios to model-supported ratios.

This module provides centralized aspect ratio handling for all image generation providers,
ensuring consistent behavior and optimal mapping of user requests to model capabilities.
"""

from typing import Dict, List, Optional, Tuple, Union

from ssa.utils.logging import get_log

log = get_log(__file__)


def aspect_ratio_to_float(aspect_ratio: str) -> float:
    """
    Convert aspect ratio string to float ratio.

    Args:
        aspect_ratio: Ratio in format "W:H" (e.g., "16:9", "1:1", "4:3")

    Returns:
        Float ratio (width / height)

    Raises:
        ValueError: If aspect_ratio format is invalid

    Examples:
        >>> aspect_ratio_to_float("16:9")
        1.7777777777777777
        >>> aspect_ratio_to_float("1:1")
        1.0
        >>> aspect_ratio_to_float("3:4")
        0.75
    """
    try:
        parts = aspect_ratio.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid aspect ratio format: {aspect_ratio}. Expected 'W:H'"
            )

        width_ratio = float(parts[0])
        height_ratio = float(parts[1])

        if width_ratio <= 0 or height_ratio <= 0:
            raise ValueError(
                f"Aspect ratio components must be positive: {aspect_ratio}"
            )

        return width_ratio / height_ratio
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid aspect ratio '{aspect_ratio}': {e}")


def aspect_ratio_to_dimensions(
    aspect_ratio: str, target_size: int = 1024
) -> tuple[int, int]:
    """
    Convert aspect ratio string to width/height dimensions.

    Args:
        aspect_ratio: Ratio in format "W:H" (e.g., "16:9", "1:1", "4:3")
        target_size: Target size for the larger dimension

    Returns:
        (width, height) tuple maintaining the aspect ratio

    Raises:
        ValueError: If aspect_ratio format is invalid

    Examples:
        >>> aspect_ratio_to_dimensions("16:9", 1024)
        (1024, 576)
        >>> aspect_ratio_to_dimensions("1:1", 1024)
        (1024, 1024)
        >>> aspect_ratio_to_dimensions("9:16", 1024)
        (576, 1024)
    """
    try:
        # Parse the aspect ratio string
        width_ratio, height_ratio = aspect_ratio.split(":")
        width_ratio = float(width_ratio)
        height_ratio = float(height_ratio)

        if width_ratio <= 0 or height_ratio <= 0:
            raise ValueError("Aspect ratio components must be positive numbers")

    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid aspect ratio format: '{aspect_ratio}'. "
            f"Expected format 'W:H' (e.g., '16:9', '1:1', '4:3')"
        ) from e

    # Calculate the actual ratio
    ratio = width_ratio / height_ratio

    if ratio >= 1.0:
        # Landscape or square: width is the larger dimension
        width = target_size
        height = int(target_size / ratio)
    else:
        # Portrait: height is the larger dimension
        height = target_size
        width = int(target_size * ratio)

    return (width, height)


def get_provider_dimensions_legacy(
    aspect_ratio: str, provider_id: str
) -> tuple[int, int]:
    """
    DEPRECATED: Use resolve_aspect_ratio_and_dimensions() instead.

    Get provider-specific optimal dimensions for a given aspect ratio.

    This function is kept for backward compatibility but new code should use
    resolve_aspect_ratio_and_dimensions() which provides more comprehensive
    aspect ratio handling.

    Args:
        aspect_ratio: Aspect ratio in format "W:H" (e.g., "16:9", "1:1", "4:3")
        provider_id: Provider identifier (e.g., "imagen/4.0", "openai/gpt-4o")

    Returns:
        (width, height) tuple with provider-optimized dimensions

    Raises:
        ValueError: If aspect_ratio format is invalid or provider not supported
    """
    # Provider-specific dimension mappings
    PROVIDER_DIMENSIONS = {
        # Imagen models - based on ASPECT_RATIO_MAP in imagen.py
        "imagen/3.0": {
            "1:1": (1024, 1024),
            "3:4": (896, 1280),
            "4:3": (1280, 896),
            "9:16": (768, 1408),
            "16:9": (1408, 768),
        },
        "imagen/4.0": {
            "1:1": (1024, 1024),
            "3:4": (896, 1280),
            "4:3": (1280, 896),
            "9:16": (768, 1408),
            "16:9": (1408, 768),
        },
        "imagen/4.0-fast": {
            "1:1": (1024, 1024),
            "3:4": (896, 1280),
            "4:3": (1280, 896),
            "9:16": (768, 1408),
            "16:9": (1408, 768),
        },
        "imagen/4.0-ultra": {
            "1:1": (1024, 1024),
            "3:4": (896, 1280),
            "4:3": (1280, 896),
            "9:16": (768, 1408),
            "16:9": (1408, 768),
        },
        # OpenAI models - based on OPENAI_ASPECT_RATIO_TO_SIZE in openai.py
        "openai/gpt-4o": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
        "openai/gpt-4.5-preview": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
        "openai/dall-e-3": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
    }

    # Get provider-specific dimensions
    provider_dims = PROVIDER_DIMENSIONS.get(provider_id)
    if not provider_dims:
        # Fallback to generic conversion for unknown providers
        log.warning(
            f"No provider-specific dimensions found for {provider_id}, using generic conversion"
        )
        return aspect_ratio_to_dimensions(aspect_ratio, target_size=1024)

    # Try direct lookup first
    dimensions = provider_dims.get(aspect_ratio)
    if dimensions:
        log.debug(
            f"Provider {provider_id}: Direct mapping '{aspect_ratio}' -> {dimensions}"
        )
        return dimensions

    # If no direct match, find closest supported aspect ratio and use its dimensions
    supported_ratios = list(provider_dims.keys())
    closest_ratio = find_closest_supported_aspect_ratio(aspect_ratio, supported_ratios)

    dimensions = provider_dims[closest_ratio]
    log.debug(
        f"Provider {provider_id}: Mapped '{aspect_ratio}' -> '{closest_ratio}' -> {dimensions}"
    )

    return dimensions


# Backward compatibility - redirect old function calls to new unified function
def get_provider_dimensions(aspect_ratio: str, provider_id: str) -> tuple[int, int]:
    """
    Backward compatibility wrapper for get_provider_dimensions.

    New code should use resolve_aspect_ratio_and_dimensions() directly.
    """
    width, height, _ = resolve_aspect_ratio_and_dimensions(
        aspect_ratio, provider_id, 1024, 1024
    )
    return width, height


def find_closest_aspect_ratio(width: int, height: int) -> str:
    """
    Find the closest standard aspect ratio for given width and height.

    Args:
        width: Width in pixels
        height: Height in pixels

    Returns:
        String representation of the closest aspect ratio (e.g., "16:9", "4:3", "9:16")
    """
    # Common aspect ratios (ratio_value, string_representation)
    # Include both landscape and portrait versions
    accepted_ratios_ = [
        "1:1",
        "1:2",
        "1:3",
        "2:3",
        "3:4",
        "3:7",
        "4:5",
        "4:7",
        "9:16",
        "9:21",
    ]

    # make a list of standard ratios of the form (a/b, "a:b") for all elements in accepted_ratios_, and their reverses
    standard_ratios = []
    for ratio_str in accepted_ratios_:
        # Add the original ratio
        ratio_float = aspect_ratio_to_float(ratio_str)
        standard_ratios.append((ratio_float, ratio_str))

        # Add the reverse ratio (unless it's 1:1)
        if ratio_str != "1:1":
            parts = ratio_str.split(":")
            reverse_str = f"{parts[1]}:{parts[0]}"
            reverse_float = aspect_ratio_to_float(reverse_str)
            standard_ratios.append((reverse_float, reverse_str))

    # Calculate the actual aspect ratio
    if height == 0:
        return "1:1"  # Return a safe default for invalid height
    actual_ratio = width / height

    # Find the closest ratio
    closest_ratio = min(standard_ratios, key=lambda x: abs(x[0] - actual_ratio))

    return closest_ratio[1]


def find_closest_supported_aspect_ratio(
    requested_ratio: str, supported_ratios: List[str], tolerance: float = 0.1
) -> str:
    """
    Find the closest supported aspect ratio to the requested one.

    Args:
        requested_ratio: Requested aspect ratio string (e.g., "16:9")
        supported_ratios: List of supported aspect ratio strings
        tolerance: Maximum allowed difference before falling back to 1:1

    Returns:
        Closest supported aspect ratio string

    Examples:
        >>> find_closest_supported_aspect_ratio("16:10", ["16:9", "4:3", "1:1"])
        "16:9"
        >>> find_closest_supported_aspect_ratio("2:1", ["1:1", "3:2", "4:3"])
        "3:2"
    """
    if not supported_ratios:
        log.warning("No supported aspect ratios provided, defaulting to '1:1'")
        return "1:1"

    # If exact match exists, return it
    if requested_ratio in supported_ratios:
        return requested_ratio

    try:
        requested_float = aspect_ratio_to_float(requested_ratio)
    except ValueError as e:
        log.error(f"Invalid requested aspect ratio '{requested_ratio}': {e}")
        return supported_ratios[0] if supported_ratios else "1:1"

    # Find closest ratio by mathematical distance
    closest_ratio = None
    min_distance = float("inf")

    for supported_ratio in supported_ratios:
        try:
            supported_float = aspect_ratio_to_float(supported_ratio)
            distance = abs(requested_float - supported_float)

            if distance < min_distance:
                min_distance = distance
                closest_ratio = supported_ratio

        except ValueError as e:
            log.warning(f"Invalid supported aspect ratio '{supported_ratio}': {e}")
            continue

    if closest_ratio is None:
        log.warning(
            "No valid supported aspect ratios found, defaulting to first available"
        )
        return supported_ratios[0] if supported_ratios else "1:1"

    # Check if the closest ratio is within tolerance
    if min_distance > tolerance:
        log.warning(
            f"Closest aspect ratio '{closest_ratio}' (distance: {min_distance:.3f}) "
            f"exceeds tolerance {tolerance} for requested '{requested_ratio}'"
        )

    log.debug(
        f"Mapped aspect ratio '{requested_ratio}' -> '{closest_ratio}' "
        f"(distance: {min_distance:.3f})"
    )

    return closest_ratio


# Model aspect ratio registry with comprehensive dimension mappings
MODEL_ASPECT_RATIOS: Dict[
    str,
    Dict[str, Union[List[str], List[Tuple[int, int]], Dict[str, Tuple[int, int]], str]],
] = {
    "aws/2ndsetai": {
        "supported_aspect_ratios": [
            "1.91:1"
        ],  # Initial, flexible via Nova Canvas API (1:4 to 4:1)
        "supported_resolutions": [(1200, 628)],
        "aspect_ratio_dimensions": {
            "1.91:1": (1200, 628),
        },
        "control_method": "predefined_custom",
    },
    "aws/kandinsky-3-1": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Flexible via Nova Canvas API
        "supported_resolutions": [(1024, 1024)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
        },
        "control_method": "custom_pixels",
    },
    "aws/pixart-sigma-900m": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Flexible via Nova Canvas API
        "supported_resolutions": [(1024, 1024)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
        },
        "control_method": "custom_pixels",
    },
    "aws/sana-1-0": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Flexible via Nova Canvas API
        "supported_resolutions": [(4096, 4096)],
        "aspect_ratio_dimensions": {
            "1:1": (4096, 4096),
            "4:3": (4096, 3072),
            "3:4": (3072, 4096),
            "16:9": (4096, 2304),
            "9:16": (2304, 4096),
        },
        "control_method": "custom_pixels",
    },
    "aws/sana-1-5": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Flexible via Nova Canvas API
        "supported_resolutions": [(4096, 4096)],
        "aspect_ratio_dimensions": {
            "1:1": (4096, 4096),
            "4:3": (4096, 3072),
            "3:4": (3072, 4096),
            "16:9": (4096, 2304),
            "9:16": (2304, 4096),
        },
        "control_method": "custom_pixels",
    },
    "aws/sana-1600m-cnet": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Flexible via Nova Canvas API
        "supported_resolutions": [(1024, 1024)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
        },
        "control_method": "custom_pixels",
    },
    "aws/sana-sprint": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Flexible via Nova Canvas API
        "supported_resolutions": [(1024, 1024)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
        },
        "control_method": "custom_pixels",
    },
    # Azure models
    "azure/gpt-image-1": {
        "supported_aspect_ratios": ["1:1", "3:2", "2:3"],
        "supported_resolutions": [(1024, 1024), (1536, 1024), (1024, 1536)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
        "control_method": "predefined_presets",
    },
    # BFL models
    "bfl.ml/v1/flux-dev": {
        "supported_aspect_ratios": [
            "3:7",
            "1:2",
            "2:3",
            "3:4",
            "1:1",
            "4:3",
            "3:2",
            "2:1",
            "7:3",
        ],  # Any ratio from 3:7 to 7:3
        "supported_resolutions": "~1_megapixel",
        "aspect_ratio_dimensions": {
            "3:7": (655, 1524),  # ~1MP
            "1:2": (724, 1448),
            "2:3": (836, 1254),
            "3:4": (886, 1182),
            "1:1": (1024, 1024),
            "4:3": (1182, 886),
            "3:2": (1254, 836),
            "2:1": (1448, 724),
            "7:3": (1524, 655),
        },
        "control_method": "custom_ratio",
    },
    # FAL models
    "fal-ai/bagel": {
        "supported_aspect_ratios": ["1:1"],  # From example
        "supported_resolutions": [(1024, 1024)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
        },
        "control_method": "undocumented",
    },
    "fal-ai/flux-dev": {
        "supported_aspect_ratios": [
            "1:1",
            "3:4",
            "4:3",
            "9:16",
            "16:9",
            "9:21",
            "21:9",
        ],
        "supported_resolutions": "not_specified",
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "3:4": (886, 1182),
            "4:3": (1182, 886),
            "9:16": (724, 1288),
            "16:9": (1288, 724),
            "9:21": (616, 1440),
            "21:9": (1440, 616),
        },
        "control_method": "predefined_presets",
    },
    "fal-ai/hidream-i1-fast": {
        "supported_aspect_ratios": ["4:3", "16:9", "1:1"],  # Portrait/Landscape, Square
        "supported_resolutions": "square_hd_square_custom",
        "aspect_ratio_dimensions": {
            "4:3": (1024, 768),
            "16:9": (1024, 576),
            "1:1": (1024, 1024),
        },
        "control_method": "presets_custom_pixels",
    },
    "fal-ai/hidream-i1-full": {
        "supported_aspect_ratios": ["4:3", "16:9", "1:1"],  # Portrait/Landscape, Square
        "supported_resolutions": "square_hd_square_custom",
        "aspect_ratio_dimensions": {
            "4:3": (1024, 768),
            "16:9": (1024, 576),
            "1:1": (1024, 1024),
        },
        "control_method": "presets_custom_pixels",
    },
    "fal-ai/qwen-image-edit": {
        "supported_aspect_ratios": [
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "9:16",
            "16:9",
            "1:3",
            "3:1",
        ],
        "supported_resolutions": "multiples_of_112",
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "2:3": (896, 1344),
            "3:2": (1344, 896),
            "3:4": (896, 1120),
            "4:3": (1120, 896),
            "9:16": (784, 1344),
            "16:9": (1344, 784),
            "1:3": (672, 1904),
            "3:1": (1904, 672),
        },
        "control_method": "match_input_image",
    },
    "fal-ai/seedream-3.0": {
        "supported_aspect_ratios": [
            "1:1",
            "3:4",
            "4:3",
            "16:9",
            "9:16",
            "2:3",
            "3:2",
            "21:9",
        ],
        "supported_resolutions": "native_2k",
        "aspect_ratio_dimensions": {
            "1:1": (2048, 2048),
            "3:4": (1774, 2366),
            "4:3": (2366, 1774),
            "16:9": (2560, 1440),
            "9:16": (1440, 2560),
            "2:3": (1670, 2506),
            "3:2": (2506, 1670),
            "21:9": (3008, 1290),
        },
        "control_method": "predefined_presets",
    },
    # Gemini models
    "gemini-image/2.5-flash": {
        "supported_aspect_ratios": [],  # does not support aspect ratios directly
        "supported_resolutions": "undocumented",
        "aspect_ratio_dimensions": {},  # No aspect ratio support
        "control_method": "ask nicely",
    },
    # Imagen models
    "imagen/3.0": {
        "supported_aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        "supported_resolutions": [
            (1024, 1024),
            (1280, 896),
            (896, 1280),
            (1408, 768),
            (768, 1408),
        ],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1280, 896),
            "3:4": (896, 1280),
            "16:9": (1408, 768),
            "9:16": (768, 1408),
        },
        "control_method": "predefined_presets",
    },
    "imagen/4.0": {
        "supported_aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        "supported_resolutions": [
            (1024, 1024),
            (1280, 896),
            (896, 1280),
            (1408, 768),
            (768, 1408),
            (2048, 2048),
        ],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1280, 896),
            "3:4": (896, 1280),
            "16:9": (1408, 768),
            "9:16": (768, 1408),
        },
        "control_method": "predefined_presets",
    },
    "imagen/4.0-fast": {
        "supported_aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        "supported_resolutions": [
            (1024, 1024),
            (1280, 896),
            (896, 1280),
            (1408, 768),
            (768, 1408),
        ],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1280, 896),
            "3:4": (896, 1280),
            "16:9": (1408, 768),
            "9:16": (768, 1408),
        },
        "control_method": "predefined_presets",
    },
    "imagen/4.0-ultra": {
        "supported_aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
        "supported_resolutions": [
            (1024, 1024),
            (1280, 896),
            (896, 1280),
            (1408, 768),
            (768, 1408),
            (2048, 2048),
        ],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "4:3": (1280, 896),
            "3:4": (896, 1280),
            "16:9": (1408, 768),
            "9:16": (768, 1408),
        },
        "control_method": "predefined_presets",
    },
    # OpenAI models
    "openai/gpt-4o": {
        "supported_aspect_ratios": ["1:1", "3:2", "2:3"],
        "supported_resolutions": [(1024, 1024), (1536, 1024), (1024, 1536)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
        "control_method": "predefined_presets",
    },
    "openai/gpt-4.5-preview": {
        "supported_aspect_ratios": ["1:1", "3:2", "2:3"],
        "supported_resolutions": [(1024, 1024), (1536, 1024), (1024, 1536)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
        "control_method": "predefined_presets",
    },
    "openai/dall-e-3": {
        "supported_aspect_ratios": ["1:1", "3:2", "2:3"],
        "supported_resolutions": [(1024, 1024), (1536, 1024), (1024, 1536)],
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "3:2": (1536, 1024),
            "2:3": (1024, 1536),
        },
        "control_method": "predefined_presets",
    },
    # Replicate models
    "replicate/flux-dev": {
        "supported_aspect_ratios": [
            "1:1",
            "16:9",
            "21:9",
            "3:2",
            "2:3",
            "4:5",
            "5:4",
            "3:4",
            "4:3",
            "9:16",
            "9:21",
        ],
        "supported_resolutions": "controlled_by_megapixels",
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1152),
            "5:4": (1152, 896),
            "3:4": (896, 1152),
            "4:3": (1152, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        },
        "control_method": "predefined_presets",
    },
    "replicate/flux-schnell": {
        "supported_aspect_ratios": [
            "1:1",
            "16:9",
            "21:9",
            "3:2",
            "2:3",
            "4:5",
            "5:4",
            "3:4",
            "4:3",
            "9:16",
            "9:21",
        ],
        "supported_resolutions": "controlled_by_megapixels",
        "aspect_ratio_dimensions": {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1152),
            "5:4": (1152, 896),
            "3:4": (896, 1152),
            "4:3": (1152, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        },
        "control_method": "predefined_presets",
    },
    "replicate/flux-pro-ultra": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Custom width/height support
        "supported_resolutions": "up_to_4_megapixels",
        "aspect_ratio_dimensions": {
            "1:1": (2048, 2048),  # Up to 4MP
            "4:3": (2304, 1728),
            "3:4": (1728, 2304),
            "16:9": (2688, 1512),
            "9:16": (1512, 2688),
        },
        "control_method": "custom_pixels",
    },
    "replicate/ideogram-v3-quality": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
            "3:2",
            "2:3",
            "21:9",
            "9:21",
            "5:4",
            "4:5",
        ],  # 15+ presets
        "supported_resolutions": "extensive_list_up_to_1.5mp",
        "control_method": "presets_custom_res",
    },
    "replicate/ideogram-v3-turbo": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
            "3:2",
            "2:3",
            "21:9",
            "9:21",
            "5:4",
            "4:5",
        ],  # 15+ presets
        "supported_resolutions": "extensive_list_up_to_1.5mp",
        "control_method": "presets_custom_res",
    },
    "replicate/photon": {
        "supported_aspect_ratios": [
            "1:1",
            "3:4",
            "4:3",
            "9:16",
            "16:9",
            "9:21",
            "21:9",
        ],
        "supported_resolutions": "not_specified",
        "control_method": "predefined_presets",
    },
    "replicate/photon-flash": {
        "supported_aspect_ratios": [
            "1:1",
            "3:4",
            "4:3",
            "9:16",
            "16:9",
            "9:21",
            "21:9",
        ],
        "supported_resolutions": "not_specified",
        "control_method": "predefined_presets",
    },
    "replicate/qwen-edit": {
        "supported_aspect_ratios": [
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "9:16",
            "16:9",
            "1:3",
            "3:1",
        ],
        "supported_resolutions": "multiples_of_112",
        "control_method": "match_input_image",
    },
    "replicate/recraft-v3": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # aspect_ratio supported but not enumerated
        "supported_resolutions": [(1024, 1024)],  # Default
        "control_method": "presets_custom_size",
    },
    "replicate/sana": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Base model capability
        "supported_resolutions": [(4096, 4096)],
        "control_method": "undocumented",
    },
    "replicate/sana-sprint": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
        ],  # Base model capability
        "supported_resolutions": [(1024, 1024)],
        "control_method": "undocumented",
    },
    "replicate/flux-kontext-pro": {
        "supported_aspect_ratios": [
            "1:1",
            "16:9",
            "21:9",
            "3:2",
            "2:3",
            "4:5",
            "5:4",
            "3:4",
            "4:3",
            "9:16",
            "9:21",
        ],
        "supported_resolutions": "~1mp_total_pixels",
        "control_method": "match_input_presets",
    },
    "azure/flux-kontext-pro": {
        "supported_aspect_ratios": [
            "1:1",
            "16:9",
            "21:9",
            "3:2",
            "2:3",
            "4:5",
            "5:4",
            "3:4",
            "4:3",
            "9:16",
            "9:21",
        ],
        "supported_resolutions": "~1mp_total_pixels",
        "control_method": "match_input_presets",
    },
    "replicate/flux-kontext-max": {
        "supported_aspect_ratios": [
            "1:1",
            "16:9",
            "21:9",
            "3:2",
            "2:3",
            "4:5",
            "5:4",
            "3:4",
            "4:3",
            "9:16",
            "9:21",
        ],
        "supported_resolutions": "~1mp_total_pixels",
        "control_method": "match_input_presets",
    },
    "replicate/sd-35l-turbo": {
        "supported_aspect_ratios": ["1:1"],
        "supported_resolutions": "not_specified",
        "control_method": "predefined_presets",
    },
    "replicate/seedream-v4": {
        "supported_aspect_ratios": [
            "1:1",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
            "3:2",
            "2:3",
            "21:9",
        ],
        "supported_resolutions": "1k_to_4k_presets_custom",
        "aspect_ratio_dimensions": {
            "1:1": (2048, 2048),
            "4:3": (2366, 1774),
            "3:4": (1774, 2366),
            "16:9": (2560, 1440),
            "9:16": (1440, 2560),
            "3:2": (2506, 1670),
            "2:3": (1670, 2506),
            "21:9": (3008, 1290),
        },
        "control_method": "predefined_presets",
    },
    # XAI models
    "xai/grok-2-image": {
        "supported_aspect_ratios": [],  # does not support aspect ratios
        "supported_resolutions": "fixed_1024x1024",
        "aspect_ratio_dimensions": {},  # No aspect ratio support, fixed 1024x1024
        "control_method": "does not support aspect ratios",
    },
}


def get_model_aspect_ratios(model_id: str) -> List[str]:
    """
    Get supported aspect ratios for a specific model.

    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o", "replicate/flux-dev")

    Returns:
        List of supported aspect ratio strings

    Examples:
        >>> get_model_aspect_ratios("openai/gpt-4o")
        ["1:1", "3:2", "2:3"]
        >>> get_model_aspect_ratios("unknown/model")
        ["1:1"]
    """
    model_info = MODEL_ASPECT_RATIOS.get(model_id, {})
    supported_ratios = model_info.get("supported_aspect_ratios", ["1:1"])

    if not supported_ratios:
        log.warning(
            f"No aspect ratios defined for model '{model_id}', defaulting to ['1:1']"
        )
        return ["1:1"]

    return supported_ratios


def map_aspect_ratio_to_model(
    requested_ratio: str, model_id: str, tolerance: float = 0.1
) -> Dict[str, Union[str, List[str]]]:
    """
    Map a requested aspect ratio to the closest supported ratio for a specific model.

    Args:
        requested_ratio: Requested aspect ratio string (e.g., "16:9")
        model_id: Model identifier (e.g., "openai/gpt-4o")
        tolerance: Maximum allowed difference before warning

    Returns:
        Dictionary containing:
        - 'mapped_ratio': Closest supported aspect ratio
        - 'supported_ratios': All supported ratios for the model
        - 'control_method': How the model accepts aspect ratios

    Examples:
        >>> map_aspect_ratio_to_model("16:10", "openai/gpt-4o")
        {
            'mapped_ratio': '3:2',
            'supported_ratios': ['1:1', '3:2', '2:3'],
            'control_method': 'predefined_presets'
        }
    """
    supported_ratios = get_model_aspect_ratios(model_id)
    mapped_ratio = find_closest_supported_aspect_ratio(
        requested_ratio, supported_ratios, tolerance
    )

    model_info = MODEL_ASPECT_RATIOS.get(model_id, {})
    control_method = model_info.get("control_method", "unknown")

    return {
        "mapped_ratio": mapped_ratio,
        "supported_ratios": supported_ratios,
        "control_method": control_method,
    }


def get_model_resolutions(model_id: str) -> Union[List[Tuple[int, int]], str]:
    """
    Get supported resolutions for a specific model.

    Args:
        model_id: Model identifier

    Returns:
        List of (width, height) tuples or string description
    """
    model_info = MODEL_ASPECT_RATIOS.get(model_id, {})
    return model_info.get("supported_resolutions", "not_specified")


def validate_aspect_ratio(aspect_ratio: str) -> bool:
    """
    Validate aspect ratio string format.

    Args:
        aspect_ratio: Aspect ratio string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        aspect_ratio_to_float(aspect_ratio)
        return True
    except ValueError:
        return False


def resolve_aspect_ratio_and_dimensions(
    aspect_ratio: Optional[str],
    provider_id: str,
    width: int = 1024,
    height: int = 1024,
) -> tuple[int, int, str]:
    """
    Unified function that handles aspect ratio mapping and dimension resolution.

    This replaces the scattered approach in providers with a single, consistent system
    that maps aspect ratios to provider-optimal dimensions and returns the final
    dimensions and aspect ratio to use.

    Args:
        aspect_ratio: Optional aspect ratio string (e.g., "16:9", "1:1", "4:3")
        provider_id: Provider identifier (e.g., "imagen/4.0", "openai/gpt-4o")
        width: Fallback width if no aspect_ratio provided
        height: Fallback height if no aspect_ratio provided

    Returns:
        (width, height, final_aspect_ratio) tuple where:
        - width, height: Optimal dimensions for the provider
        - final_aspect_ratio: The actual aspect ratio being used

    Logic:
        1. If aspect_ratio provided: map to closest supported + get provider dimensions
        2. If no aspect_ratio: use provided width/height, find closest supported ratio
        3. Log the mapping process for debugging
        4. Handle fallbacks gracefully for unsupported providers

    Examples:
        >>> resolve_aspect_ratio_and_dimensions("16:9", "imagen/4.0")
        (1408, 768, "16:9")
        >>> resolve_aspect_ratio_and_dimensions(None, "openai/gpt-4o", 1024, 512)
        (1536, 1024, "3:2")  # Closest to 2:1 is 3:2
    """
    model_info = MODEL_ASPECT_RATIOS.get(provider_id, {})
    aspect_ratio_dimensions = model_info.get("aspect_ratio_dimensions", {})

    # Case 1: Aspect ratio provided - map it to provider's optimal dimensions
    if aspect_ratio is not None:
        # First try to map the requested ratio to a supported one
        mapping_result = map_aspect_ratio_to_model(aspect_ratio, provider_id)
        mapped_ratio = mapping_result["mapped_ratio"]

        # Try to get provider-specific dimensions for the mapped ratio
        if aspect_ratio_dimensions and mapped_ratio in aspect_ratio_dimensions:
            final_width, final_height = aspect_ratio_dimensions[mapped_ratio]
            log.debug(
                f"Provider {provider_id}: Mapped aspect_ratio '{aspect_ratio}' -> '{mapped_ratio}' -> dimensions {final_width}x{final_height}"
            )
            return final_width, final_height, mapped_ratio

        # Fallback to generic conversion if no provider-specific dimensions
        try:
            final_width, final_height = aspect_ratio_to_dimensions(
                mapped_ratio, target_size=max(width, height)
            )
            log.warning(
                f"Provider {provider_id}: No specific dimensions for '{mapped_ratio}', using generic conversion -> {final_width}x{final_height}"
            )
            return final_width, final_height, mapped_ratio
        except ValueError as e:
            log.error(
                f"Provider {provider_id}: Failed to resolve aspect_ratio '{aspect_ratio}': {e}"
            )
            log.warning(
                f"Provider {provider_id}: Falling back to provided dimensions {width}x{height}"
            )
            # Fall through to Case 2 logic

    # Case 2: No aspect ratio provided - use width/height and find closest ratio
    if width <= 0 or height <= 0:
        log.warning(
            f"Provider {provider_id}: Invalid dimensions {width}x{height}, using default 1024x1024"
        )
        width, height = 1024, 1024

    # Find the closest aspect ratio for the given dimensions
    closest_ratio = find_closest_aspect_ratio(width, height)

    # Check if we have provider-specific dimensions for this ratio
    if aspect_ratio_dimensions and closest_ratio in aspect_ratio_dimensions:
        final_width, final_height = aspect_ratio_dimensions[closest_ratio]
        log.debug(
            f"Provider {provider_id}: Mapped dimensions {width}x{height} -> ratio '{closest_ratio}' -> optimal dimensions {final_width}x{final_height}"
        )
        return final_width, final_height, closest_ratio

    # If no provider-specific mapping, try to find the closest supported ratio
    supported_ratios = model_info.get("supported_aspect_ratios", [])
    if supported_ratios:
        mapped_ratio = find_closest_supported_aspect_ratio(
            closest_ratio, supported_ratios
        )
        if aspect_ratio_dimensions and mapped_ratio in aspect_ratio_dimensions:
            final_width, final_height = aspect_ratio_dimensions[mapped_ratio]
            log.debug(
                f"Provider {provider_id}: Mapped {width}x{height} -> '{closest_ratio}' -> supported '{mapped_ratio}' -> {final_width}x{final_height}"
            )
            return final_width, final_height, mapped_ratio

    # Final fallback: use the provided dimensions
    log.debug(
        f"Provider {provider_id}: Using provided dimensions {width}x{height} with closest ratio '{closest_ratio}'"
    )
    return width, height, closest_ratio
