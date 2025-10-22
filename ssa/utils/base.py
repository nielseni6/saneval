"""
Core utility functions for SANEval.

This module provides foundational utilities including:
- Random string generation and hashing
- Temporary directory and file path management
- Configuration flattening
- Unique identifier generation

Note: This module has no internal SSA dependencies to avoid circular imports.
"""

import argparse
import base64
import os
import random
import string
import time
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Optional

import pytz

# Import centralized config for default output paths
from ssa.config import DEFAULT_OUTPUT_BASE as _DEFAULT_OUTPUT_BASE_PATH

# Do not add any SSA Dependencies here!!

alphanumeric_characters = string.ascii_letters + string.digits

# Chosen b/c odds of collision in 100mm samples is 0.000155%
DEFAULT_RANDOM_CHARS = 12

# Default base directory for debug outputs when output_dir is not configured
# Creates a timestamped run directory under the centralized base path
DEFAULT_OUTPUT_BASE = (
    _DEFAULT_OUTPUT_BASE_PATH / f"run-{time.strftime('%Y%m%d_%H%M%S')}"
)

EST = pytz.timezone("America/New_York")


ssa_base = Path(__file__).parent.parent
repo_base = ssa_base.parent


def short_randomstr(num: int = DEFAULT_RANDOM_CHARS) -> str:
    return "".join(random.choice(alphanumeric_characters) for _ in range(num))


def unique_id(name: str = "run", random_chars: int = DEFAULT_RANDOM_CHARS) -> str:
    now = datetime.now(pytz.UTC).astimezone(EST)
    return f"{now.strftime('%Y-%m-%d-%H%M')}-{name}-{short_randomstr(num=random_chars)}"


def hash_str_to_alphanumeric(text: str, chars: int = 22) -> str:
    # Generate SHA256 hash (256 bits)
    hash_bytes = sha256(text.encode()).digest()
    # Encode in base64, remove non-alphanumeric, and take first n chars
    return (
        base64.b64encode(hash_bytes).decode().replace("+", "").replace("/", "")[:chars]
    )


# Private module-level variables for lazy initialization
_base_tempdir: Optional[Path] = None
_run_tempdir: Optional[Path] = None


def get_base_tempdir() -> Path:
    """
    Get the base temporary directory, creating it if needed.

    Uses lazy initialization to avoid creating directories at import time.
    The directory can be customized via the SSA_TEMP_DIR environment variable.

    Returns:
        Path to the base temporary directory (default: /tmp/ssa/)
    """
    global _base_tempdir
    if _base_tempdir is None:
        _base_tempdir = Path(os.getenv("SSA_TEMP_DIR", "/tmp/ssa"))
        _base_tempdir.mkdir(parents=True, exist_ok=True)
    return _base_tempdir


def get_run_tempdir() -> Path:
    """
    Get the run-specific temporary directory, creating it if needed.

    Uses lazy initialization to avoid creating directories at import time.
    Each call to reset_run_dir() will update this directory.

    Returns:
        Path to the run-specific temporary directory
    """
    global _run_tempdir
    if _run_tempdir is None:
        base_dir = get_base_tempdir()
        _run_tempdir = base_dir / unique_id()
        _run_tempdir.mkdir(parents=True, exist_ok=True)
    return _run_tempdir


def prepare_artifact_path(name_prefix="temp", suffix="", run_id=0):
    """
    Generates a unique, non-existent file path within /tmp/ssa/{run_id}/,
    intended for final output files such as run_result.json and aggregates.json.
    """
    base_dir = get_base_tempdir()

    # Making a unique directory for run_result.json and aggregates.json
    unique_dir = base_dir / run_id
    os.makedirs(unique_dir, exist_ok=True)

    # Return Path to that directory and file
    filename = f"{name_prefix}{suffix}"
    artifact_path = Path(unique_dir) / filename
    assert not artifact_path.exists()
    return artifact_path


def create_temp_download_directory(
    experiment_name="", run_id=0, random_chars=DEFAULT_RANDOM_CHARS
):
    """
    Creates a unique temporary directory for a specific run, organized by experiment.
    /tmp/ssa/{experiment_name}/{run_id}
    """
    base_dir = get_base_tempdir()
    random_str = f"-{short_randomstr(num=random_chars)}" if random_chars else ""
    unique_dir = base_dir / experiment_name / random_str / run_id
    os.makedirs(unique_dir)
    temp_dir_path = Path(unique_dir)
    return temp_dir_path


def reset_run_dir(name=None):
    """
    Reset the run-specific temporary directory.

    Args:
        name: Optional name for the run directory. If not provided, generates a unique ID.
    """
    global _run_tempdir
    base_dir = get_base_tempdir()
    run_dir_name = name or unique_id()
    _run_tempdir = base_dir / run_dir_name
    _run_tempdir.mkdir(parents=True, exist_ok=True)


def get_temp_file(name_prefix="temp", suffix="", random_chars=DEFAULT_RANDOM_CHARS):
    """Get a tempfile inside the current run_tempdir"""
    run_dir = get_run_tempdir()
    random_str = f"-{short_randomstr(num=random_chars)}" if random_chars else ""
    filename = f"{name_prefix}{random_str}{suffix}"
    temp_file_path = Path(run_dir) / filename
    assert not temp_file_path.exists()
    return temp_file_path


def flatten_cfg(value):
    """Flattens enums into strings for configs"""
    return value.value if hasattr(value, "value") else value


def str2bool(v):
    """Useful for argparse to be able to exactly specify boolean args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
