import argparse
import base64
import os
import random
import string
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytz

# Do not add any SSA Dependencies here!!

alphanumeric_characters = string.ascii_letters + string.digits

# Chosen b/c odds of collision in 100mm samples is 0.000155%
DEFAULT_RANDOM_CHARS = 12

EST = pytz.timezone("America/New_York")


ssa_base = Path(__file__).parent.parent
repo_base = ssa_base.parent


def short_randomstr(num=DEFAULT_RANDOM_CHARS):
    return "".join(random.choice(alphanumeric_characters) for _ in range(num))


def unique_id(name="run", random_chars=DEFAULT_RANDOM_CHARS):
    now = datetime.now(pytz.UTC).astimezone(EST)
    return f"{now.strftime('%Y-%m-%d-%H%M')}-{name}-{short_randomstr(num=random_chars)}"


def hash_str_to_alphanumeric(text: str, chars=22) -> str:
    # Generate SHA256 hash (256 bits)
    hash_bytes = sha256(text.encode()).digest()
    # Encode in base64, remove non-alphanumeric, and take first n chars
    return (
        base64.b64encode(hash_bytes).decode().replace("+", "").replace("/", "")[:chars]
    )


base_tempdir = Path("/tmp/ssa/")
run_tempdir = base_tempdir / unique_id()
os.makedirs(run_tempdir, exist_ok=True)


def prepare_artifact_path(name_prefix="temp", suffix="", run_id=0):
    """
    Generates a unique, non-existent file path within /tmp/ssa/{run_id}/,
    intended for final output files such as run_result.json and aggregates.json.
    """
    global base_tempdir

    # Making a unique directory for run_result.json and aggregates.json
    unique_dir = base_tempdir / run_id
    os.makedirs(unique_dir, exist_ok=True)

    # Return Path to that directory and file
    filename = f"{name_prefix}{suffix}"
    res = Path(unique_dir) / filename
    assert not res.exists()
    return res


def create_temp_download_directory(
    experiment_name="", run_id=0, random_chars=DEFAULT_RANDOM_CHARS
):
    """
    Creates a unique temporary directory for a specific run, organized by experiment.
    /tmp/ssa/{experiment_name}/{run_id}
    """
    global base_tempdir
    random_str = f"-{short_randomstr(num=random_chars)}" if random_chars else ""
    unique_dir = base_tempdir / experiment_name / random_str / run_id
    os.makedirs(unique_dir)
    res = Path(unique_dir)
    return res


def reset_run_dir(name=None):
    global run_tempdir
    run_dir_name = name or unique_id()
    run_tempdir = base_tempdir / run_dir_name
    os.makedirs(run_tempdir, exist_ok=True)


def get_temp_file(name_prefix="temp", suffix="", random_chars=DEFAULT_RANDOM_CHARS):
    """Get a tempfile inside the current run_tempdir"""
    global run_tempdir
    random_str = f"-{short_randomstr(num=random_chars)}" if random_chars else ""
    filename = f"{name_prefix}{random_str}{suffix}"
    res = Path(run_tempdir) / filename
    assert not res.exists()
    return res


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
