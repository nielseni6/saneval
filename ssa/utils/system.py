import datetime
import getpass
import hashlib
import json
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

# from ssa.utils.git import git_info
from ssa.utils.logging import get_log

log = get_log(__file__)


def get_user():
    if env_user := os.getenv("SSA_USER"):
        return env_user
    user = getpass.getuser()

    # This isn't great but better than nothing
    if user == "sagemaker-user":
        user += f"-{os.getenv('SAGEMAKER_SPACE_NAME')}"
    return user


def hash_dict(dictionary):
    dict_str = json.dumps(dictionary, sort_keys=True)
    hasher = hashlib.new("sha256")
    hasher.update(dict_str.encode("utf-8"))
    return hasher.hexdigest()


def default_context():
    """Return default context information for benchmark runs."""
    now = datetime.datetime.now()
    context_info = {
        "user": get_user(),
        "hostname": socket.gethostname(),
        "datetime": now.astimezone().isoformat(),
        "sys_argv": sys.argv,
    }

    # Add AWS Info if available
    if ecs_agent := os.getenv("ECS_AGENT_URI"):
        context_info["ecs_task_id"] = ecs_agent.split("/")[-1].split("-")[0]
    for k in ["AWS_REGION", "AWS_EXECUTION_ENV"]:
        if k in os.environ:
            context_info[k.lower()] = os.environ[k]

    return context_info


def get_next_seeds(initial_seed, n):
    """Generate n subsequent deterministic seeds from initial seed."""
    seeds = []
    next_seed = initial_seed
    for _ in range(n):
        # Option 1: Simple LCG
        next_seed = (next_seed * 1103515245 + 12345) & 0x7FFFFFFF

        # Option 2: Alternative using hash
        # next_seed = hash(str(next_seed)) & 0x7fffffff

        seeds.append(next_seed)
    return seeds


def cmd(command, echocmd=False, noisy=False, use_logger=True, check_code=True, cwd="."):
    """Run a shell command and return its output.

    Args:
        command (str): Command to execute
        noisy (bool): If True, prints output to stdout in realtime
        log (bool): If true, uses log

    Returns:
        str: Command output

    Raises:
        subprocess.CalledProcessError: If command returns non-zero exit status
    """

    def _writeout(txt):
        if use_logger:
            log.info(txt.rstrip())
        else:
            sys.stdout.write(txt)
            sys.stdout.flush()

    if echocmd:
        _writeout(f"\n{command}\n")
    output = []
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        if noisy:
            _writeout(line)
        output.append(line)

    return_code = process.wait()
    if check_code and return_code != 0:
        if not noisy:
            log.error("Command failed, printing recent output...")
            for line in output[-50:]:
                _writeout(line)
        raise subprocess.CalledProcessError(return_code, command)

    return "".join(output)


def open_url(url):
    """Util to always open links in a specific chrome profile"""
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_path):
        webbrowser.open(url)
        return

    profile = os.getenv("SSA_CHROME_PROFILE", "Default")
    try:
        cmd(f'"{chrome_path}" --profile-directory=\'{profile}\' "{url}"')
    except Exception as e:
        print(f"Error opening Chrome: {e}")
        webbrowser.open(url)


def parse_config_arg(obj):
    """parse an object as a dictionary. Accepts 'key-val', '{"key": "val"}', or '/path/to/obj.json'"""
    if not obj:
        return {}

    # Try as simple key=value
    if "=" in obj:
        key, value = obj.split("=", 1)
        return {key.strip(): value.strip()}

    # Try as JSON string
    if obj.startswith("{"):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format: {obj}")

    # Try as file path
    path = Path(obj)
    if path.exists():
        return json.loads(path.read_text())

    raise ValueError(f"Invalid obj format: {obj}")


def get_subconfigs(configs, key):
    keydot = f"{key}."
    return {
        k.replace(keydot, ""): v for k, v in configs.items() if k.startswith(keydot)
    }


def apply_overrides(base_config, overrides):
    if not overrides:
        return base_config

    # Create a temporary config object to handle type conversion
    config_class = base_config.__class__
    temp_config = config_class(**overrides)

    # Then filter back to just specified fields
    type_fixed_overrides = {
        k: v for k, v in temp_config.model_dump().items() if k in overrides
    }
    if len(type_fixed_overrides) < len(overrides):
        dropped = overrides.keys() - type_fixed_overrides.keys()
        raise Exception(
            f"{config_class.__name__} Unrecognized config overrides: {dropped}"
        )

    updated_config = base_config.model_copy(update=type_fixed_overrides)
    log.debug(
        f"{config_class.__name__} Applied config overrides {type_fixed_overrides}\nresult={updated_config}"
    )
    return updated_config
