import logging
import os
import re
import shutil
import sys

import colorlog

# Only dependency should be base
from ssa.utils.base import get_temp_file

####################################
# String Utils for logging


def heading(txt):
    return f"==== {txt} ===="


def str_to_bool(value):
    return value.lower() in ("true", "t", "yes", "y", "1")


def truncate(txt, limit=100):
    """Truncate string txt to limit chars"""
    txt = txt.replace("\n", " ").replace("\r", " ")
    if len(txt) <= limit:
        return txt
    return f"{txt[0:limit]}..."


def dict_truncate(data, limit=100):
    res = {}
    for k, v in data.items():
        if len(str(v)) > limit:
            v = str(v)[0:limit] + "..."
        res[k] = v
    return res


def format_list(lst, trunc=True):
    width = 100 if trunc else 999999999
    return "\n" + "\n".join(f" - {truncate(str(e), limit=width)}" for e in lst)


def grey_str(txt):
    return f"\033[90m{txt}\033[0m"


def strip_color_codes(text):
    if not isinstance(text, str) or text is None:
        text = ""
    return re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", text)


####################################
# Initialize logging

plain_format = "%(message)s"
verbose_format = "[%(asctime)s]-[%(name)s]-[%(levelname)s] %(message)s"


def build_formatter(plain, color):
    msg_format = plain_format if plain else verbose_format
    if color:
        return colorlog.ColoredFormatter(
            "%(log_color)s" + msg_format,
            log_colors={
                "DEBUG": "blue",
                # 'INFO': 'white',  # let info default console
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    return logging.Formatter(msg_format)


# Read initial logging config from env variables/defaults
plain = str_to_bool(os.getenv("SSA_LOGS_PLAIN", "False"))
color = str_to_bool(os.getenv("SSA_LOGS_COLOR", "True"))
level = logging.DEBUG
if verbose := os.getenv("SSA_LOGS_VERBOSE"):
    level = logging.DEBUG if str_to_bool(verbose) else logging.INFO
formatter = build_formatter(plain, color)

# Logging setup
root_name = "ssa"
log = logging.getLogger(root_name)
log.propagate = False
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(level)
ch.setFormatter(formatter)
log.addHandler(ch)


def get_log(subname):
    """Get a child logger of the root one"""
    subname = subname.split("/")[-1].replace(".py", "")
    logger_name = f"{root_name}.{subname}"
    return logging.getLogger(logger_name)


####################################
# LogFileHandler


class LogFileHandler:
    def __init__(self):
        self.fh = None
        self.path = None

    def __enter__(self):
        # Create and configure new handler
        self.path = get_temp_file("logs", ".txt")
        self.fh = logging.FileHandler(self.path)
        self.fh.setLevel(logging.DEBUG)
        formatter = build_formatter(False, False)
        self.fh.setFormatter(formatter)
        log.addHandler(self.fh)

        # exclusive lets you log to just this file
        self.exclusive = logging.getLogger(f"exclusive_{id(self)}")
        self.exclusive.addHandler(self.fh)
        self.exclusive.setLevel(logging.DEBUG)
        self.exclusive.propagate = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fh:
            log.removeHandler(self.fh)
            self.fh.close()


####################################
# Allow for dynamic overrides


def set_log_level(level):
    ch.setLevel(level)


def set_verbose(verbose: bool):
    set_log_level(logging.DEBUG if verbose else logging.INFO)


def set_log_format():
    msg_format = plain_format if plain else verbose_format
    if color:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + msg_format,
            log_colors={
                "DEBUG": "blue",
                # 'INFO': 'white',  # let info default console
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        formatter = logging.Formatter(msg_format)
    ch.setFormatter(formatter)


def rebuild_formatter():
    ch.setFormatter(plain, color)


def set_plain(val: bool = True):
    global plain
    plain = val
    rebuild_formatter()


def set_color(val: bool = True):
    global color
    color = val
    rebuild_formatter()


####################################
# Utilities for script argparsing


def add_log_args(parser):
    parser.add_argument("--verbose", action="store_true", help="Verbose debug logs")
    parser.add_argument(
        "--max-verbose", action="store_true", help="Maximum verbose debug logs"
    )
    parser.add_argument(
        "--plain-logs", action="store_true", help="Plain formatted logs"
    )


def process_log_args(args):
    if args.max_verbose:
        set_verbose(True)
        # Enable root verbose logging to diagnose low-level / third party calling issues
        logging.basicConfig(level=logging.DEBUG)
    if args.verbose:
        set_verbose(True)
    if args.plain_logs:
        set_plain()


####################################
# Utilities for flush-printing


def visible_length(text):
    return len(strip_color_codes(text))


def crjust(text, width, fillchar=" "):
    visible_len = visible_length(text)
    padding_needed = max(0, width - visible_len)
    return fillchar * padding_needed + text


def simple_table(rows, buf=2, just=crjust):
    """Formats a simple table.  based on a 2d array of strings"""
    widths = []
    num_cols = len(rows[0])
    for col in range(num_cols):
        # Ignore rows with less than a full set of columns
        widths.append(
            max(visible_length(str(row[col])) for row in rows if len(row) == num_cols)
            + buf
        )

    res = ""
    # Print headers
    for i, cell in enumerate(rows[0]):
        res += just(str(cell), widths[i])
    res += "\n"

    # Print separator line
    for i, cell in enumerate(rows[0]):
        res += just(str("-" * len(cell)), widths[i])
    res += "\n"

    # Print data rows
    for row in rows[1:]:
        for i, cell in enumerate(row):
            res += just(str(cell), widths[i])
        res += "\n"
    return res


def fill_line(text, extra_buffer=0):
    width = shutil.get_terminal_size().columns - 3 - extra_buffer
    end_color = text.endswith("\033[0m")
    res = text[:width] + ("..." if len(text) > width else "")
    if len(res) < width:
        res += " " * (width - len(res))
    if end_color:
        res += "\033[0m"
    return res


def flush_print(lines, buf=1):
    height = shutil.get_terminal_size().lines - buf
    for i in range(height - len(lines)):
        lines.append(" ")
    lines = [fill_line(line) for line in lines]
    sys.stdout.write("\r\033[K")
    sys.stdout.write("\n".join(lines[:-1]))
    sys.stdout.write("\n" + lines[-1])
    sys.stdout.write(f"\033[{len(lines)-1}A")
    sys.stdout.flush()


class SuppressLogs:
    """Context manager to temporarily suppress verbose logs during data collection."""

    def __init__(self):
        self.loggers = {}

    def __enter__(self):
        # Suppress common noisy loggers
        logger_names = [
            "ssa.s3",
            "botocore",
            "urllib3",
            "requests",
            "boto3",
        ]

        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            self.loggers[logger_name] = logger.level
            logger.setLevel(logging.ERROR)  # Only show errors

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original log levels
        for logger_name, original_level in self.loggers.items():
            logging.getLogger(logger_name).setLevel(original_level)
