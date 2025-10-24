"""
Security utilities for path validation and input sanitization.

This module provides security functions to prevent path traversal attacks
and ensure safe file system access through whitelist-based validation.
"""

import os
from pathlib import Path
from typing import List, Optional, Set, Union

from ssa.utils.logging import get_log

log = get_log(__file__)

# Maximum allowed path length (Linux/Unix standard)
MAX_PATH_LENGTH = 4096

# Get the project root directory (parent of ssa package)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def get_default_allowed_dirs(purpose: str = "read") -> Set[Path]:
    """
    Get default allowed directories based on the access purpose.

    Args:
        purpose: The access purpose - "corpus", "input", "output", or "config"

    Returns:
        Set of allowed base directories as resolved Path objects
    """
    # Common allowed directories
    common = {
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "prompts",
    }

    if purpose == "corpus":
        # For corpus files, only allow data directories
        return common

    elif purpose == "input" or purpose == "read":
        # For input files (images, etc.), allow images directory and working dir
        return common | {
            PROJECT_ROOT / "images",
            PROJECT_ROOT,  # Allow project root for relative paths
        }

    elif purpose == "output" or purpose == "write":
        # For output, allow results directory and temporary directories
        # Also allow any path the user explicitly specifies (validated for safety)
        return common | {
            PROJECT_ROOT / "results",
            PROJECT_ROOT / "output",
            PROJECT_ROOT,  # Allow project root
            Path.cwd(),  # Allow current working directory
            Path("/tmp"),  # Allow temp directory  # nosec B108
        }

    elif purpose == "config":
        # For config files, allow data and project root
        return common | {PROJECT_ROOT}

    else:
        # Default: most restrictive
        return common


def canonicalize_path(path: Union[str, Path]) -> Path:
    """
    Canonicalize a path by resolving symlinks and relative references.

    This converts the path to an absolute path and resolves all symbolic links
    and relative path components (. and ..).

    Args:
        path: Path to canonicalize

    Returns:
        Resolved absolute Path object

    Raises:
        ValueError: If path is invalid or empty
    """
    if not path:
        raise ValueError("Path cannot be empty")

    try:
        path_obj = Path(path)
        # Resolve to absolute path and follow symlinks
        resolved = path_obj.resolve(strict=False)
        return resolved
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot canonicalize path '{path}': {e}") from e


def is_path_safe(path: Union[str, Path], max_length: int = MAX_PATH_LENGTH) -> bool:
    """
    Check if a path is safe (basic safety checks).

    This performs basic validation without checking against whitelists:
    - Path is not empty
    - Path length is within limits
    - Path doesn't contain null bytes
    - Path can be canonicalized

    Args:
        path: Path to validate
        max_length: Maximum allowed path length

    Returns:
        True if path passes basic safety checks
    """
    if not path:
        return False

    path_str = str(path)

    # Check for null bytes (potential security issue)
    if "\x00" in path_str:
        log.warning(f"Path contains null bytes: {path_str[:50]}")
        return False

    # Check length
    if len(path_str) > max_length:
        log.warning(f"Path exceeds maximum length ({max_length}): {len(path_str)}")
        return False

    # Try to canonicalize (this will fail for invalid paths)
    try:
        canonicalize_path(path)
        return True
    except ValueError:
        return False


def is_within_directory(path: Path, directory: Path) -> bool:
    """
    Check if a path is within a directory (after canonicalization).

    This uses path resolution to detect path traversal attempts.

    Args:
        path: The path to check (will be canonicalized)
        directory: The base directory (will be canonicalized)

    Returns:
        True if path is within directory
    """
    try:
        # Canonicalize both paths
        canon_path = canonicalize_path(path)
        canon_dir = canonicalize_path(directory)

        # Check if path is within directory
        # Using relative_to() will raise ValueError if path is not relative to directory
        try:
            canon_path.relative_to(canon_dir)
            return True
        except ValueError:
            return False

    except ValueError:
        return False


def validate_path(
    path: Union[str, Path],
    allowed_base_dirs: Optional[List[Union[str, Path]]] = None,
    purpose: str = "read",
    must_exist: bool = False,
    allow_create: bool = False,
) -> Path:
    """
    Validate a path against security rules and whitelist.

    This is the main security validation function. It:
    1. Performs basic safety checks
    2. Canonicalizes the path
    3. Validates against whitelist of allowed directories
    4. Optionally checks existence or creation permissions

    Args:
        path: Path to validate
        allowed_base_dirs: List of allowed base directories. If None, uses defaults based on purpose.
        purpose: Purpose of access - "corpus", "input", "output", "config", "read", "write"
        must_exist: If True, path must exist
        allow_create: If True, allow non-existent paths that can be created

    Returns:
        Canonicalized Path object if validation succeeds

    Raises:
        ValueError: If validation fails with details about the violation
        FileNotFoundError: If must_exist=True and path doesn't exist
        PermissionError: If path creation is needed but not allowed

    Example:
        >>> # Validate corpus file
        >>> path = validate_path("data/prompts/test.yaml", purpose="corpus")
        >>> # Validate output directory (can be created)
        >>> path = validate_path("results/run1", purpose="output", allow_create=True)
    """
    # Basic safety checks
    if not is_path_safe(path):
        raise ValueError(f"Path failed basic safety checks: {path}")

    # Canonicalize path
    try:
        canon_path = canonicalize_path(path)
    except ValueError as e:
        raise ValueError(f"Cannot validate path '{path}': {e}") from e

    # Get allowed directories
    if allowed_base_dirs is None:
        allowed_dirs = get_default_allowed_dirs(purpose)
    else:
        allowed_dirs = {canonicalize_path(d) for d in allowed_base_dirs}

    # Check if path is within any allowed directory
    is_allowed = False
    for allowed_dir in allowed_dirs:
        if is_within_directory(canon_path, allowed_dir):
            is_allowed = True
            break

    if not is_allowed:
        # Build helpful error message
        allowed_dirs_str = "\n  - ".join(str(d) for d in sorted(allowed_dirs))
        raise ValueError(
            f"Path '{path}' is not within allowed directories.\n"
            f"Resolved to: {canon_path}\n"
            f"Allowed directories for '{purpose}':\n  - {allowed_dirs_str}"
        )

    # Check existence requirements
    if must_exist and not canon_path.exists():
        raise FileNotFoundError(f"Path does not exist: {canon_path}")

    # Check creation permissions
    if allow_create and not canon_path.exists():
        # Check if parent directory exists and is writable
        parent = canon_path.parent
        if not parent.exists():
            if allow_create:
                # Check if we can create the parent directories
                try:
                    # Don't actually create, just check if parent's parent exists
                    if not parent.parent.exists():
                        raise PermissionError(
                            f"Cannot create path '{canon_path}': parent directory tree doesn't exist"
                        )
                except OSError as e:
                    raise PermissionError(
                        f"Cannot validate creation of '{canon_path}': {e}"
                    ) from e
        elif not os.access(parent, os.W_OK):
            raise PermissionError(
                f"Cannot create path '{canon_path}': parent directory is not writable"
            )

    log.debug(f"Path validated successfully: {canon_path} (purpose={purpose})")
    return canon_path


def validate_directory(
    path: Union[str, Path],
    purpose: str = "read",
    must_exist: bool = False,
    allow_create: bool = False,
) -> Path:
    """
    Validate a directory path with security checks.

    This is a convenience wrapper around validate_path() that also ensures
    the path is a directory (not a file).

    Args:
        path: Directory path to validate
        purpose: Purpose of access
        must_exist: If True, directory must exist
        allow_create: If True, allow non-existent directories

    Returns:
        Canonicalized directory Path object

    Raises:
        ValueError: If validation fails
        NotADirectoryError: If path exists but is not a directory
    """
    canon_path = validate_path(
        path, purpose=purpose, must_exist=must_exist, allow_create=allow_create
    )

    # If it exists, verify it's a directory
    if canon_path.exists() and not canon_path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {canon_path}")

    return canon_path


def validate_file(
    path: Union[str, Path],
    purpose: str = "read",
    must_exist: bool = True,
    allowed_extensions: Optional[Set[str]] = None,
) -> Path:
    """
    Validate a file path with security checks.

    This is a convenience wrapper around validate_path() that also ensures
    the path is a file and optionally validates the extension.

    Args:
        path: File path to validate
        purpose: Purpose of access
        must_exist: If True, file must exist
        allowed_extensions: Set of allowed file extensions (e.g., {'.json', '.yaml'})

    Returns:
        Canonicalized file Path object

    Raises:
        ValueError: If validation fails or extension not allowed
        IsADirectoryError: If path exists but is a directory
    """
    canon_path = validate_path(path, purpose=purpose, must_exist=must_exist)

    # If it exists, verify it's a file
    if canon_path.exists() and not canon_path.is_file():
        raise IsADirectoryError(f"Path exists but is not a file: {canon_path}")

    # Check extension if specified
    if allowed_extensions is not None:
        ext = canon_path.suffix.lower()
        if ext not in allowed_extensions:
            allowed_ext_str = ", ".join(sorted(allowed_extensions))
            raise ValueError(
                f"File extension '{ext}' not allowed. Allowed extensions: {allowed_ext_str}"
            )

    return canon_path


class SecurityError(Exception):
    """Exception raised for security violations."""

    pass
