"""AWS utilities for Bedrock integration.

This module provides simplified AWS credential management and client caching
for SANEval's Bedrock integration. It supports multiple authentication methods:
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- AWS profiles (AWS_PROFILE)
- IAM roles (container environments)
"""

import os
from typing import Optional, Tuple

import boto3
import botocore
from botocore.exceptions import ClientError

from ssa.utils.logging import get_log

log = get_log(__file__)

DEFAULT_REGION = "us-east-1"

# Track clients we've created to avoid recreating them
_boto_clients = {}

# Track credentials we've validated to avoid repeated checks
_credentials_validated = {}


def is_container_environment() -> bool:
    """Check if running in a container environment (ECS, Lambda, etc.)."""
    # ECS task metadata endpoint is available in ECS containers
    if os.getenv("ECS_CONTAINER_METADATA_URI"):
        return True
    # AWS Lambda sets AWS_EXECUTION_ENV
    if os.getenv("AWS_EXECUTION_ENV"):
        return True
    # AWS_CONTAINER_CREDENTIALS_RELATIVE_URI is set in ECS/Lambda
    if os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
        return True
    return False


def validate_credentials(session: boto3.Session) -> Tuple[bool, str]:
    """
    Validate AWS credentials by attempting to get caller identity.

    Args:
        session: Boto3 session to validate

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        arn = identity.get("Arn", "")

        if not arn.startswith("arn:aws"):
            return False, f"Invalid ARN format: {arn}"

        log.debug(f"AWS credentials validated: {arn}")
        return True, f"AWS authenticated as: {arn}"

    except botocore.exceptions.NoCredentialsError:
        return False, "No AWS credentials found"
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "InvalidClientTokenId":
            return False, "Invalid security token"
        elif error_code == "InvalidAccessKeyId":
            return False, "Invalid access key"
        return False, f"Client error: {error_code}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def setup_aws(
    profile: Optional[str] = None, force_refresh: bool = False
) -> Optional[str]:
    """
    Set up AWS credentials for Bedrock access.

    This function configures AWS credentials using one of three methods:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. AWS profiles (AWS_PROFILE or explicitly passed profile name)
    3. IAM roles (automatically used in container environments)

    Args:
        profile: Optional AWS profile name to use. If None, uses environment
                variables or default profile detection
        force_refresh: If True, skip validation cache and revalidate credentials

    Returns:
        The profile name used, or None if using environment variables/IAM roles

    Raises:
        Exception: If credentials cannot be configured or validated
    """
    global _credentials_validated

    # Set default region if not already set
    if "AWS_DEFAULT_REGION" not in os.environ:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        log.debug(f"Set AWS_DEFAULT_REGION to {DEFAULT_REGION}")

    # Check if running in container environment (ECS, Lambda, etc.)
    if is_container_environment():
        cache_key = "CONTAINER_IAM_ROLE"
        if not force_refresh and _credentials_validated.get(cache_key):
            log.debug("Using cached container IAM role credentials")
            return None

        # Container environments use IAM role credentials automatically
        session = boto3.Session()
        valid, message = validate_credentials(session)
        if valid:
            log.info(message)
            _credentials_validated[cache_key] = True
            return None
        else:
            raise Exception(f"Container IAM role credentials invalid: {message}")

    # Check for direct AWS credentials (CI/automated environments)
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        cache_key = "ENV_CREDENTIALS"
        if not force_refresh and _credentials_validated.get(cache_key):
            log.debug("Using cached environment variable credentials")
            return None

        # Clear any profile settings that might interfere
        original_profile = os.environ.get("AWS_PROFILE")
        if original_profile:
            del os.environ["AWS_PROFILE"]

        try:
            session = boto3.Session()
            valid, message = validate_credentials(session)
            if valid:
                log.info(message)
                _credentials_validated[cache_key] = True
                return None
            else:
                raise Exception(f"Environment variable credentials invalid: {message}")
        finally:
            # Restore original profile setting
            if original_profile:
                os.environ["AWS_PROFILE"] = original_profile

    # Profile-based authentication (local development)
    if profile is None:
        profile = os.environ.get("AWS_PROFILE")

    if profile:
        cache_key = f"PROFILE_{profile}"
        if not force_refresh and _credentials_validated.get(cache_key):
            log.debug(f"Using cached profile credentials: {profile}")
            os.environ["AWS_PROFILE"] = profile
            return profile

        try:
            session = boto3.Session(profile_name=profile)
            valid, message = validate_credentials(session)
            if valid:
                log.info(message)
                _credentials_validated[cache_key] = True
                os.environ["AWS_PROFILE"] = profile
                return profile
            else:
                raise Exception(f"Profile '{profile}' credentials invalid: {message}")
        except botocore.exceptions.ProfileNotFound:
            raise Exception(
                f"AWS profile '{profile}' not found. "
                f"Please configure it with 'aws configure --profile {profile}' "
                f"or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )

    # No credentials configured
    raise Exception(
        "No AWS credentials configured. Please either:\n"
        "1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, or\n"
        "2. Set AWS_PROFILE environment variable to your profile name, or\n"
        "3. Configure default credentials with 'aws configure'"
    )


def boto_client(name: str, region: str = DEFAULT_REGION, config=None):
    """
    Create or retrieve a cached boto3 client.

    Args:
        name: AWS service name (e.g., 'bedrock-runtime', 's3', 'sts')
        region: AWS region (default: us-east-1)
        config: Optional botocore.config.Config object for client configuration

    Returns:
        Boto3 client for the specified service

    Note:
        Clients are cached unless a custom config is provided
    """
    global _boto_clients

    # Only cache clients without custom config to avoid conflicts
    use_cache = config is None
    cache_key = f"{name}-{region}"

    if use_cache and cache_key in _boto_clients:
        log.debug(f"Using cached boto client: {cache_key}")
        return _boto_clients[cache_key]

    # Ensure AWS is set up
    profile = setup_aws()

    # Create session with or without profile
    if profile:
        session = boto3.Session(profile_name=profile)
    else:
        session = boto3.Session()

    # Create client with optional config
    if config:
        client = session.client(name, region_name=region, config=config)
    else:
        client = session.client(name, region_name=region)

    # Test the client to ensure it works before caching
    try:
        if name == "sts":
            client.get_caller_identity()
        elif name == "s3":
            client.list_buckets(MaxKeys=1)
        # For other services (like bedrock-runtime), assume they work if setup succeeded
        # since testing them might incur costs or require specific permissions
    except Exception as e:
        log.error(f"Failed to validate {name} client: {e}")
        raise

    # Cache the client if no custom config
    if use_cache:
        _boto_clients[cache_key] = client
        log.debug(f"Cached boto client: {cache_key}")

    return client
