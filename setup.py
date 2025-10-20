"""
Setup script for SSA (Standardized Scoring Architecture)
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys
import subprocess

# Define virtual environment name
VENV_NAME = "saneval_env"
VENV_PATH = Path(__file__).parent / VENV_NAME

def create_venv_if_needed():
    """Create virtual environment if it doesn't exist and we're not already in one."""
    # Check if we're already in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print(f"✓ Already running in a virtual environment: {sys.prefix}")
        return

    # Check if saneval_env exists
    if VENV_PATH.exists():
        print(f"✓ Virtual environment '{VENV_NAME}' already exists at {VENV_PATH}")
        print(f"\n⚠️  To use it, activate with:")
        print(f"   source {VENV_NAME}/bin/activate  (Unix/Mac)")
        print(f"   {VENV_NAME}\\Scripts\\activate  (Windows)")
    else:
        print(f"Creating virtual environment '{VENV_NAME}'...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_PATH)],
                check=True
            )
            print(f"✓ Virtual environment '{VENV_NAME}' created successfully!")
            print(f"\n⚠️  Next steps:")
            print(f"   1. Activate the environment:")
            print(f"      source {VENV_NAME}/bin/activate  (Unix/Mac)")
            print(f"      {VENV_NAME}\\Scripts\\activate  (Windows)")
            print(f"   2. Run: pip install -e .")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create virtual environment: {e}")
            sys.exit(1)

# Create venv before setup
create_venv_if_needed()

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="ssa",
    version="0.1.0",
    description="Standardized Scoring Architecture for image evaluation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'ssa-benchmark=ssa.benchmark:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
