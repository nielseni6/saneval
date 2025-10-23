"""
Setup script for SSA (Standardized Scoring Architecture)
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ssa",
    version="0.1.0",
    description="Standardized Scoring Architecture for image evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SANEval Contributors",
    author_email="saneval@example.com",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=[
        "benchmark",
        "image-generation",
        "evaluation",
        "vlm",
        "image-composition",
        "scoring",
    ],
    entry_points={
        "console_scripts": [
            "ssa-benchmark=ssa.benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
