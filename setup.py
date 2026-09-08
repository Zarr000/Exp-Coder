"""
Exp-Coder - Original Large Language Model

Setup configuration for package installation.
Author: Zarr
Studio: Exp Works
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

# Core implementation lives in src/ (kept importable as `src.*` for legacy
# compatibility); `exp_coder` is the canonical public facade package.
packages = find_packages(where="src") + ["exp_coder"]

setup(
    name="exp-coder",
    version="0.1.0",
    author="Zarr",
    description="An original decoder-only language model built from scratch for software development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zarr000/Exp-Coder",
    packages=packages,
    package_dir={"": "src", "exp_coder": "exp_coder"},
    package_data={"": ["*.yaml", "*.yml"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
)