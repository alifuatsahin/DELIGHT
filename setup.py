"""Setup script for DELIGHT package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="delight-3d",
    version="0.1.0",
    author="DELIGHT Authors",
    author_email="",
    description="Deep Compression Latent Diffusion for Generation of High-quality Three-dimensional shapes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alifuatsahin/DELIGHT",
    packages=find_packages(exclude=["tests", "scripts", "third_party"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "delight-train=train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/*.yml"],
    },
    keywords="3d-generation, point-cloud, diffusion-models, vae, deep-learning, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/alifuatsahin/DELIGHT/issues",
        "Source": "https://github.com/alifuatsahin/DELIGHT",
    },
)
