"""Setup configuration for seed_everything package."""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies (minimal)
install_requires = []

# Optional dependencies for different frameworks
extras_require = {
    "numpy": ["numpy>=1.17.0"],
    "torch": ["torch>=1.7.0"],
    "tensorflow": ["tensorflow>=2.0.0"],
    "jax": ["jax>=0.2.0", "jaxlib>=0.1.0"],
    "sklearn": ["scikit-learn>=0.24.0", "numpy>=1.17.0"],
    "distributed": ["torch>=1.7.0"],  # For torch.distributed
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0",
        "flake8>=3.9.0",
        "mypy>=0.900",
    ],
}

# "all" includes all framework dependencies (excluding dev)
extras_require["all"] = list(set(
    dep
    for key, deps in extras_require.items()
    if key not in ["dev"]
    for dep in deps
))

setup(
    name="seed_everything",
    version="0.1.0",
    author="seed_everything contributors",
    description="Comprehensive seeding for reproducible ML training across all major frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuv4r4j/seed_everything",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    keywords="seed random reproducibility machine-learning deep-learning pytorch tensorflow jax",
)
