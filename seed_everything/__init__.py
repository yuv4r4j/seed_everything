"""
seed_everything: Comprehensive seeding for reproducible ML training.

This package provides utilities to set seeds for reproducible machine learning
model training across all major ML frameworks and distributed training packages.
"""

__version__ = "0.1.0"

from .core import seed_everything
from .python_seeds import seed_python
from .numpy_seeds import seed_numpy
from .torch_seeds import seed_torch
from .tensorflow_seeds import seed_tensorflow
from .jax_seeds import seed_jax
from .sklearn_seeds import get_sklearn_random_state, seed_sklearn_estimator
from .distributed import seed_distributed, get_worker_init_fn, get_rank
from .utils import validate_seed, get_seed_info

__all__ = [
    # Main API
    'seed_everything',
    
    # Individual framework seeders
    'seed_python',
    'seed_numpy',
    'seed_torch',
    'seed_tensorflow',
    'seed_jax',
    'get_sklearn_random_state',
    'seed_sklearn_estimator',
    
    # Distributed training
    'seed_distributed',
    'get_worker_init_fn',
    'get_rank',
    
    # Utilities
    'validate_seed',
    'get_seed_info',
    
    # Version
    '__version__',
]
