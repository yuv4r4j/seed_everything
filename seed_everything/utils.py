"""Utility functions for seed validation and management."""

import logging
from typing import Dict, Any

# Configure logger for the package
logger = logging.getLogger(__name__)


def validate_seed(seed: int) -> None:
    """
    Validate that the seed is a non-negative integer within the valid range.
    
    Args:
        seed: The seed value to validate
        
    Raises:
        TypeError: If seed is not an integer
        ValueError: If seed is negative or out of valid range
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")
    
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    
    # Maximum seed value for most systems (2^32 - 1)
    max_seed = 2**32 - 1
    if seed > max_seed:
        raise ValueError(f"Seed must be <= {max_seed}, got {seed}")


def get_seed_info() -> Dict[str, Any]:
    """
    Get information about the current seed state for all detected frameworks.
    
    Returns:
        Dictionary containing seed state information for available frameworks
    """
    info = {
        "python_available": True,
        "numpy_available": False,
        "torch_available": False,
        "tensorflow_available": False,
        "jax_available": False,
    }
    
    try:
        import numpy as np
        info["numpy_available"] = True
        info["numpy_version"] = np.__version__
    except ImportError:
        pass
    
    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        info["tensorflow_available"] = True
        info["tensorflow_version"] = tf.__version__
    except ImportError:
        pass
    
    try:
        import jax
        info["jax_available"] = True
        info["jax_version"] = jax.__version__
    except ImportError:
        pass
    
    return info


def log_seeding(framework: str, seed: int, warn: bool = True) -> None:
    """
    Log seeding operation for a framework.
    
    Args:
        framework: Name of the framework being seeded
        seed: The seed value used
        warn: Whether to emit warnings about potential non-deterministic operations
    """
    logger.info(f"Seeded {framework} with seed={seed}")
    
    if warn and framework == "torch":
        logger.warning(
            "PyTorch seeding configured. Note that some operations may still be non-deterministic. "
            "See https://pytorch.org/docs/stable/notes/randomness.html for details."
        )
    elif warn and framework == "tensorflow":
        logger.warning(
            "TensorFlow seeding configured. Some operations may still be non-deterministic. "
            "See https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism"
        )
