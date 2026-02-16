"""
Core seed_everything function - the main entry point for seeding all frameworks.
"""

from typing import Optional, Dict, Any

from .utils import validate_seed, log_seeding
from .python_seeds import seed_python
from .numpy_seeds import seed_numpy
from .torch_seeds import seed_torch
from .tensorflow_seeds import seed_tensorflow
from .jax_seeds import seed_jax


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
    warn: bool = True
) -> Dict[str, Any]:
    """
    Seed all available ML frameworks and Python's random module.
    
    This is the main entry point for the seed_everything package. It seeds:
    - Python's built-in random module and PYTHONHASHSEED
    - NumPy (if available)
    - PyTorch CPU and CUDA (if available)
    - TensorFlow (if available)
    - JAX (if available)
    
    Each framework seeding gracefully handles ImportError, so only installed
    frameworks will be seeded.
    
    Args:
        seed: The seed value (non-negative integer, default: 42)
        deterministic: If True, configure frameworks for deterministic operations (default: True)
                       This affects PyTorch (cudnn.deterministic, use_deterministic_algorithms)
                       and TensorFlow (TF_DETERMINISTIC_OPS)
        warn: Whether to emit warnings about non-deterministic operations (default: True)
        
    Returns:
        Dictionary with seeding results and JAX PRNG key (if JAX is available)
        
    Raises:
        TypeError: If seed is not an integer
        ValueError: If seed is negative or out of valid range
        
    Example:
        >>> import seed_everything
        >>> seed_everything.seed_everything(42)
        >>> # Now all frameworks use seed 42
        
        >>> # For JAX, you'll get a PRNG key back
        >>> result = seed_everything.seed_everything(42)
        >>> jax_key = result.get('jax_key')
    """
    validate_seed(seed)
    
    result = {
        'seed': seed,
        'deterministic': deterministic,
    }
    
    # Seed Python standard library
    seed_python(seed, warn=warn)
    result['python'] = True
    
    # Seed NumPy
    numpy_rng = seed_numpy(seed, warn=warn)
    result['numpy'] = numpy_rng is not None
    if numpy_rng is not None:
        result['numpy_rng'] = numpy_rng
    
    # Seed PyTorch
    seed_torch(seed, deterministic=deterministic, benchmark=False, warn=warn)
    # Check if torch was actually seeded
    try:
        import torch
        result['torch'] = True
        result['torch_cuda'] = torch.cuda.is_available()
    except ImportError:
        result['torch'] = False
    
    # Seed TensorFlow
    seed_tensorflow(seed, deterministic=deterministic, warn=warn)
    # Check if tensorflow was actually seeded
    try:
        import tensorflow
        result['tensorflow'] = True
    except ImportError:
        result['tensorflow'] = False
    
    # Seed JAX
    jax_key = seed_jax(seed, warn=warn)
    result['jax'] = jax_key is not None
    if jax_key is not None:
        result['jax_key'] = jax_key
    
    return result
