"""NumPy seeding functions."""

from typing import Optional

from .utils import validate_seed, log_seeding

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def seed_numpy(seed: int, warn: bool = True) -> Optional['np.random.Generator']:
    """
    Seed NumPy's random number generator.
    
    This function seeds both the legacy numpy.random.seed() for backwards compatibility
    and creates a modern numpy.random.Generator with PCG64 for new code.
    
    Args:
        seed: The seed value (non-negative integer)
        warn: Whether to emit warnings (default: True)
        
    Returns:
        A numpy.random.Generator instance with PCG64, or None if NumPy is not available
    """
    if not NUMPY_AVAILABLE:
        return None
    
    validate_seed(seed)
    
    # Seed legacy global RNG
    np.random.seed(seed)
    
    # Create and return a modern Generator instance
    rng = np.random.Generator(np.random.PCG64(seed))
    
    log_seeding("NumPy", seed, warn)
    
    return rng
