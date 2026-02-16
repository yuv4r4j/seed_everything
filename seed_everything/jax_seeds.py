"""JAX seeding functions."""

from typing import Optional, Any

from .utils import validate_seed, log_seeding

try:
    import jax
    import jax.random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def seed_jax(seed: int, warn: bool = True) -> Optional[Any]:
    """
    Create a JAX PRNG key from the given seed.
    
    JAX uses a different random number generation approach than other frameworks.
    Instead of global state, it uses explicit PRNG keys that must be passed around.
    
    Args:
        seed: The seed value (non-negative integer)
        warn: Whether to emit warnings (default: True)
        
    Returns:
        A JAX PRNG key, or None if JAX is not available
    """
    if not JAX_AVAILABLE:
        return None
    
    validate_seed(seed)
    
    # Create a PRNG key from the seed
    key = jax.random.PRNGKey(seed)
    
    log_seeding("JAX", seed, warn)
    
    return key
