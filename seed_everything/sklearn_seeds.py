"""scikit-learn seeding utilities."""

from typing import Optional, Any

from .utils import validate_seed, log_seeding

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def get_sklearn_random_state(seed: int, warn: bool = True) -> Optional['np.random.RandomState']:
    """
    Create a numpy RandomState instance for use with scikit-learn estimators.
    
    scikit-learn estimators accept a random_state parameter that can be an integer
    or a numpy.random.RandomState instance. This function creates a RandomState
    for consistent seeding across sklearn operations.
    
    Args:
        seed: The seed value (non-negative integer)
        warn: Whether to emit warnings (default: True)
        
    Returns:
        A numpy.random.RandomState instance, or None if NumPy is not available
    """
    if not NUMPY_AVAILABLE:
        return None
    
    validate_seed(seed)
    
    random_state = np.random.RandomState(seed)
    
    log_seeding("scikit-learn", seed, warn)
    
    return random_state


def seed_sklearn_estimator(estimator: Any, seed: int) -> Any:
    """
    Set the random_state of a scikit-learn estimator if it has that parameter.
    
    Args:
        estimator: A scikit-learn estimator object
        seed: The seed value (non-negative integer)
        
    Returns:
        The estimator with random_state set (returns original if no random_state parameter)
    """
    validate_seed(seed)
    
    if hasattr(estimator, 'random_state'):
        estimator.random_state = seed
    elif hasattr(estimator, 'set_params'):
        try:
            estimator.set_params(random_state=seed)
        except (ValueError, TypeError):
            # Estimator doesn't have random_state parameter
            pass
    
    return estimator
