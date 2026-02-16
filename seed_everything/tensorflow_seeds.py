"""TensorFlow and Keras seeding functions."""

import os
from typing import Optional

from .utils import validate_seed, log_seeding

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def seed_tensorflow(seed: int, deterministic: bool = True, warn: bool = True) -> None:
    """
    Seed TensorFlow and configure for deterministic operations.
    
    Args:
        seed: The seed value (non-negative integer)
        deterministic: If True, configure TensorFlow for deterministic operations (default: True)
        warn: Whether to emit warnings (default: True)
    """
    if not TENSORFLOW_AVAILABLE:
        return
    
    validate_seed(seed)
    
    # Seed TensorFlow
    tf.random.set_seed(seed)
    
    # Configure for deterministic operations
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # For TensorFlow 2.x, also try to enable op determinism
        if hasattr(tf.config.experimental, 'enable_op_determinism'):
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception as e:
                if warn:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Could not enable TensorFlow op determinism: {e}")
    
    log_seeding("TensorFlow", seed, warn)
