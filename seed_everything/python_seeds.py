"""Python standard library seeding functions."""

import os
import random
from typing import Optional

from .utils import validate_seed, log_seeding


def seed_python(seed: int, warn: bool = True) -> None:
    """
    Seed Python's built-in random module and set PYTHONHASHSEED.
    
    Args:
        seed: The seed value (non-negative integer)
        warn: Whether to emit warnings (default: True)
    """
    validate_seed(seed)
    
    # Seed the random module
    random.seed(seed)
    
    # Set PYTHONHASHSEED environment variable
    # Note: This must be set before Python starts to be fully effective,
    # but we set it here for subprocess and documentation purposes
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    log_seeding("Python", seed, warn)
