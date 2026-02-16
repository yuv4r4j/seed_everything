"""PyTorch seeding functions."""

import os
from typing import Optional

from .utils import validate_seed, log_seeding

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def seed_torch(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    warn: bool = True
) -> None:
    """
    Seed PyTorch for CPU and CUDA devices with optional deterministic configuration.
    
    Args:
        seed: The seed value (non-negative integer)
        deterministic: If True, configure PyTorch for deterministic operations (default: True)
        benchmark: If True, enable cuDNN benchmark mode for performance (default: False)
                   Note: benchmark mode may introduce non-determinism
        warn: Whether to emit warnings (default: True)
    """
    if not TORCH_AVAILABLE:
        return
    
    validate_seed(seed)
    
    # Seed PyTorch CPU
    torch.manual_seed(seed)
    
    # Seed PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Configure cuDNN for determinism
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    
    # Enable deterministic algorithms (PyTorch >= 1.8)
    if deterministic and hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            # Some operations don't support deterministic mode
            if warn:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Could not enable torch.use_deterministic_algorithms: {e}. "
                    "Some operations may still be non-deterministic."
                )
    
    log_seeding("PyTorch", seed, warn)
