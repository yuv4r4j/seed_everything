"""Distributed training seeding utilities."""

import os
from typing import Optional, Callable, Any

from .utils import validate_seed, log_seeding

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_rank() -> Optional[int]:
    """
    Get the rank of the current process in distributed training.
    
    Attempts to detect rank from various distributed training frameworks:
    - PyTorch Distributed (torch.distributed)
    - Horovod
    - DeepSpeed
    - Environment variables (RANK, LOCAL_RANK, SLURM_PROCID)
    
    Returns:
        The rank as an integer, or None if not in distributed mode
    """
    # Try PyTorch distributed
    if TORCH_AVAILABLE:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except (ImportError, RuntimeError):
            pass
    
    # Try Horovod
    try:
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank()
    except (ImportError, ValueError):
        pass
    
    # Try DeepSpeed
    try:
        import deepspeed
        if hasattr(deepspeed, 'comm') and deepspeed.comm.is_initialized():
            return deepspeed.comm.get_rank()
    except (ImportError, AttributeError):
        pass
    
    # Try environment variables
    for env_var in ['RANK', 'LOCAL_RANK', 'SLURM_PROCID', 'PMI_RANK']:
        if env_var in os.environ:
            try:
                return int(os.environ[env_var])
            except ValueError:
                pass
    
    return None


def seed_distributed(
    seed: int,
    rank: Optional[int] = None,
    warn: bool = True
) -> int:
    """
    Seed for distributed training with rank-aware seeding.
    
    Each rank/worker gets a deterministic but different seed (base_seed + rank)
    to ensure different data ordering per worker while maintaining reproducibility.
    
    Args:
        seed: The base seed value (non-negative integer)
        rank: The rank of the current process. If None, attempts to auto-detect.
        warn: Whether to emit warnings (default: True)
        
    Returns:
        The actual seed used (base_seed + rank)
    """
    validate_seed(seed)
    
    # Auto-detect rank if not provided
    if rank is None:
        rank = get_rank()
        if rank is None:
            rank = 0  # Default to rank 0 if not in distributed mode
    
    # Compute rank-specific seed
    rank_seed = seed + rank
    
    # Ensure rank_seed doesn't overflow
    max_seed = 2**32 - 1
    if rank_seed > max_seed:
        rank_seed = rank_seed % max_seed
    
    log_seeding(f"Distributed (rank={rank})", rank_seed, warn)
    
    # Configure NCCL for reproducibility
    if 'NCCL_DEBUG' not in os.environ:
        os.environ['NCCL_DEBUG'] = 'WARN'
    
    return rank_seed


def worker_init_fn(worker_id: int, base_seed: int = 42) -> Callable[[int], None]:
    """
    Create a worker initialization function for PyTorch DataLoader.
    
    This ensures each DataLoader worker has a deterministic but different seed.
    Use this with torch.utils.data.DataLoader(worker_init_fn=...).
    
    Args:
        worker_id: The worker ID (automatically provided by DataLoader)
        base_seed: The base seed value (default: 42)
        
    Returns:
        A function that can be used as worker_init_fn for DataLoader
    """
    def _init_fn(wid: int) -> None:
        """Initialize worker with deterministic seed."""
        import random
        import numpy as np
        
        # Compute worker-specific seed
        worker_seed = base_seed + wid
        
        # Seed Python
        random.seed(worker_seed)
        
        # Seed NumPy
        try:
            np.random.seed(worker_seed)
        except:
            pass
        
        # Seed PyTorch
        if TORCH_AVAILABLE:
            import torch
            torch.manual_seed(worker_seed)
    
    return _init_fn


def get_worker_init_fn(base_seed: int = 42) -> Callable[[int], None]:
    """
    Get a worker initialization function for PyTorch DataLoader.
    
    This is a convenience function that returns a worker_init_fn that can be
    directly passed to torch.utils.data.DataLoader.
    
    Example:
        >>> loader = DataLoader(dataset, worker_init_fn=get_worker_init_fn(42))
    
    Args:
        base_seed: The base seed value (default: 42)
        
    Returns:
        A function that can be used as worker_init_fn for DataLoader
    """
    def _init_fn(worker_id: int) -> None:
        """Initialize worker with deterministic seed."""
        import random
        
        # Compute worker-specific seed
        worker_seed = base_seed + worker_id
        
        # Seed Python
        random.seed(worker_seed)
        
        # Seed NumPy
        try:
            import numpy as np
            np.random.seed(worker_seed)
        except ImportError:
            pass
        
        # Seed PyTorch
        if TORCH_AVAILABLE:
            import torch
            torch.manual_seed(worker_seed)
    
    return _init_fn
