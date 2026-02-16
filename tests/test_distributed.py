"""Tests for distributed training utilities."""

import os
import pytest
from seed_everything.distributed import (
    get_rank,
    seed_distributed,
    get_worker_init_fn,
)


def test_get_rank_no_distributed():
    """Test get_rank returns None when not in distributed mode."""
    # Clear all distributed environment variables
    for var in ['RANK', 'LOCAL_RANK', 'SLURM_PROCID', 'PMI_RANK']:
        os.environ.pop(var, None)
    
    rank = get_rank()
    # Should return None or 0 (implementation detail)
    assert rank is None or rank == 0


def test_get_rank_from_env_var():
    """Test get_rank reads from environment variables."""
    os.environ['RANK'] = '3'
    rank = get_rank()
    assert rank == 3
    
    os.environ.pop('RANK')


def test_get_rank_local_rank():
    """Test get_rank reads from LOCAL_RANK."""
    os.environ['LOCAL_RANK'] = '2'
    rank = get_rank()
    assert rank == 2
    
    os.environ.pop('LOCAL_RANK')


def test_seed_distributed_basic():
    """Test basic distributed seeding."""
    seed = seed_distributed(42, rank=0, warn=False)
    assert seed == 42
    
    seed = seed_distributed(42, rank=1, warn=False)
    assert seed == 43
    
    seed = seed_distributed(42, rank=5, warn=False)
    assert seed == 47


def test_seed_distributed_auto_rank():
    """Test distributed seeding with auto-detected rank."""
    os.environ['RANK'] = '2'
    seed = seed_distributed(42, warn=False)
    assert seed == 44
    
    os.environ.pop('RANK')


def test_seed_distributed_overflow():
    """Test seed overflow handling."""
    # Test with a very large seed that would overflow
    max_seed = 2**32 - 1
    seed = seed_distributed(max_seed, rank=10, warn=False)
    # Should wrap around
    assert seed < max_seed


def test_seed_distributed_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_distributed(-1, rank=0, warn=False)
    
    with pytest.raises(TypeError):
        seed_distributed(42.0, rank=0, warn=False)


def test_get_worker_init_fn():
    """Test worker_init_fn creation."""
    worker_fn = get_worker_init_fn(base_seed=42)
    
    assert callable(worker_fn)
    
    # Call the function (simulating DataLoader worker initialization)
    worker_fn(0)  # Should not raise
    worker_fn(1)  # Should not raise


def test_worker_init_fn_seeds_correctly():
    """Test that worker_init_fn actually seeds the RNG."""
    import random
    
    worker_fn = get_worker_init_fn(base_seed=42)
    
    # Initialize worker 0
    worker_fn(0)
    val1 = random.random()
    
    # Initialize worker 0 again
    worker_fn(0)
    val2 = random.random()
    
    # Should get the same value
    assert val1 == val2


def test_worker_init_fn_different_workers():
    """Test that different workers get different seeds."""
    import random
    
    worker_fn = get_worker_init_fn(base_seed=42)
    
    # Initialize worker 0
    worker_fn(0)
    val1 = random.random()
    
    # Initialize worker 1
    worker_fn(1)
    val2 = random.random()
    
    # Should get different values
    assert val1 != val2


def test_distributed_sets_nccl_debug():
    """Test that seed_distributed sets NCCL_DEBUG."""
    os.environ.pop('NCCL_DEBUG', None)
    
    seed_distributed(42, rank=0, warn=False)
    
    assert 'NCCL_DEBUG' in os.environ
    assert os.environ['NCCL_DEBUG'] == 'WARN'
