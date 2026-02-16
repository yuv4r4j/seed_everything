"""Tests for core seed_everything function."""

import pytest
from seed_everything.core import seed_everything


def test_seed_everything_basic():
    """Test basic seed_everything functionality."""
    result = seed_everything(42, warn=False)
    
    assert isinstance(result, dict)
    assert result['seed'] == 42
    assert result['python'] is True


def test_seed_everything_deterministic():
    """Test that deterministic flag is recorded."""
    result = seed_everything(42, deterministic=True, warn=False)
    assert result['deterministic'] is True
    
    result = seed_everything(42, deterministic=False, warn=False)
    assert result['deterministic'] is False


def test_seed_everything_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_everything(-1, warn=False)
    
    with pytest.raises(TypeError):
        seed_everything(42.0, warn=False)


def test_seed_everything_numpy():
    """Test numpy seeding in seed_everything."""
    result = seed_everything(42, warn=False)
    
    try:
        import numpy
        assert result['numpy'] is True
        assert 'numpy_rng' in result
    except ImportError:
        assert result['numpy'] is False


def test_seed_everything_torch():
    """Test torch seeding in seed_everything."""
    result = seed_everything(42, warn=False)
    
    try:
        import torch
        assert result['torch'] is True
        assert 'torch_cuda' in result
    except ImportError:
        assert result['torch'] is False


def test_seed_everything_tensorflow():
    """Test tensorflow seeding in seed_everything."""
    result = seed_everything(42, warn=False)
    
    try:
        import tensorflow
        assert result['tensorflow'] is True
    except ImportError:
        assert result['tensorflow'] is False


def test_seed_everything_jax():
    """Test JAX seeding in seed_everything."""
    result = seed_everything(42, warn=False)
    
    try:
        import jax
        assert result['jax'] is True
        assert 'jax_key' in result
    except ImportError:
        assert result['jax'] is False


def test_seed_everything_reproducibility():
    """Test that seed_everything produces reproducible results."""
    import random
    
    # First run
    seed_everything(12345, warn=False)
    val1 = random.random()
    
    # Second run with same seed
    seed_everything(12345, warn=False)
    val2 = random.random()
    
    assert val1 == val2


def test_seed_everything_different_seeds():
    """Test that different seeds produce different results."""
    import random
    
    seed_everything(42, warn=False)
    val1 = random.random()
    
    seed_everything(123, warn=False)
    val2 = random.random()
    
    assert val1 != val2


def test_seed_everything_default_seed():
    """Test that default seed is 42."""
    result = seed_everything(warn=False)
    assert result['seed'] == 42


def test_seed_everything_all_frameworks():
    """Test seeding all available frameworks."""
    result = seed_everything(42, warn=False)
    
    # Python should always be seeded
    assert result['python'] is True
    
    # Check that result contains keys for all frameworks
    assert 'numpy' in result
    assert 'torch' in result
    assert 'tensorflow' in result
    assert 'jax' in result
