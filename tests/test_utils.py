"""Tests for utility functions."""

import pytest
from seed_everything.utils import validate_seed, get_seed_info


def test_validate_seed_valid():
    """Test that valid seeds pass validation."""
    validate_seed(0)
    validate_seed(42)
    validate_seed(12345)
    validate_seed(2**32 - 1)


def test_validate_seed_negative():
    """Test that negative seeds raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_seed(-1)
    
    with pytest.raises(ValueError, match="non-negative"):
        validate_seed(-100)


def test_validate_seed_too_large():
    """Test that seeds larger than 2^32-1 raise ValueError."""
    with pytest.raises(ValueError, match="<="):
        validate_seed(2**32)
    
    with pytest.raises(ValueError, match="<="):
        validate_seed(2**33)


def test_validate_seed_not_integer():
    """Test that non-integer seeds raise TypeError."""
    with pytest.raises(TypeError, match="integer"):
        validate_seed(42.0)
    
    with pytest.raises(TypeError, match="integer"):
        validate_seed("42")
    
    with pytest.raises(TypeError, match="integer"):
        validate_seed(None)


def test_get_seed_info():
    """Test that get_seed_info returns framework availability."""
    info = get_seed_info()
    
    assert isinstance(info, dict)
    assert info["python_available"] is True
    assert "numpy_available" in info
    assert "torch_available" in info
    assert "tensorflow_available" in info
    assert "jax_available" in info


def test_get_seed_info_numpy():
    """Test numpy information in seed info."""
    info = get_seed_info()
    
    try:
        import numpy
        assert info["numpy_available"] is True
        assert "numpy_version" in info
    except ImportError:
        assert info["numpy_available"] is False


def test_get_seed_info_torch():
    """Test torch information in seed info."""
    info = get_seed_info()
    
    try:
        import torch
        assert info["torch_available"] is True
        assert "torch_version" in info
        assert "cuda_available" in info
    except ImportError:
        assert info["torch_available"] is False
