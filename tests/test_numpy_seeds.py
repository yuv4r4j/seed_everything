"""Tests for NumPy seeding."""

import pytest

# Try to import numpy, skip tests if not available
numpy = pytest.importorskip("numpy", minversion="1.17")

from seed_everything.numpy_seeds import seed_numpy


def test_seed_numpy_basic():
    """Test that seed_numpy sets the numpy random seed."""
    seed_numpy(42, warn=False)
    
    # Generate some random numbers
    val1 = numpy.random.rand()
    val2 = numpy.random.randint(0, 1000)
    
    # Re-seed and verify we get the same sequence
    seed_numpy(42, warn=False)
    assert val1 == numpy.random.rand()
    assert val2 == numpy.random.randint(0, 1000)


def test_seed_numpy_different_seeds():
    """Test that different seeds produce different sequences."""
    seed_numpy(42, warn=False)
    val1 = numpy.random.rand()
    
    seed_numpy(123, warn=False)
    val2 = numpy.random.rand()
    
    assert val1 != val2


def test_seed_numpy_returns_generator():
    """Test that seed_numpy returns a Generator instance."""
    rng = seed_numpy(42, warn=False)
    
    assert rng is not None
    assert isinstance(rng, numpy.random.Generator)


def test_seed_numpy_generator_reproducibility():
    """Test that the returned Generator is reproducible."""
    rng1 = seed_numpy(42, warn=False)
    vals1 = rng1.random(10)
    
    rng2 = seed_numpy(42, warn=False)
    vals2 = rng2.random(10)
    
    numpy.testing.assert_array_equal(vals1, vals2)


def test_seed_numpy_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_numpy(-1, warn=False)
    
    with pytest.raises(TypeError):
        seed_numpy(42.0, warn=False)


def test_seed_numpy_array_operations():
    """Test reproducibility with various numpy operations."""
    seed_numpy(12345, warn=False)
    
    # Generate various arrays
    arr1 = numpy.random.randn(5, 5)
    arr2 = numpy.random.randint(0, 100, size=10)
    arr3 = numpy.random.choice([1, 2, 3, 4, 5], size=20)
    
    # Re-seed and verify exact reproduction
    seed_numpy(12345, warn=False)
    numpy.testing.assert_array_equal(arr1, numpy.random.randn(5, 5))
    numpy.testing.assert_array_equal(arr2, numpy.random.randint(0, 100, size=10))
    numpy.testing.assert_array_equal(arr3, numpy.random.choice([1, 2, 3, 4, 5], size=20))
