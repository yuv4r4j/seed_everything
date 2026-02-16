"""Tests for JAX seeding."""

import pytest

# Try to import jax, skip tests if not available
jax = pytest.importorskip("jax", minversion="0.2")

from seed_everything.jax_seeds import seed_jax


def test_seed_jax_basic():
    """Test that seed_jax returns a PRNG key."""
    key = seed_jax(42, warn=False)
    
    assert key is not None
    # JAX keys are typically arrays
    assert hasattr(key, 'shape')


def test_seed_jax_different_seeds():
    """Test that different seeds produce different keys."""
    key1 = seed_jax(42, warn=False)
    key2 = seed_jax(123, warn=False)
    
    # Keys should be different
    assert not jax.numpy.array_equal(key1, key2)


def test_seed_jax_same_seed():
    """Test that same seed produces same key."""
    key1 = seed_jax(42, warn=False)
    key2 = seed_jax(42, warn=False)
    
    # Keys should be identical
    jax.numpy.testing.assert_array_equal(key1, key2)


def test_seed_jax_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_jax(-1, warn=False)
    
    with pytest.raises(TypeError):
        seed_jax(42.0, warn=False)


def test_seed_jax_random_operations():
    """Test reproducibility with JAX random operations."""
    key1 = seed_jax(12345, warn=False)
    
    # Split the key for multiple operations
    key1, subkey1 = jax.random.split(key1)
    val1 = jax.random.normal(subkey1, shape=(5,))
    
    # Re-seed and verify exact reproduction
    key2 = seed_jax(12345, warn=False)
    key2, subkey2 = jax.random.split(key2)
    val2 = jax.random.normal(subkey2, shape=(5,))
    
    jax.numpy.testing.assert_array_equal(val1, val2)


def test_seed_jax_key_splitting():
    """Test that key splitting is deterministic."""
    key = seed_jax(42, warn=False)
    
    # Split the key
    key1a, key1b = jax.random.split(key, 2)
    
    # Re-seed and split again
    key = seed_jax(42, warn=False)
    key2a, key2b = jax.random.split(key, 2)
    
    # Split keys should match
    jax.numpy.testing.assert_array_equal(key1a, key2a)
    jax.numpy.testing.assert_array_equal(key1b, key2b)
