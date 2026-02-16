"""Tests for TensorFlow seeding."""

import os
import pytest

# Try to import tensorflow, skip tests if not available
tf = pytest.importorskip("tensorflow", minversion="2.0")

from seed_everything.tensorflow_seeds import seed_tensorflow


def test_seed_tensorflow_basic():
    """Test that seed_tensorflow sets the tensorflow random seed."""
    seed_tensorflow(42, warn=False)
    
    # Generate some random tensors
    val1 = tf.random.uniform([1]).numpy()[0]
    
    # Re-seed and verify we get the same sequence
    seed_tensorflow(42, warn=False)
    val2 = tf.random.uniform([1]).numpy()[0]
    
    assert val1 == val2


def test_seed_tensorflow_different_seeds():
    """Test that different seeds produce different sequences."""
    seed_tensorflow(42, warn=False)
    val1 = tf.random.uniform([1]).numpy()[0]
    
    seed_tensorflow(123, warn=False)
    val2 = tf.random.uniform([1]).numpy()[0]
    
    assert val1 != val2


def test_seed_tensorflow_deterministic_env_vars():
    """Test that deterministic mode sets environment variables."""
    seed_tensorflow(42, deterministic=True, warn=False)
    
    assert os.environ.get('TF_DETERMINISTIC_OPS') == '1'
    assert os.environ.get('TF_CUDNN_DETERMINISTIC') == '1'


def test_seed_tensorflow_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_tensorflow(-1, warn=False)
    
    with pytest.raises(TypeError):
        seed_tensorflow(42.0, warn=False)


def test_seed_tensorflow_reproducibility():
    """Test full reproducibility of tensorflow operations."""
    seed_tensorflow(12345, warn=False)
    
    # Generate various tensors
    tensor1 = tf.random.normal([5, 5])
    tensor2 = tf.random.uniform([10], minval=0, maxval=100, dtype=tf.int32)
    
    # Re-seed and verify exact reproduction
    seed_tensorflow(12345, warn=False)
    tensor1_repeat = tf.random.normal([5, 5])
    tensor2_repeat = tf.random.uniform([10], minval=0, maxval=100, dtype=tf.int32)
    
    tf.debugging.assert_equal(tensor1, tensor1_repeat)
    tf.debugging.assert_equal(tensor2, tensor2_repeat)


def test_seed_tensorflow_without_deterministic():
    """Test seeding without deterministic mode."""
    # Clear env vars
    os.environ.pop('TF_DETERMINISTIC_OPS', None)
    os.environ.pop('TF_CUDNN_DETERMINISTIC', None)
    
    seed_tensorflow(42, deterministic=False, warn=False)
    
    # Env vars should not be set (or might be set from previous tests)
    # Just verify no error occurs
    val1 = tf.random.uniform([1]).numpy()[0]
    
    seed_tensorflow(42, deterministic=False, warn=False)
    val2 = tf.random.uniform([1]).numpy()[0]
    
    assert val1 == val2
