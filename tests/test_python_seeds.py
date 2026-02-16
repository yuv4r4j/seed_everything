"""Tests for Python standard library seeding."""

import os
import random
import pytest
from seed_everything.python_seeds import seed_python


def test_seed_python_basic():
    """Test that seed_python sets the random seed."""
    seed_python(42, warn=False)
    
    # Generate some random numbers
    val1 = random.random()
    val2 = random.randint(0, 1000)
    
    # Re-seed and verify we get the same sequence
    seed_python(42, warn=False)
    assert val1 == random.random()
    assert val2 == random.randint(0, 1000)


def test_seed_python_different_seeds():
    """Test that different seeds produce different sequences."""
    seed_python(42, warn=False)
    val1 = random.random()
    
    seed_python(123, warn=False)
    val2 = random.random()
    
    assert val1 != val2


def test_seed_python_sets_pythonhashseed():
    """Test that seed_python sets PYTHONHASHSEED."""
    seed_python(42, warn=False)
    assert os.environ.get('PYTHONHASHSEED') == '42'
    
    seed_python(123, warn=False)
    assert os.environ.get('PYTHONHASHSEED') == '123'


def test_seed_python_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_python(-1, warn=False)
    
    with pytest.raises(TypeError):
        seed_python(42.0, warn=False)


def test_seed_python_reproducibility():
    """Test full reproducibility of Python random operations."""
    seed_python(12345, warn=False)
    
    # Generate various random values
    randoms = [random.random() for _ in range(10)]
    integers = [random.randint(0, 100) for _ in range(10)]
    choices = [random.choice(['a', 'b', 'c']) for _ in range(10)]
    
    # Re-seed and verify exact reproduction
    seed_python(12345, warn=False)
    assert randoms == [random.random() for _ in range(10)]
    assert integers == [random.randint(0, 100) for _ in range(10)]
    assert choices == [random.choice(['a', 'b', 'c']) for _ in range(10)]
