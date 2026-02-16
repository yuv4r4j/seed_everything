"""Tests for scikit-learn seeding utilities."""

import pytest

# Try to import numpy, skip tests if not available
numpy = pytest.importorskip("numpy", minversion="1.17")

from seed_everything.sklearn_seeds import get_sklearn_random_state, seed_sklearn_estimator


def test_get_sklearn_random_state_basic():
    """Test that get_sklearn_random_state returns a RandomState."""
    rs = get_sklearn_random_state(42, warn=False)
    
    assert rs is not None
    assert isinstance(rs, numpy.random.RandomState)


def test_get_sklearn_random_state_reproducibility():
    """Test that RandomState produces reproducible results."""
    rs1 = get_sklearn_random_state(42, warn=False)
    vals1 = rs1.rand(10)
    
    rs2 = get_sklearn_random_state(42, warn=False)
    vals2 = rs2.rand(10)
    
    numpy.testing.assert_array_equal(vals1, vals2)


def test_get_sklearn_random_state_different_seeds():
    """Test that different seeds produce different results."""
    rs1 = get_sklearn_random_state(42, warn=False)
    vals1 = rs1.rand(10)
    
    rs2 = get_sklearn_random_state(123, warn=False)
    vals2 = rs2.rand(10)
    
    assert not numpy.array_equal(vals1, vals2)


def test_get_sklearn_random_state_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        get_sklearn_random_state(-1, warn=False)
    
    with pytest.raises(TypeError):
        get_sklearn_random_state(42.0, warn=False)


def test_seed_sklearn_estimator_with_random_state():
    """Test seeding an estimator that has random_state attribute."""
    # Create a mock estimator
    class MockEstimator:
        def __init__(self):
            self.random_state = None
    
    estimator = MockEstimator()
    seed_sklearn_estimator(estimator, 42)
    
    assert estimator.random_state == 42


def test_seed_sklearn_estimator_with_set_params():
    """Test seeding an estimator that has set_params method."""
    # Create a mock estimator
    class MockEstimator:
        def __init__(self):
            self._random_state = None
        
        def set_params(self, **params):
            if 'random_state' in params:
                self._random_state = params['random_state']
            return self
        
        @property
        def random_state(self):
            return self._random_state
    
    estimator = MockEstimator()
    seed_sklearn_estimator(estimator, 42)
    
    assert estimator.random_state == 42


def test_seed_sklearn_estimator_without_random_state():
    """Test seeding an estimator without random_state (should not error)."""
    # Create a mock estimator without random_state
    class MockEstimator:
        def __init__(self):
            self.param = None
    
    estimator = MockEstimator()
    # Should not raise an error
    result = seed_sklearn_estimator(estimator, 42)
    
    assert result is estimator


def test_seed_sklearn_estimator_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    class MockEstimator:
        def __init__(self):
            self.random_state = None
    
    estimator = MockEstimator()
    
    with pytest.raises(ValueError):
        seed_sklearn_estimator(estimator, -1)
    
    with pytest.raises(TypeError):
        seed_sklearn_estimator(estimator, 42.0)
