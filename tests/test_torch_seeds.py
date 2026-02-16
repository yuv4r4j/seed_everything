"""Tests for PyTorch seeding."""

import pytest

# Try to import torch, skip tests if not available
torch = pytest.importorskip("torch", minversion="1.7")

from seed_everything.torch_seeds import seed_torch


def test_seed_torch_basic():
    """Test that seed_torch sets the torch random seed."""
    seed_torch(42, warn=False)
    
    # Generate some random tensors
    val1 = torch.rand(1).item()
    val2 = torch.randint(0, 1000, (1,)).item()
    
    # Re-seed and verify we get the same sequence
    seed_torch(42, warn=False)
    assert val1 == torch.rand(1).item()
    assert val2 == torch.randint(0, 1000, (1,)).item()


def test_seed_torch_different_seeds():
    """Test that different seeds produce different sequences."""
    seed_torch(42, warn=False)
    val1 = torch.rand(1).item()
    
    seed_torch(123, warn=False)
    val2 = torch.rand(1).item()
    
    assert val1 != val2


def test_seed_torch_deterministic_mode():
    """Test that deterministic mode is set correctly."""
    seed_torch(42, deterministic=True, warn=False)
    
    if hasattr(torch.backends, 'cudnn'):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


def test_seed_torch_benchmark_mode():
    """Test that benchmark mode can be enabled."""
    seed_torch(42, deterministic=False, benchmark=True, warn=False)
    
    if hasattr(torch.backends, 'cudnn'):
        assert torch.backends.cudnn.benchmark is True


def test_seed_torch_invalid_seed():
    """Test that invalid seeds raise appropriate errors."""
    with pytest.raises(ValueError):
        seed_torch(-1, warn=False)
    
    with pytest.raises(TypeError):
        seed_torch(42.0, warn=False)


def test_seed_torch_cuda_seeding():
    """Test that CUDA seeding works when CUDA is available."""
    seed_torch(42, warn=False)
    
    if torch.cuda.is_available():
        # Generate CUDA random numbers
        val1 = torch.cuda.FloatTensor(1).normal_().item()
        
        seed_torch(42, warn=False)
        val2 = torch.cuda.FloatTensor(1).normal_().item()
        
        assert val1 == val2


def test_seed_torch_reproducibility():
    """Test full reproducibility of torch operations."""
    seed_torch(12345, warn=False)
    
    # Generate various tensors
    tensor1 = torch.randn(5, 5)
    tensor2 = torch.randint(0, 100, (10,))
    
    # Re-seed and verify exact reproduction
    seed_torch(12345, warn=False)
    torch.testing.assert_close(tensor1, torch.randn(5, 5))
    torch.testing.assert_close(tensor2, torch.randint(0, 100, (10,)))
