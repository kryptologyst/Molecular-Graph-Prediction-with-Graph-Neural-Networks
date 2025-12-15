#!/usr/bin/env python3
"""Simple test script to verify the installation and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.utils import set_seed, get_device
        from src.models.gnn_models import GCN, GIN, MPNN
        from src.data.datasets import Tox21Dataset
        from src.train.trainer import Trainer
        from src.eval.metrics import compute_metrics
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test that models can be created."""
    try:
        from src.models.gnn_models import GCN, GIN, MPNN
        from omegaconf import DictConfig
        
        # Test GCN
        config = DictConfig({
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1,
            'use_batch_norm': True,
            'use_residual': False,
            'num_tasks': 12
        })
        
        gcn = GCN(config)
        gin = GIN(config)
        mpnn = MPNN(config)
        
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_device_detection():
    """Test device detection."""
    try:
        from src.utils import get_device
        device = get_device("auto")
        print(f"✓ Device detection successful: {device}")
        return True
    except Exception as e:
        print(f"✗ Device detection error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running basic functionality tests...")
    print("-" * 40)
    
    tests = [
        test_imports,
        test_model_creation,
        test_device_detection
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed! The installation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
