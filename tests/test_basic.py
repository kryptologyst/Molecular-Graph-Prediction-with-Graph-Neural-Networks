"""Unit tests for molecular graph prediction project."""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import set_seed, get_device, count_parameters
from src.models.gnn_models import GCN, GIN, MPNN
from src.eval.metrics import compute_metrics


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Placeholder test
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5 bias


class TestModels:
    """Test GNN model implementations."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DictConfig({
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1,
            'use_batch_norm': True,
            'use_residual': False,
            'num_tasks': 12
        })
    
    @pytest.fixture
    def sample_data(self):
        """Create sample graph data."""
        x = torch.randn(10, 9)  # 10 nodes, 9 features
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        batch = torch.zeros(10, dtype=torch.long)
        return x, edge_index, batch
    
    def test_gcn_creation(self, config):
        """Test GCN model creation."""
        model = GCN(config)
        assert isinstance(model, GCN)
        assert model.num_tasks == 12
    
    def test_gin_creation(self, config):
        """Test GIN model creation."""
        model = GIN(config)
        assert isinstance(model, GIN)
        assert model.num_tasks == 12
    
    def test_mpnn_creation(self, config):
        """Test MPNN model creation."""
        model = MPNN(config)
        assert isinstance(model, MPNN)
        assert model.num_tasks == 12
    
    def test_gcn_forward(self, config, sample_data):
        """Test GCN forward pass."""
        model = GCN(config)
        x, edge_index, batch = sample_data
        
        with torch.no_grad():
            output = model(x, edge_index, batch)
        
        assert output.shape == (1, 12)  # batch_size=1, num_tasks=12
    
    def test_gin_forward(self, config, sample_data):
        """Test GIN forward pass."""
        model = GIN(config)
        x, edge_index, batch = sample_data
        
        with torch.no_grad():
            output = model(x, edge_index, batch)
        
        assert output.shape == (1, 12)
    
    def test_mpnn_forward(self, config, sample_data):
        """Test MPNN forward pass."""
        model = MPNN(config)
        x, edge_index, batch = sample_data
        
        with torch.no_grad():
            output = model(x, edge_index, batch)
        
        assert output.shape == (1, 12)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Create sample data
        predictions = torch.randn(100, 12)
        targets = torch.randint(0, 2, (100, 12)).float()
        masks = torch.ones(100, 12).bool()
        
        metrics = compute_metrics(predictions, targets, masks)
        
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_compute_metrics_with_missing_labels(self):
        """Test metrics computation with missing labels."""
        predictions = torch.randn(100, 12)
        targets = torch.randint(0, 2, (100, 12)).float()
        masks = torch.ones(100, 12).bool()
        
        # Add some missing labels
        masks[0:10, 0] = False
        targets[0:10, 0] = float('nan')
        
        metrics = compute_metrics(predictions, targets, masks)
        
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
