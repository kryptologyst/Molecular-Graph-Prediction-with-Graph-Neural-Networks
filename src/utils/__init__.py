"""Utility functions for molecular graph prediction project."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If "auto", automatically select best available.
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the model size in megabytes.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object.
        save_path: Path to save configuration.
    """
    from omegaconf import OmegaConf
    OmegaConf.save(config, save_path)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        
    Returns:
        Path to created experiment directory.
    """
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary for logging.
    
    Args:
        metrics: Dictionary of metric names and values.
        precision: Number of decimal places.
        
    Returns:
        Formatted string of metrics.
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)
