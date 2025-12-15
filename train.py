#!/usr/bin/env python3
"""Main training script for molecular property prediction."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, create_experiment_dir
from src.data.datasets import create_data_loaders
from src.models.gnn_models import GCN, GIN, MPNN
from src.train.trainer import Trainer
from src.eval.metrics import create_leaderboard


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    
    # Set random seed
    set_seed(config.seed)
    
    # Create experiment directory
    if config.get('experiment_name'):
        exp_dir = create_experiment_dir(config.log_dir, config.experiment_name)
        config.log_dir = exp_dir
        config.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        
        # Save config
        OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("-" * 50)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(config.data)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {config.model._target_.split('.')[-1]} model...")
    model_class = {
        'GCN': GCN,
        'GIN': GIN,
        'MPNN': MPNN
    }[config.model._target_.split('.')[-1]]
    
    model = model_class(config.model)
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Train model
    print("Starting training...")
    final_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    results = {config.model._target_.split('.')[-1]: final_metrics}
    leaderboard = create_leaderboard(results)
    print(leaderboard)
    
    # Save results
    if config.get('experiment_name'):
        import json
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
    
    print(f"\nExperiment completed. Results saved to: {exp_dir}")


def run_experiments(config_path: str = "configs/config.yaml") -> None:
    """Run multiple experiments with different models."""
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Define models to test
    models_to_test = [
        ("src.models.gcn.GCN", "configs/model/gcn.yaml"),
        ("src.models.gin.GIN", "configs/model/gin.yaml"),
    ]
    
    all_results = {}
    
    for model_target, model_config_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_target.split('.')[-1]}")
        print(f"{'='*60}")
        
        # Load model config
        model_config = OmegaConf.load(model_config_path)
        
        # Update config
        config.model = model_config
        config.model._target_ = model_target
        config.experiment_name = f"molecular_prediction_{model_target.split('.')[-1].lower()}"
        
        # Run experiment
        try:
            # Set seed for reproducibility
            set_seed(config.seed)
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(config.data)
            
            # Create model
            model_class = {
                'GCN': GCN,
                'GIN': GIN,
                'MPNN': MPNN
            }[model_target.split('.')[-1]]
            
            model = model_class(config.model)
            
            # Create trainer
            trainer = Trainer(model, config)
            
            # Train model
            final_metrics = trainer.train(train_loader, val_loader, test_loader)
            
            # Store results
            model_name = model_target.split('.')[-1]
            all_results[model_name] = final_metrics
            
        except Exception as e:
            print(f"Error training {model_target}: {e}")
            continue
    
    # Print final leaderboard
    print(f"\n{'='*60}")
    print("FINAL LEADERBOARD")
    print(f"{'='*60}")
    
    leaderboard = create_leaderboard(all_results)
    print(leaderboard)
    
    # Save results
    import json
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to: experiment_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train molecular property prediction models")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                       help="Run single experiment or multiple experiments")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Run single experiment with Hydra
        main()
    else:
        # Run multiple experiments
        run_experiments(args.config)
