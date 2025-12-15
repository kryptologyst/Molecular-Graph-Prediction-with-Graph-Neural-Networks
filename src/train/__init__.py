"""Training utilities and trainer classes."""

import os
import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm

from ..utils import get_device, format_metrics, count_parameters, get_model_size_mb
from ..eval.metrics import compute_metrics


class Trainer:
    """Trainer class for molecular property prediction models."""
    
    def __init__(self, model: nn.Module, config: DictConfig):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            config: Configuration object.
        """
        self.model = model
        self.config = config
        self.device = get_device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup logging
        self.writer = None
        if config.get('log_dir'):
            self.writer = SummaryWriter(config.log_dir)
        
        # Initialize wandb if configured
        if config.get('wandb', {}).get('project'):
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.get('entity'),
                tags=config.wandb.get('tags', []),
                config=dict(config)
            )
            wandb.watch(self.model)
        
        # Training state
        self.best_val_score = -float('inf')
        self.patience_counter = 0
        self.epoch = 0
        
        # Print model info
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Model size: {get_model_size_mb(self.model):.2f} MB")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('lr', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        loss_fn = self.config.get('loss_fn', 'bce_with_logits').lower()
        
        if loss_fn == 'bce':
            return nn.BCELoss()
        elif loss_fn == 'bce_with_logits':
            return nn.BCEWithLogitsLoss()
        elif loss_fn == 'mse':
            return nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                predictions = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            else:
                predictions = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Handle missing labels
            mask = ~torch.isnan(batch.y)
            
            if mask.sum() == 0:
                continue  # Skip batch if no valid labels
            
            # Compute loss
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                loss = self.criterion(predictions[mask], batch.y[mask])
            else:
                loss = self.criterion(torch.sigmoid(predictions[mask]), batch.y[mask])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(batch.y.detach().cpu())
            all_masks.append(mask.detach().cpu())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = compute_metrics(all_predictions, all_targets, all_masks)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    predictions = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                else:
                    predictions = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Handle missing labels
                mask = ~torch.isnan(batch.y)
                
                if mask.sum() == 0:
                    continue
                
                # Compute loss
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    loss = self.criterion(predictions[mask], batch.y[mask])
                else:
                    loss = self.criterion(torch.sigmoid(predictions[mask]), batch.y[mask])
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.y.cpu())
                all_masks.append(mask.cpu())
        
        # Compute validation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = compute_metrics(all_predictions, all_targets, all_masks)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(self, train_loader, val_loader, test_loader=None) -> Dict[str, float]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader (optional).
            
        Returns:
            Dictionary of final test metrics.
        """
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.get('epochs', 100)):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('roc_auc', val_metrics.get('accuracy', 0)))
                else:
                    self.scheduler.step()
            
            # Logging
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Early stopping and checkpointing
            if self._should_save_checkpoint(val_metrics):
                self._save_checkpoint(val_metrics)
            
            if self._should_stop_early(val_metrics):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            print(f"Epoch {epoch:03d} | "
                  f"Train: {format_metrics(train_metrics)} | "
                  f"Val: {format_metrics(val_metrics)}")
        
        # Load best model for testing
        self._load_best_checkpoint()
        
        # Final test evaluation
        if test_loader:
            test_metrics = self.validate(test_loader)
            print(f"Final Test: {format_metrics(test_metrics)}")
            return test_metrics
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return val_metrics
    
    def _log_metrics(self, train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float], epoch: int):
        """Log metrics to various backends."""
        # TensorBoard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Wandb
        if wandb.run:
            log_dict = {}
            for key, value in train_metrics.items():
                log_dict[f'train/{key}'] = value
            for key, value in val_metrics.items():
                log_dict[f'val/{key}'] = value
            log_dict['epoch'] = epoch
            log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
    
    def _should_save_checkpoint(self, val_metrics: Dict[str, float]) -> bool:
        """Check if checkpoint should be saved."""
        score = val_metrics.get('roc_auc', val_metrics.get('accuracy', 0))
        return score > self.best_val_score
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        if not self.config.get('early_stopping', True):
            return False
        
        score = val_metrics.get('roc_auc', val_metrics.get('accuracy', 0))
        
        if score > self.best_val_score:
            self.best_val_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        patience = self.config.get('patience', 20)
        return self.patience_counter >= patience
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': dict(self.config)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
        
        if self.config.get('save_last', True):
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'last_model.pt'))
    
    def _load_best_checkpoint(self):
        """Load the best checkpoint."""
        checkpoint_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'best_model.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        else:
            print("No checkpoint found, using current model")
