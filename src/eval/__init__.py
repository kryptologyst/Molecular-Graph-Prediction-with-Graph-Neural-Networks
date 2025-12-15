"""Evaluation metrics and utilities."""

from typing import Dict, Optional
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   masks: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive metrics for molecular property prediction.
    
    Args:
        predictions: Model predictions of shape [batch_size, num_tasks].
        targets: Ground truth labels of shape [batch_size, num_tasks].
        masks: Valid label masks of shape [batch_size, num_tasks].
        
    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}
    
    # Convert to numpy for sklearn metrics
    pred_np = predictions.numpy()
    target_np = targets.numpy()
    mask_np = masks.numpy()
    
    # Handle multi-label classification
    if pred_np.shape[1] > 1:
        # Multi-label metrics
        metrics.update(_compute_multilabel_metrics(pred_np, target_np, mask_np))
    else:
        # Single-label metrics
        metrics.update(_compute_singlelabel_metrics(pred_np, target_np, mask_np))
    
    return metrics


def _compute_multilabel_metrics(predictions: np.ndarray, targets: np.ndarray, 
                               masks: np.ndarray) -> Dict[str, float]:
    """Compute metrics for multi-label classification."""
    metrics = {}
    
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-predictions))
    
    # Flatten for overall metrics
    valid_mask = masks.flatten()
    valid_preds = probs.flatten()[valid_mask]
    valid_targets = targets.flatten()[valid_mask]
    
    if len(valid_targets) > 0:
        # Overall metrics
        metrics['roc_auc'] = roc_auc_score(valid_targets, valid_preds)
        metrics['average_precision'] = average_precision_score(valid_targets, valid_preds)
        
        # Binary predictions for accuracy
        binary_preds = (valid_preds > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(valid_targets, binary_preds)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_targets, binary_preds, average='macro', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
    
    # Per-task metrics
    task_metrics = []
    for task_idx in range(predictions.shape[1]):
        task_mask = masks[:, task_idx]
        if task_mask.sum() > 0:
            task_preds = probs[:, task_idx][task_mask]
            task_targets = targets[:, task_idx][task_mask]
            
            if len(np.unique(task_targets)) > 1:  # Check if both classes present
                task_auc = roc_auc_score(task_targets, task_preds)
                task_metrics.append(task_auc)
    
    if task_metrics:
        metrics['mean_task_auc'] = np.mean(task_metrics)
        metrics['std_task_auc'] = np.std(task_metrics)
    
    return metrics


def _compute_singlelabel_metrics(predictions: np.ndarray, targets: np.ndarray, 
                                masks: np.ndarray) -> Dict[str, float]:
    """Compute metrics for single-label classification."""
    metrics = {}
    
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-predictions))
    
    # Get valid predictions
    valid_mask = masks.flatten()
    valid_preds = probs.flatten()[valid_mask]
    valid_targets = targets.flatten()[valid_mask]
    
    if len(valid_targets) > 0:
        # ROC AUC
        if len(np.unique(valid_targets)) > 1:
            metrics['roc_auc'] = roc_auc_score(valid_targets, valid_preds)
        
        # Average Precision
        metrics['average_precision'] = average_precision_score(valid_targets, valid_preds)
        
        # Accuracy
        binary_preds = (valid_preds > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(valid_targets, binary_preds)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_targets, binary_preds, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
    
    return metrics


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                              masks: torch.Tensor) -> Dict[str, float]:
    """Compute metrics for regression tasks.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth values.
        masks: Valid label masks.
        
    Returns:
        Dictionary of regression metrics.
    """
    metrics = {}
    
    # Get valid predictions
    valid_mask = masks.bool()
    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    if len(valid_preds) > 0:
        # Mean Absolute Error
        mae = torch.mean(torch.abs(valid_preds - valid_targets)).item()
        metrics['mae'] = mae
        
        # Root Mean Square Error
        mse = torch.mean((valid_preds - valid_targets) ** 2).item()
        rmse = np.sqrt(mse)
        metrics['rmse'] = rmse
        metrics['mse'] = mse
        
        # Mean Absolute Percentage Error
        mape = torch.mean(torch.abs((valid_targets - valid_preds) / (valid_targets + 1e-8))).item() * 100
        metrics['mape'] = mape
        
        # R-squared
        ss_res = torch.sum((valid_targets - valid_preds) ** 2)
        ss_tot = torch.sum((valid_targets - torch.mean(valid_targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['r2'] = r2.item()
    
    return metrics


def create_leaderboard(results: Dict[str, Dict[str, float]]) -> str:
    """Create a formatted leaderboard from results.
    
    Args:
        results: Dictionary mapping model names to their metrics.
        
    Returns:
        Formatted leaderboard string.
    """
    if not results:
        return "No results to display."
    
    # Get all metric names
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Create header
    header = f"{'Model':<20}"
    for metric in all_metrics:
        header += f"{metric:<12}"
    
    # Create rows
    rows = [header]
    rows.append("-" * len(header))
    
    # Sort models by primary metric (ROC-AUC or accuracy)
    primary_metric = 'roc_auc' if 'roc_auc' in all_metrics else 'accuracy'
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(primary_metric, 0),
        reverse=True
    )
    
    for model_name, metrics in sorted_models:
        row = f"{model_name:<20}"
        for metric in all_metrics:
            value = metrics.get(metric, 0)
            if isinstance(value, float):
                row += f"{value:<12.4f}"
            else:
                row += f"{value:<12}"
        rows.append(row)
    
    return "\n".join(rows)
