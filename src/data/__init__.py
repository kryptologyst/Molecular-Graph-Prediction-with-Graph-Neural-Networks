"""Data utilities and augmentation functions."""

import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Compose
import numpy as np


class EdgeDrop:
    """Randomly drop edges from the graph.
    
    Args:
        p: Probability of dropping each edge.
    """
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if self.p == 0:
            return data
        
        num_edges = data.edge_index.size(1)
        keep_mask = torch.rand(num_edges) > self.p
        
        data.edge_index = data.edge_index[:, keep_mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep_mask]
        
        return data


class FeatureMask:
    """Randomly mask node features.
    
    Args:
        p: Probability of masking each feature.
        mask_value: Value to use for masking.
    """
    
    def __init__(self, p: float = 0.1, mask_value: float = 0.0):
        self.p = p
        self.mask_value = mask_value
    
    def __call__(self, data: Data) -> Data:
        if self.p == 0:
            return data
        
        mask = torch.rand(data.x.size()) > self.p
        data.x = data.x * mask + (1 - mask) * self.mask_value
        
        return data


class NodeDrop:
    """Randomly drop nodes from the graph.
    
    Args:
        p: Probability of dropping each node.
    """
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if self.p == 0:
            return data
        
        num_nodes = data.x.size(0)
        keep_mask = torch.rand(num_nodes) > self.p
        
        # Update node features
        data.x = data.x[keep_mask]
        
        # Update edge indices
        node_mapping = torch.cumsum(keep_mask.long(), dim=0) - 1
        node_mapping[~keep_mask] = -1
        
        # Filter edges connecting to kept nodes
        edge_mask = (node_mapping[data.edge_index[0]] >= 0) & (node_mapping[data.edge_index[1]] >= 0)
        data.edge_index = data.edge_index[:, edge_mask]
        
        # Update edge indices to new node indices
        data.edge_index[0] = node_mapping[data.edge_index[0]]
        data.edge_index[1] = node_mapping[data.edge_index[1]]
        
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]
        
        return data


def get_augmentation_transforms(augment_types: List[str], augment_prob: float = 0.1) -> Compose:
    """Get augmentation transforms based on configuration.
    
    Args:
        augment_types: List of augmentation types to apply.
        augment_prob: Probability of applying each augmentation.
        
    Returns:
        Compose transform object.
    """
    transforms = []
    
    for aug_type in augment_types:
        if aug_type == "edge_drop":
            transforms.append(EdgeDrop(p=augment_prob))
        elif aug_type == "feature_mask":
            transforms.append(FeatureMask(p=augment_prob))
        elif aug_type == "node_drop":
            transforms.append(NodeDrop(p=augment_prob))
    
    return Compose(transforms)


def collate_fn(batch: List[Data]) -> Batch:
    """Custom collate function for batching graphs.
    
    Args:
        batch: List of Data objects.
        
    Returns:
        Batched Data object.
    """
    return Batch.from_data_list(batch)


def get_scaffold_split(smiles_list: List[str], train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
    """Split molecules based on molecular scaffolds.
    
    Args:
        smiles_list: List of SMILES strings.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        scaffolds = {}
        for idx, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    if scaffold_smiles not in scaffolds:
                        scaffolds[scaffold_smiles] = []
                    scaffolds[scaffold_smiles].append(idx)
            except:
                # If scaffold generation fails, assign to a unique scaffold
                scaffolds[f"unknown_{idx}"] = [idx]
        
        # Sort scaffolds by size (largest first)
        scaffold_groups = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for group in scaffold_groups:
            if len(train_indices) / len(smiles_list) < train_ratio:
                train_indices.extend(group)
            elif len(val_indices) / len(smiles_list) < val_ratio:
                val_indices.extend(group)
            else:
                test_indices.extend(group)
        
        return train_indices, val_indices, test_indices
        
    except ImportError:
        # Fallback to random split if RDKit is not available
        import random
        indices = list(range(len(smiles_list)))
        random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        return train_indices, val_indices, test_indices
