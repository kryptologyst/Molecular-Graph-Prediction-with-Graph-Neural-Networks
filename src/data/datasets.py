"""Dataset classes for molecular graph prediction."""

import os
import pickle
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig

from ..utils import get_scaffold_split


class Tox21Dataset(InMemoryDataset):
    """Tox21 dataset for molecular property prediction.
    
    This dataset contains molecular graphs with 12 toxicity prediction tasks.
    Each molecule is represented as a graph where atoms are nodes and bonds are edges.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        pre_filter: Optional[callable] = None,
        config: Optional[DictConfig] = None
    ):
        """Initialize Tox21 dataset.
        
        Args:
            root: Root directory where dataset will be saved.
            split: Dataset split ('train', 'val', 'test').
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pre_filter: Pre-filter to apply before saving.
            config: Configuration object.
        """
        self.split = split
        self.config = config or DictConfig({})
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load the appropriate split
        if split == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == "val":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == "test":
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            raise ValueError(f"Unknown split: {split}")
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return ["tox21.csv"]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return ["train.pt", "val.pt", "test.pt"]
    
    def download(self):
        """Download the dataset."""
        try:
            from torch_geometric.datasets import MoleculeNet
            
            # Download Tox21 dataset
            dataset = MoleculeNet(root=self.raw_dir, name='Tox21')
            
            # Convert to CSV format for easier processing
            self._convert_to_csv(dataset)
            
        except ImportError:
            # Create synthetic data if PyG is not available
            self._create_synthetic_data()
    
    def _convert_to_csv(self, dataset):
        """Convert PyG dataset to CSV format."""
        import pandas as pd
        
        data_list = []
        for i, data in enumerate(dataset):
            smiles = data.smiles if hasattr(data, 'smiles') else f"C{i}"
            
            # Convert graph to edge list
            edge_list = data.edge_index.t().numpy()
            
            # Get labels (handle missing values)
            labels = data.y.numpy() if data.y is not None else np.zeros(12)
            
            data_list.append({
                'smiles': smiles,
                'edge_list': edge_list,
                'labels': labels,
                'node_features': data.x.numpy() if data.x is not None else np.zeros((1, 9))
            })
        
        df = pd.DataFrame(data_list)
        df.to_csv(os.path.join(self.raw_dir, "tox21.csv"), index=False)
    
    def _create_synthetic_data(self):
        """Create synthetic molecular data for testing."""
        import pandas as pd
        
        # Create synthetic SMILES and labels
        smiles_list = [
            "CCO", "CC(=O)O", "CCN", "CC(C)O", "CCCC",
            "c1ccccc1", "CCc1ccccc1", "CC(=O)c1ccccc1",
            "CCN(CC)CC", "CC(C)(C)O", "CCCCCC", "CCCCCCC"
        ] * 50  # Repeat to get more samples
        
        data_list = []
        for i, smiles in enumerate(smiles_list):
            # Create random edge list (simplified)
            num_atoms = len(smiles.replace('(', '').replace(')', '').replace('=', ''))
            edges = []
            for j in range(num_atoms - 1):
                edges.append([j, j + 1])
            
            # Random labels for 12 tasks
            labels = np.random.randint(0, 2, 12).astype(float)
            labels[np.random.choice(12, 3)] = np.nan  # Some missing labels
            
            data_list.append({
                'smiles': smiles,
                'edge_list': np.array(edges),
                'labels': labels,
                'node_features': np.random.randn(num_atoms, 9)
            })
        
        df = pd.DataFrame(data_list)
        df.to_csv(os.path.join(self.raw_dir, "tox21.csv"), index=False)
    
    def process(self):
        """Process the raw data and create splits."""
        import pandas as pd
        
        # Load raw data
        df = pd.read_csv(os.path.join(self.raw_dir, "tox21.csv"))
        
        # Convert to PyG Data objects
        data_list = []
        smiles_list = []
        
        for _, row in df.iterrows():
            # Parse edge list
            edge_list = eval(row['edge_list']) if isinstance(row['edge_list'], str) else row['edge_list']
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Parse node features
            node_features = eval(row['node_features']) if isinstance(row['node_features'], str) else row['node_features']
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Parse labels
            labels = eval(row['labels']) if isinstance(row['labels'], str) else row['labels']
            y = torch.tensor(labels, dtype=torch.float)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.smiles = row['smiles']
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
            smiles_list.append(row['smiles'])
        
        # Create splits
        if self.config.get('split_type', 'random') == 'scaffold':
            train_idx, val_idx, test_idx = get_scaffold_split(
                smiles_list,
                self.config.get('train_ratio', 0.7),
                self.config.get('val_ratio', 0.15),
                self.config.get('test_ratio', 0.15)
            )
        else:
            # Random split
            indices = np.random.permutation(len(data_list))
            n_train = int(len(indices) * self.config.get('train_ratio', 0.7))
            n_val = int(len(indices) * self.config.get('val_ratio', 0.15))
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
        
        # Save splits
        torch.save(self.collate(data_list[train_idx]), self.processed_paths[0])
        torch.save(self.collate(data_list[val_idx]), self.processed_paths[1])
        torch.save(self.collate(data_list[test_idx]), self.processed_paths[2])


def create_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from ..data.augmentation import get_augmentation_transforms
    
    # Get augmentation transforms
    augment_transforms = None
    if config.get('augment', False):
        augment_transforms = get_augmentation_transforms(
            config.get('augment_types', ['edge_drop', 'feature_mask']),
            config.get('augment_prob', 0.1)
        )
    
    # Create datasets
    train_dataset = Tox21Dataset(
        root=config.data_dir,
        split="train",
        transform=augment_transforms,
        config=config
    )
    
    val_dataset = Tox21Dataset(
        root=config.data_dir,
        split="val",
        config=config
    )
    
    test_dataset = Tox21Dataset(
        root=config.data_dir,
        split="test",
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader
