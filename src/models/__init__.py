"""Graph Neural Network models for molecular property prediction."""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm1d, LayerNorm
from omegaconf import DictConfig


class GCN(nn.Module):
    """Graph Convolutional Network for molecular property prediction.
    
    This model uses Graph Convolutional layers with batch normalization,
    dropout, and residual connections for molecular property prediction.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize GCN model.
        
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.use_residual = config.get('use_residual', False)
        self.num_tasks = config.num_tasks
        
        # Input dimension will be set during forward pass
        self.input_dim = None
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Pooling layer
        self.pooling = global_mean_pool
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tasks)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN model.
        
        Args:
            x: Node features tensor of shape [num_nodes, num_features].
            edge_index: Edge index tensor of shape [2, num_edges].
            batch: Batch assignment tensor of shape [num_nodes].
            
        Returns:
            Graph-level predictions of shape [batch_size, num_tasks].
        """
        if self.input_dim is None:
            self.input_dim = x.size(1)
            self._build_layers()
        
        # Store input for residual connections
        h = x
        
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index)
            
            if self.use_batch_norm:
                h_new = bn(h_new)
            
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.use_residual and i > 0 and h.size() == h_new.size():
                h_new = h_new + h
            
            h = h_new
        
        # Global pooling
        h = self.pooling(h, batch)
        
        # Classification
        out = self.classifier(h)
        
        return out
    
    def _build_layers(self):
        """Build GCN layers based on input dimension."""
        # First layer
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        self.batch_norms.append(BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(BatchNorm1d(self.hidden_dim))


class GIN(nn.Module):
    """Graph Isomorphism Network for molecular property prediction.
    
    GIN is particularly effective for molecular graphs due to its
    ability to capture graph isomorphism properties.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize GIN model.
        
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.use_residual = config.get('use_residual', True)
        self.num_tasks = config.num_tasks
        self.eps = config.get('eps', 0.0)
        self.train_eps = config.get('train_eps', True)
        
        # Input dimension will be set during forward pass
        self.input_dim = None
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        # Pooling layer
        self.pooling = global_add_pool  # GIN typically uses sum pooling
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tasks)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GIN model.
        
        Args:
            x: Node features tensor of shape [num_nodes, num_features].
            edge_index: Edge index tensor of shape [2, num_edges].
            batch: Batch assignment tensor of shape [num_nodes].
            
        Returns:
            Graph-level predictions of shape [batch_size, num_tasks].
        """
        if self.input_dim is None:
            self.input_dim = x.size(1)
            self._build_layers()
        
        # Store input for residual connections
        h = x
        
        # Apply GIN layers
        for i, (conv, bn, mlp) in enumerate(zip(self.convs, self.batch_norms, self.mlps)):
            h_new = conv(h, edge_index)
            
            if self.use_batch_norm:
                h_new = bn(h_new)
            
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Apply MLP
            h_new = mlp(h_new)
            
            # Residual connection
            if self.use_residual and i > 0 and h.size() == h_new.size():
                h_new = h_new + h
            
            h = h_new
        
        # Global pooling
        h = self.pooling(h, batch)
        
        # Classification
        out = self.classifier(h)
        
        return out
    
    def _build_layers(self):
        """Build GIN layers based on input dimension."""
        # First layer
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ),
            eps=self.eps,
            train_eps=self.train_eps
        ))
        self.batch_norms.append(BatchNorm1d(self.hidden_dim))
        self.mlps.append(nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                ),
                eps=self.eps,
                train_eps=self.train_eps
            ))
            self.batch_norms.append(BatchNorm1d(self.hidden_dim))
            self.mlps.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))


class MPNN(nn.Module):
    """Message Passing Neural Network for molecular property prediction.
    
    MPNN is a general framework that can handle edge features and
    is particularly suitable for molecular graphs with bond information.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize MPNN model.
        
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.num_tasks = config.num_tasks
        
        # Input dimension will be set during forward pass
        self.input_dim = None
        self.edge_dim = None
        
        # Message passing layers
        self.message_nns = nn.ModuleList()
        self.update_nns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Pooling layer
        self.pooling = global_mean_pool
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_tasks)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the MPNN model.
        
        Args:
            x: Node features tensor of shape [num_nodes, num_features].
            edge_index: Edge index tensor of shape [2, num_edges].
            batch: Batch assignment tensor of shape [num_nodes].
            edge_attr: Edge features tensor of shape [num_edges, edge_features].
            
        Returns:
            Graph-level predictions of shape [batch_size, num_tasks].
        """
        if self.input_dim is None:
            self.input_dim = x.size(1)
            self.edge_dim = edge_attr.size(1) if edge_attr is not None else 0
            self._build_layers()
        
        h = x
        
        # Apply message passing layers
        for i, (message_nn, update_nn, bn) in enumerate(zip(self.message_nns, self.update_nns, self.batch_norms)):
            # Message passing
            messages = self._message_passing(h, edge_index, edge_attr, message_nn)
            
            # Update node features
            h_new = update_nn(torch.cat([h, messages], dim=-1))
            
            if self.use_batch_norm:
                h_new = bn(h_new)
            
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            h = h_new
        
        # Global pooling
        h = self.pooling(h, batch)
        
        # Classification
        out = self.classifier(h)
        
        return out
    
    def _message_passing(self, x: torch.Tensor, edge_index: torch.Tensor, 
                        edge_attr: Optional[torch.Tensor], message_nn: nn.Module) -> torch.Tensor:
        """Perform message passing step.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge features.
            message_nn: Message neural network.
            
        Returns:
            Aggregated messages for each node.
        """
        row, col = edge_index
        
        # Create message inputs
        if edge_attr is not None:
            message_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        else:
            message_input = torch.cat([x[row], x[col]], dim=-1)
        
        # Compute messages
        messages = message_nn(message_input)
        
        # Aggregate messages
        out = torch.zeros(x.size(0), messages.size(1), device=x.device)
        out.scatter_add_(0, col.unsqueeze(1).expand_as(messages), messages)
        
        return out
    
    def _build_layers(self):
        """Build MPNN layers based on input dimensions."""
        # Message and update networks
        for _ in range(self.num_layers):
            # Message network
            if self.edge_dim > 0:
                message_input_dim = 2 * self.input_dim + self.edge_dim
            else:
                message_input_dim = 2 * self.input_dim
            
            self.message_nns.append(nn.Sequential(
                nn.Linear(message_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
            
            # Update network
            self.update_nns.append(nn.Sequential(
                nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
            
            self.batch_norms.append(BatchNorm1d(self.hidden_dim))
            
            # Update input dimension for next layer
            self.input_dim = self.hidden_dim
