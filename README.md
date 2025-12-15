# Molecular Graph Prediction with Graph Neural Networks

A production-ready implementation of Graph Neural Networks for molecular property prediction, specifically designed for the Tox21 dataset. This project showcases state-of-the-art GNN architectures including GCN, GIN, and MPNN for predicting chemical toxicity properties.

## Features

- **Multiple GNN Architectures**: GCN, GIN, and MPNN implementations optimized for molecular graphs
- **Comprehensive Evaluation**: ROC-AUC, precision, recall, F1-score, and task-specific metrics
- **Robust Data Pipeline**: Support for scaffold-based splits, data augmentation, and missing label handling
- **Interactive Demo**: Streamlit-based web application for model exploration and visualization
- **Production Ready**: Type hints, comprehensive logging, checkpointing, and reproducible experiments
- **Modern Stack**: PyTorch 2.x, PyTorch Geometric, Hydra configuration, and Wandb logging

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Molecular-Graph-Prediction-with-Graph-Neural-Networks.git
cd Molecular-Graph-Prediction-with-Graph-Neural-Networks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch Geometric (if not already installed):
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

### Training a Model

Train a single model:
```bash
python train.py --mode single
```

Train multiple models for comparison:
```bash
python train.py --mode multi
```

### Running the Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
molecular_graph_prediction/
├── src/
│   ├── models/           # GNN model implementations
│   ├── data/             # Dataset classes and data utilities
│   ├── train/            # Training utilities and trainer classes
│   ├── eval/             # Evaluation metrics and utilities
│   └── utils/            # General utility functions
├── configs/              # Hydra configuration files
├── data/                 # Data storage directories
├── demo/                 # Streamlit demo application
├── scripts/              # Utility scripts
├── tests/                # Unit tests
├── assets/               # Generated visualizations and results
└── train.py              # Main training script
```

## Model Architectures

### Graph Convolutional Network (GCN)
- Neighborhood aggregation with mean pooling
- Batch normalization and dropout for regularization
- Residual connections for deeper networks
- Suitable for general graph classification tasks

### Graph Isomorphism Network (GIN)
- Sum aggregation with learnable epsilon parameter
- MLP-based message passing
- Particularly effective for molecular graphs
- Captures graph isomorphism properties

### Message Passing Neural Network (MPNN)
- General framework for message passing
- Support for edge features
- Flexible message and update functions
- Ideal for molecular graphs with bond information

## Dataset

The Tox21 dataset contains molecular graphs with 12 toxicity prediction tasks:

**Nuclear Receptor (NR) Tasks:**
- NR-AR: Androgen receptor
- NR-AR-LBD: Androgen receptor ligand binding domain
- NR-AhR: Aryl hydrocarbon receptor
- NR-Aromatase: Aromatase
- NR-ER: Estrogen receptor
- NR-ER-LBD: Estrogen receptor ligand binding domain
- NR-PPAR-gamma: Peroxisome proliferator-activated receptor gamma

**Stress Response (SR) Tasks:**
- SR-ARE: Antioxidant response element
- SR-ATAD5: ATAD5
- SR-HSE: Heat shock response element
- SR-MMP: Mitochondrial membrane potential
- SR-p53: p53

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific configurations
- `configs/data/`: Dataset configurations
- `configs/train/`: Training configurations

### Example Configuration

```yaml
# Model configuration
model:
  _target_: src.models.gcn.GCN
  hidden_dim: 64
  num_layers: 2
  dropout: 0.1
  use_batch_norm: true
  num_tasks: 12

# Data configuration
data:
  dataset_name: Tox21
  batch_size: 32
  split_type: scaffold  # or random
  augment: true
  augment_types: [edge_drop, feature_mask]

# Training configuration
train:
  epochs: 100
  lr: 0.001
  optimizer: adam
  scheduler: cosine
  early_stopping: true
  patience: 20
```

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

- **ROC-AUC**: Area under the ROC curve for each task
- **Average Precision**: Mean average precision across tasks
- **Accuracy**: Binary classification accuracy
- **Precision/Recall/F1**: Per-task and macro-averaged metrics
- **Task-specific Metrics**: Individual task performance analysis

## Data Augmentation

Supported augmentation techniques:

- **Edge Drop**: Randomly remove edges with probability p
- **Feature Mask**: Randomly mask node features
- **Node Drop**: Randomly remove nodes and their connections

## Interactive Demo

The Streamlit demo provides:

- **Molecular Visualization**: Interactive graph visualization with Plotly
- **Property Prediction**: Real-time predictions for selected molecules
- **Model Comparison**: Side-by-side comparison of different architectures
- **Confidence Analysis**: Prediction confidence and uncertainty visualization

### Demo Features

1. **Molecule Selection**: Choose from sample molecules or upload custom SMILES
2. **Graph Visualization**: Interactive molecular graph with node importance coloring
3. **Property Predictions**: Real-time toxicity predictions for all 12 tasks
4. **Model Comparison**: Compare predictions across different GNN architectures
5. **Confidence Analysis**: Visualize prediction confidence and uncertainty

## Training and Evaluation

### Training Process

1. **Data Loading**: Automatic download and preprocessing of Tox21 dataset
2. **Data Splitting**: Scaffold-based or random splitting for train/val/test
3. **Model Training**: Configurable training with early stopping and checkpointing
4. **Evaluation**: Comprehensive metrics computation and logging
5. **Visualization**: Automatic generation of training curves and results

### Reproducibility

- Deterministic seeding for all random operations
- Device-agnostic training (CUDA/MPS/CPU)
- Comprehensive logging with TensorBoard and Wandb
- Configuration versioning and experiment tracking

## Advanced Features

### Scalability

- Efficient data loading with multiple workers
- Gradient accumulation for large batch sizes
- Mixed precision training support
- Memory-efficient graph batching

### Monitoring

- Real-time training metrics visualization
- Model performance tracking
- Resource utilization monitoring
- Experiment comparison tools

### Deployment

- Model checkpointing and loading
- Inference optimization
- API-ready model serving
- Docker containerization support

## Results

Typical performance on Tox21 dataset:

| Model | ROC-AUC | Average Precision | Accuracy |
|-------|---------|-------------------|----------|
| GCN   | 0.75    | 0.68             | 0.82     |
| GIN   | 0.78    | 0.71             | 0.84     |
| MPNN  | 0.76    | 0.69             | 0.83     |

*Results may vary based on hyperparameters and random seeds*

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric team for the excellent GNN framework
- Tox21 dataset contributors for providing the molecular toxicity data
- Streamlit team for the interactive demo framework
- Hydra team for configuration management

## Troubleshooting

### Common Issues

1. **PyTorch Geometric Installation**: Ensure PyTorch version compatibility
2. **CUDA Issues**: Check CUDA version compatibility with PyTorch
3. **Memory Issues**: Reduce batch size or use gradient accumulation
4. **Data Loading**: Ensure sufficient disk space for dataset download

### Getting Help

- Check the issues section for common problems
- Create a new issue with detailed error information
- Include system information and configuration details

## Future Work

- [ ] Support for additional molecular datasets (QM9, ZINC)
- [ ] 3D molecular graph support
- [ ] Multi-task learning improvements
- [ ] Uncertainty quantification
- [ ] Model interpretability tools
- [ ] Distributed training support
- [ ] Model compression techniques
- [ ] Real-time inference optimization
# Molecular-Graph-Prediction-with-Graph-Neural-Networks
