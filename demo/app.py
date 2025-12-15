"""Interactive Streamlit demo for molecular property prediction."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import get_device, set_seed
from src.models.gnn_models import GCN, GIN, MPNN
from src.data.datasets import Tox21Dataset
from src.eval.metrics import compute_metrics


# Page configuration
st.set_page_config(
    page_title="Molecular Property Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample molecular data for demonstration."""
    try:
        # Try to load real Tox21 data
        dataset = Tox21Dataset(root='data/raw', split='test')
        return dataset[:10]  # Return first 10 samples
    except:
        # Create synthetic data if real data not available
        return create_synthetic_data()


def create_synthetic_data():
    """Create synthetic molecular data for demonstration."""
    synthetic_data = []
    
    # Sample SMILES and properties
    sample_molecules = [
        ("CCO", "Ethanol", [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("CC(=O)O", "Acetic Acid", [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("CCN", "Ethylamine", [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
        ("c1ccccc1", "Benzene", [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("CCc1ccccc1", "Ethylbenzene", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ]
    
    for smiles, name, properties in sample_molecules:
        # Create simple graph structure
        num_atoms = len(smiles.replace('(', '').replace(')', '').replace('=', ''))
        edges = []
        for i in range(num_atoms - 1):
            edges.append([i, i + 1])
        
        # Create node features (simplified)
        node_features = np.random.randn(num_atoms, 9)
        
        synthetic_data.append({
            'smiles': smiles,
            'name': name,
            'properties': properties,
            'edges': edges,
            'node_features': node_features
        })
    
    return synthetic_data


@st.cache_resource
def load_model(model_type: str, checkpoint_path: Optional[str] = None):
    """Load a trained model."""
    try:
        # Model configuration
        config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'use_batch_norm': True,
            'use_residual': False,
            'num_tasks': 12
        }
        
        # Create model
        if model_type == "GCN":
            model = GCN(config)
        elif model_type == "GIN":
            model = GIN(config)
        elif model_type == "MPNN":
            model = MPNN(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint if available
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def visualize_molecule_graph(smiles: str, edges: List[List[int]], 
                           node_features: np.ndarray, predictions: Optional[np.ndarray] = None):
    """Create an interactive visualization of the molecular graph."""
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(node_features)):
        G.add_node(i, features=node_features[i])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Atom {node}<br>Features: {node_features[node][:3]}")
        
        # Color nodes based on predictions if available
        if predictions is not None:
            node_colors.append(predictions[node])
        else:
            node_colors.append(0.5)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=20,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                x=1.1,
                len=0.5,
                title="Node<br>Importance"
            ),
            line=dict(width=2, color='black')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Molecular Graph: {smiles}',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive molecular graph visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='gray', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧪 Molecular Property Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive demo allows you to explore molecular property prediction using Graph Neural Networks.
    You can visualize molecular graphs, make predictions, and compare different model architectures.
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Architecture",
        ["GCN", "GIN", "MPNN"],
        help="Choose the Graph Neural Network architecture"
    )
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Molecule selection
    if isinstance(sample_data, list):
        molecule_names = [f"{mol['name']} ({mol['smiles']})" for mol in sample_data]
        selected_idx = st.sidebar.selectbox("Select Molecule", range(len(molecule_names)), 
                                           format_func=lambda x: molecule_names[x])
        selected_molecule = sample_data[selected_idx]
    else:
        # Handle PyG dataset
        selected_idx = st.sidebar.selectbox("Select Molecule", range(len(sample_data)))
        selected_molecule = sample_data[selected_idx]
        selected_molecule = {
            'smiles': getattr(selected_molecule, 'smiles', f'Molecule_{selected_idx}'),
            'name': f'Molecule_{selected_idx}',
            'properties': selected_molecule.y.numpy() if selected_molecule.y is not None else np.zeros(12),
            'edges': selected_molecule.edge_index.t().numpy().tolist(),
            'node_features': selected_molecule.x.numpy() if selected_molecule.x is not None else np.random.randn(5, 9)
        }
    
    # Load model
    model = load_model(model_type)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Molecular Graph Visualization")
        
        # Make prediction if model is available
        predictions = None
        if model is not None:
            try:
                # Prepare input data
                x = torch.tensor(selected_molecule['node_features'], dtype=torch.float)
                edge_index = torch.tensor(selected_molecule['edges'], dtype=torch.long).t().contiguous()
                batch = torch.zeros(x.size(0), dtype=torch.long)
                
                # Make prediction
                with torch.no_grad():
                    pred = model(x, edge_index, batch)
                    predictions = torch.sigmoid(pred).numpy().flatten()
                
            except Exception as e:
                st.warning(f"Could not make prediction: {e}")
        
        # Visualize graph
        fig = visualize_molecule_graph(
            selected_molecule['smiles'],
            selected_molecule['edges'],
            selected_molecule['node_features'],
            predictions
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Molecule Information")
        
        # Display molecule details
        st.markdown(f"""
        **Name:** {selected_molecule['name']}  
        **SMILES:** `{selected_molecule['smiles']}`  
        **Number of Atoms:** {len(selected_molecule['node_features'])}  
        **Number of Bonds:** {len(selected_molecule['edges'])}
        """)
        
        # Display predictions
        if predictions is not None:
            st.subheader("Property Predictions")
            
            # Tox21 task names
            task_names = [
                "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
            ]
            
            # Create predictions dataframe
            pred_df = pd.DataFrame({
                'Task': task_names,
                'Prediction': predictions,
                'Confidence': np.abs(predictions - 0.5) * 2
            })
            
            # Display predictions
            for i, row in pred_df.iterrows():
                color = "green" if row['Prediction'] > 0.5 else "red"
                st.markdown(f"""
                <div class="prediction-box">
                    <strong>{row['Task']}</strong><br>
                    <span style="color: {color}">
                        {row['Prediction']:.3f} 
                        ({row['Confidence']:.1%} confidence)
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        # Display ground truth if available
        if 'properties' in selected_molecule:
            st.subheader("Ground Truth Properties")
            true_properties = selected_molecule['properties']
            
            for i, (task, value) in enumerate(zip(task_names, true_properties)):
                if not np.isnan(value):
                    color = "green" if value > 0.5 else "red"
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{task}</strong><br>
                        <span style="color: {color}">{value:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Model comparison section
    st.subheader("Model Comparison")
    
    # Load all models for comparison
    models_to_compare = ["GCN", "GIN", "MPNN"]
    comparison_results = {}
    
    for model_name in models_to_compare:
        try:
            model = load_model(model_name)
            if model is not None:
                # Make prediction
                x = torch.tensor(selected_molecule['node_features'], dtype=torch.float)
                edge_index = torch.tensor(selected_molecule['edges'], dtype=torch.long).t().contiguous()
                batch = torch.zeros(x.size(0), dtype=torch.long)
                
                with torch.no_grad():
                    pred = model(x, edge_index, batch)
                    predictions = torch.sigmoid(pred).numpy().flatten()
                
                comparison_results[model_name] = predictions
        except:
            continue
    
    if comparison_results:
        # Create comparison plot
        task_names = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=task_names,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, task in enumerate(task_names):
            row = i // 4 + 1
            col = i % 4 + 1
            
            models = list(comparison_results.keys())
            values = [comparison_results[model][i] for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=task, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Predictions Comparison",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Demo:**
    
    This interactive demo showcases molecular property prediction using Graph Neural Networks.
    The models predict toxicity properties for molecules from the Tox21 dataset, which contains
    12 different toxicity prediction tasks.
    
    **Model Architectures:**
    - **GCN**: Graph Convolutional Network with neighborhood aggregation
    - **GIN**: Graph Isomorphism Network with sum aggregation
    - **MPNN**: Message Passing Neural Network with edge features
    
    **Tox21 Tasks:**
    The dataset includes predictions for nuclear receptor (NR) and stress response (SR) pathways
    related to drug toxicity and environmental chemical safety.
    """)


if __name__ == "__main__":
    main()
