"""
SE-SGformer: Spectral-Enhanced Signed Graph Transformer for Link Sign Prediction

This module implements a graph transformer architecture specifically designed for signed graph
link prediction tasks. The model combines spectral graph features with transformer attention
mechanisms to predict positive, negative, or non-existent relationships between nodes.

Key architectural components:
- Centrality-based node encoding for structural awareness
- Random walk and adjacency matrix encodings for spatial relationships
- Multi-layer transformer encoder with graph-specific attention biases
- Multi-class discriminator for link sign prediction (positive/negative/none)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
from typing import Tuple, Optional, List
from torch import Tensor
from torch_geometric.utils import coalesce, negative_sampling, structured_negative_sampling
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, roc_auc_score

from layer import CentralityEncoding, RWEncoding, ADJEncoding, GraphormerEncoderLayer, create_dummy_spatial_features
from history_extractor import HistoricalContextExtractor

# Device configuration - prioritize GPU acceleration for large-scale graph processing
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class SE_SGformer(nn.Module):
    """
    Spectral-Enhanced Signed Graph Transformer for link sign prediction.
    
    This model addresses the challenge of predicting relationship polarity (positive/negative)
    in signed social networks. Unlike traditional GNNs that treat all edges equally, this
    architecture explicitly models the structural differences between positive and negative
    relationships through specialized encodings and attention mechanisms.
    
    The "spectral enhancement" refers to incorporating eigenspace information from the
    signed adjacency matrix, which captures global structural patterns that local
    message passing might miss.
    """
    
    def __init__(self, args):
        """
        Initialize SE-SGformer with configuration parameters.
        
        Args:
            args: Configuration object containing model hyperparameters
                - num_layers: Number of transformer encoder layers
                - node_dim: Hidden dimension for node embeddings
                - num_heads: Number of attention heads in transformer layers
                - max_degree: Maximum node degree for centrality encoding
        """
        super().__init__()
        
        # Model architecture parameters
        self.num_layers = args.num_layers
        self.num = args.num
        self.input_node_dim = args.num_node_features
        self.node_dim = args.node_dim
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.max_degree = args.max_degree
        self.length = args.length
        self.max_hop = args.max_hop
        
        # Input projection - maps raw node features to model's hidden dimension
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        
        # Structural encodings - capture different aspects of graph topology
        self.centrality_encoding = CentralityEncoding(max_degree=self.max_degree, node_dim=self.node_dim)
        self.spatial_matrix = RWEncoding(num=self.num)  # Random walk-based spatial relationships
        self.adj_matrix = ADJEncoding()  # Direct adjacency relationships
        
        # Stack of transformer encoder layers - core of the attention mechanism
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim=self.node_dim, num_heads=self.num_heads,)
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.lin = nn.Linear(2 * self.node_dim, 3)  # 3-class classifier (pos/neg/none)
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)
    
    def create_spectral_features(self, pos_edge_index: Tensor, neg_edge_index: Tensor, 
                               num_nodes: Optional[int] = None) -> Tensor:
        """
        Generate spectral features from signed adjacency matrix using SVD.
        
        This method constructs a weighted adjacency matrix where positive edges have
        value +1, negative edges have value -1, and applies truncated SVD to extract
        the most informative spectral components. These features capture global
        structural patterns in the signed graph.
        
        Args:
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            neg_edge_index: Negative edge indices [2, num_neg_edges]
            num_nodes: Total number of nodes (inferred if not provided)
            
        Returns:
            Spectral feature matrix [num_nodes, input_node_dim]
        """
        # Combine positive and negative edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        # Assign edge weights: positive edges = +2, negative edges = 0 (will become -1 after offset)
        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        # Create symmetric adjacency matrix by adding reverse edges
        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        # Coalesce duplicate edges and apply offset to get signed values
        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1  # Convert to signed values: +1 for positive, -1 for negative

        # Convert to sparse matrix and apply truncated SVD
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.input_node_dim, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)
    
    def forward(self, x: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        """
        Forward pass through SE-SGformer architecture.
        
        The forward pass implements a multi-stage processing pipeline:
        1. Spatial feature preparation and normalization
        2. Node embedding projection and centrality encoding
        3. Attention bias matrix computation (spatial and adjacency)
        4. Multi-layer transformer processing with graph-aware attention
        5. Final node embedding projection
        
        Args:
            x: Input node features [num_nodes, input_node_dim]
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            neg_edge_index: Negative edge indices [2, num_neg_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Generate and normalize spatial features for attention bias
        feature = create_dummy_spatial_features(x.shape[0], self.num).to(device)
        epsilon = 1e-10
        feature_reciprocal = torch.reciprocal(feature + epsilon)
        row_sum = feature.sum(dim=2, keepdim=True)
        normalized_feature = feature_reciprocal / row_sum
        
        # Project input features to model dimension and apply centrality encoding
        x = self.node_in_lin(x)
        x = self.centrality_encoding(x, pos_edge_index, neg_edge_index)
        
        # Compute attention bias matrices for transformer layers
        spatial_matrix = self.spatial_matrix(normalized_feature).to(device)
        adj_matrix = self.adj_matrix(pos_edge_index, neg_edge_index, x.shape[0]).to(device)
        
        # Process through transformer encoder layers
        for layer in self.layers:
            x = layer(x, adj_matrix, spatial_matrix)
        
        # Final projection to output dimension
        x = self.node_out_lin(x)
        return x
    
    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Classify edge relationships as positive, negative, or non-existent.
        
        This method takes node embeddings and edge indices to predict the relationship
        type between connected nodes. The classification is performed by concatenating
        node embeddings and passing through a linear classifier.
        
        Args:
            z: Node embeddings [num_nodes, node_dim]
            edge_index: Edge indices to classify [2, num_edges]
            
        Returns:
            Log probabilities for each class [num_edges, 3]
        """
        # Concatenate embeddings of connected nodes
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)
    
    def nll_loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        """
        Compute negative log-likelihood loss for three-class edge classification.
        
        The loss function treats edge prediction as a three-class problem:
        - Class 0: Positive edges (should exist with positive sign)
        - Class 1: Negative edges (should exist with negative sign)  
        - Class 2: Non-edges (should not exist)
        
        This formulation allows the model to learn the distinction between
        "no relationship" and "negative relationship".
        
        Args:
            z: Node embeddings [num_nodes, node_dim]
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            neg_edge_index: Negative edge indices [2, num_neg_edges]
            
        Returns:
            Average negative log-likelihood loss across all edge types
        """
        # Generate non-edges through negative sampling
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))
        
        # Compute NLL loss for each edge type
        nll_loss = 0
        nll_loss += F.nll_loss(self.discriminate(z, pos_edge_index), 
                              pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(self.discriminate(z, neg_edge_index), 
                              neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(self.discriminate(z, none_edge_index), 
                              none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(self, z: Tensor, pos_edge_index: Tensor) -> Tensor:
        """
        Triplet loss to encourage positive edge embeddings to be similar.
        
        This loss ensures that nodes connected by positive edges have similar
        embeddings, while being dissimilar to randomly sampled negative nodes.
        Uses structured negative sampling to create meaningful triplets.
        
        Args:
            z: Node embeddings [num_nodes, node_dim]
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            
        Returns:
            Triplet loss for positive edges
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        """
        Triplet loss to encourage negative edge embeddings to be dissimilar.
        
        This loss ensures that nodes connected by negative edges have dissimilar
        embeddings, while potentially being similar to other nodes. The loss
        formulation is reversed compared to positive edges.
        
        Args:
            z: Node embeddings [num_nodes, node_dim]
            neg_edge_index: Negative edge indices [2, num_neg_edges]
            
        Returns:
            Triplet loss for negative edges
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        """
        Combined loss function for signed graph link prediction.
        
        Combines classification loss (NLL) with embedding geometry losses (triplet).
        The triplet losses are weighted more heavily (5x) to ensure the learned
        embeddings have proper geometric properties for signed relationships.
        
        Args:
            z: Node embeddings [num_nodes, node_dim]
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            neg_edge_index: Negative edge indices [2, num_neg_edges]
            
        Returns:
            Combined loss value
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + 5 * (loss_1 + loss_2)

    def test(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[float, float]:
        """
        Evaluate model performance using AUC and F1 metrics.
        
        Converts the three-class problem to binary classification for evaluation.
        Treats positive edges as class 1 and negative edges as class 0.
        Excludes non-edges from evaluation to focus on sign prediction accuracy.
        
        Args:
            z: Node embeddings [num_nodes, node_dim]
            pos_edge_index: Positive edge indices [2, num_pos_edges]
            neg_edge_index: Negative edge indices [2, num_neg_edges]
            
        Returns:
            Tuple of (AUC, F1) scores
        """
        with torch.no_grad():
            # Get predictions for positive and negative edges (ignore non-edge class)
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        
        # Convert to binary classification: 1 for positive, 0 for negative
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat([pred.new_ones((pos_p.size(0))), pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        # Compute metrics if both classes are present
        if len(np.unique(y)) > 1:
            auc = roc_auc_score(y, pred)
            f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
        else:
            auc, f1 = 0.0, 0.0
        return auc, f1


class Temporal_SE_SGformer(nn.Module):
    """
    Temporal extension of SE-SGformer with historical context integration.
    
    This model extends the base SE-SGformer to leverage historical information
    from previous timesteps. The key insight is that in dynamic signed graphs,
    past relationship patterns provide valuable context for predicting current
    relationships.
    
    Two combination strategies are supported:
    1. Global weight: Single learnable parameter for all nodes
    2. Adaptive weights: Node-specific weights learned via MLP
    """
    
    def __init__(self, args):
        """
        Initialize temporal SE-SGformer with historical context capabilities.
        
        Args:
            args: Configuration object with additional temporal parameters
                - use_adaptive_weights: Whether to use node-specific combination weights
                - base_weights: Initial combination weight for global strategy
        """
        super().__init__()
        
        # Original SE-SGformer (unchanged)
        self.base_model = SE_SGformer(args)
        
        # Historical context extractor
        self.history_extractor = HistoricalContextExtractor(args.node_dim)
        
        # Configurable combination strategy
        self.use_adaptive_weights = args.use_adaptive_weights
        self.base_weights = args.base_weights
        
        # Add confidence gating network
        self.confidence_gate = nn.Sequential(
            nn.Linear(args.node_dim * 2, args.node_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.node_dim // 2, args.node_dim // 4),
            nn.ReLU(),
            nn.Linear(args.node_dim // 4, 1),
            nn.Sigmoid()
        )
        
        if self.use_adaptive_weights:
            # Learnable MLP-based combination weights that adapt per node
            self.combination_mlp = nn.Sequential(
                nn.Linear(args.node_dim * 2, args.node_dim // 2),
                nn.ReLU(),
                nn.Linear(args.node_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            # Single learnable parameter for all nodes
            self.combination_weight = nn.Parameter(torch.tensor(self.base_weights))
        
        
    def forward(self, x: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor,
                historical_embeddings: Optional[List[Tensor]] = None) -> Tensor:
        """
        Forward pass with historical context integration.
        
        Processes current graph structure through base model, extracts historical
        context from previous timesteps, and intelligently combines them using
        either global or adaptive weighting strategies.
        
        Args:
            x: Current node features [num_nodes, input_node_dim]
            pos_edge_index: Current positive edges [2, num_pos_edges]
            neg_edge_index: Current negative edges [2, num_neg_edges]
            historical_embeddings: List of previous timestep embeddings (optional)
            
        Returns:
            Enhanced node embeddings incorporating historical context
        """
        # Process current timestep through base SE-SGformer
        # Get current embeddings from base SE-SGformer
        current_embeddings = self.base_model(x, pos_edge_index, neg_edge_index)
        
        # If no history, return current embeddings
        if not historical_embeddings:
            return current_embeddings
        
        # Extract historical context
        historical_context = self.history_extractor(historical_embeddings)
        
        # Combine current and historical information with confidence gating
        if historical_context is not None:
            # Compute confidence in historical context
            # Concatenate current and historical embeddings for confidence assessment
            combined_features = torch.cat([current_embeddings, historical_context], dim=1)
            confidence_scores = self.confidence_gate(combined_features)  # [num_nodes, 1]
            
            if self.use_adaptive_weights:
                # Learn adaptive combination weights per node using MLP
                adaptive_weights = self.combination_mlp(combined_features)  # [num_nodes, 1]
                
                # Apply confidence gating to the adaptive weights
                effective_weights = adaptive_weights * confidence_scores
                
                enhanced_embeddings = (1 - effective_weights) * current_embeddings + \
                                    effective_weights * historical_context
            else:
                # Use single parameter for all nodes with confidence gating
                base_weight = torch.clamp(self.combination_weight, 0.0, 1.0)
                
                # Apply confidence gating to the base weight
                effective_weights = base_weight * confidence_scores
                
                enhanced_embeddings = (1 - effective_weights) * current_embeddings + \
                                    effective_weights * historical_context
        else:
            enhanced_embeddings = current_embeddings
        
        return enhanced_embeddings
    
    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        return self.base_model.discriminate(z, edge_index)
    
    def loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        return self.base_model.loss(z, pos_edge_index, neg_edge_index)
    
    def test(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[float, float]:
        return self.base_model.test(z, pos_edge_index, neg_edge_index)