"""
This module implements historical context extraction for temporal graph neural networks.
The primary purpose is to capture and process temporal patterns from node embeddings
across multiple timesteps, enabling models to leverage historical information for
improved predictions in dynamic graph environments.

Key Components:
- LSTM-based sequential processing of historical node embeddings
- Multi-head attention mechanism for temporal pattern recognition
- Context projection for dimensionality alignment and feature transformation
"""

import torch
import torch.nn as nn
from typing import List, Optional
from torch import Tensor

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically detect and configure the best available compute device
# Priority: CUDA GPU > Apple Metal Performance Shaders (MPS) > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# ============================================================================
# HISTORICAL CONTEXT EXTRACTION MODULE
# ============================================================================

class HistoricalContextExtractor(nn.Module):
    """
    Extracts and processes historical context from temporal sequences of node embeddings.
    
    This module processes sequences of node embeddings from previous timesteps to
    capture temporal patterns and dependencies. The architecture employs a three-stage
    processing pipeline:
    1. LSTM captures temporal dependencies and sequential patterns
    2. Multi-head attention identifies salient historical moments
    3. Linear projection aligns historical context with current embeddings
    
    This design is effective for social network evolution, trust dynamics, dynamic
    link prediction, and temporal node classification tasks.
    """
    
    def __init__(self, node_dim: int):
        """
        Initialize the Historical Context Extractor.
        
        Args:
            node_dim: Dimensionality of node embeddings from the graph neural network.
                     Should match the embedding size used in the main temporal GNN.
        
        Architecture choices:
            - LSTM hidden size = node_dim // 2: Creates information bottleneck for
              compact temporal representations
            - Single LSTM layer: Prevents overfitting on temporal sequences
            - 4 attention heads: Multiple perspectives on temporal importance
        """
        super().__init__()
        self.node_dim = node_dim
        
        # Add learnable recency bias parameters
        self.decay_factor = nn.Parameter(torch.tensor(0.7))  # How much to decay older timesteps
        self.recency_strength = nn.Parameter(torch.tensor(1.5))  # How strong the recency bias is
        
        # Process historical embeddings
        self.history_processor = nn.LSTM(
            input_size=node_dim,
            hidden_size=node_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Attention over historical timesteps
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=node_dim // 2,
            num_heads=4,
            batch_first=True
        )
        
        # Project historical context
        self.context_projection = nn.Linear(node_dim // 2, node_dim)
        
    def compute_recency_weights(self, num_timesteps: int) -> Tensor:
        """Compute recency-based weights for historical timesteps"""
        device = self.decay_factor.device
        
        # Create position indices (0 = oldest, num_timesteps-1 = most recent)
        positions = torch.arange(num_timesteps, dtype=torch.float, device=device)
        
        # Apply exponential decay: older timesteps get lower weights
        decay_weights = torch.pow(self.decay_factor, num_timesteps - 1 - positions)
        
        # Apply recency bias: boost recent timesteps
        recency_weights = torch.exp(positions / self.recency_strength)
        
        # Combine decay and recency
        combined_weights = decay_weights * recency_weights
        
        # Normalize to sum to 1
        normalized_weights = combined_weights / (combined_weights.sum() + 1e-8)
        
        return normalized_weights
    def forward(self, historical_embeddings: List[Tensor]) -> Optional[Tensor]:
        """
        Extract historical context from temporal sequence of node embeddings.
        
        Processing pipeline:
        1. Stack historical embeddings into sequence tensor
        2. Process through LSTM to capture sequential dependencies
        3. Apply multi-head attention to identify salient patterns
        4. Project to appropriate dimensionality for integration
        
        Args:
            historical_embeddings: List of node embedding tensors from previous timesteps.
                                  Each tensor has shape [num_nodes, node_dim].
                                  Ordered chronologically from oldest to newest.
        
        Returns:
            Historical context tensor [num_nodes, node_dim] or None if no history.
        """
        # Handle empty historical sequences
        if not historical_embeddings:
            return None
        
        num_nodes = historical_embeddings[0].size(0)
        num_timesteps = len(historical_embeddings)
        
        # Compute recency weights for each timestep
        recency_weights = self.compute_recency_weights(num_timesteps)
        
        # Apply recency weighting to historical embeddings
        weighted_embeddings = []
        for i, embedding in enumerate(historical_embeddings):
            weighted_embedding = embedding * recency_weights[i]
            weighted_embeddings.append(weighted_embedding)
        
        # Stack weighted historical embeddings: [num_nodes, seq_len, node_dim]
        historical_stack = torch.stack(weighted_embeddings, dim=1)
        
        # Process through LSTM
        lstm_out, _ = self.history_processor(historical_stack)
        
        # Apply temporal attention
        attended_history, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep as the context
        historical_context = attended_history[:, -1, :]  # [num_nodes, hidden_dim]
        
        # Project to node_dim
        historical_context = self.context_projection(historical_context)
        
        return historical_context