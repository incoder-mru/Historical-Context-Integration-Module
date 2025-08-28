# Historical Context Integration Module (HCIM)

A temporally-weighted extension of the static signed graph neural networks for signed link prediction in dynamic networks.

## Overview

This implementation enhances the SE-SGformer (Signed Graph Transformer) with historical context awareness, enabling improved link prediction performance on temporal signed networks. The model integrates LSTM-based sequence modeling and temporal attention to adaptively aggregate node embeddings across time while preserving interpretability.

## Key Features

- **Historical Context Integration**: Uses LSTM and temporal attention to leverage previous timesteps
- **Recency Bias**: Applies learnable decay factors to weight recent information more heavily
- **Confidence Gating**: Adaptive mechanism to control the influence of historical context
- **Memory Efficient**: Batch processing for large-scale networks
- **Comprehensive Evaluation**: Multiple metrics including AUC, F1, and Precision@100

## Architecture

### Components

1. **Original SE-SGformer**: Base model with centrality encoding, spatial features, and multi-head attention
2. **Historical Context Extractor**: LSTM-based processor with temporal attention and recency weighting
3. **Temporal SE-SGformer**: Enhanced model combining current and historical embeddings with confidence gating

### Key Improvements

- **Temporal Modeling**: Processes sequences of historical embeddings
- **Adaptive Weights**: Either fixed or MLP-based combination strategies
- **Confidence Assessment**: Gates historical information based on reliability

## Installation

### Requirements

```bash
pip install torch torch-geometric numpy scipy pandas matplotlib scikit-learn
```

### Dependencies

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- NumPy, SciPy, Pandas
- Matplotlib (for visualization)
- scikit-learn (for metrics)

## Usage

### Basic Usage

```python
from temporal_sgformer import compare_approaches

# Run comparison on Bitcoin OTC dataset
results = compare_approaches(
    file_path="bitcoin_otc.csv.gz",
    target_timestep=-1,  # Use last timestep as target
    num_time_bins=6,     # Split data into 6 temporal bins
    epochs=50,           # Training epochs
    title="Bitcoin OTC Analysis"
)
```

### Configuration Options

```python
# Model configuration
args = Args(
    num_layers=2,              # Number of transformer layers
    num_heads=4,               # Multi-head attention heads
    node_dim=128,              # Node embedding dimension
    max_degree=20,             # Maximum node degree for encoding
    use_adaptive_weights=True,  # Use MLP-based combination weights
    base_weights=0.3           # Fixed combination weight (if not adaptive)
)

# Create models
baseline_model = SE_SGformer(args)
temporal_model = Temporal_SE_SGformer(args)
```

### Data Format

The code expects CSV data with columns:
- `source`: Source node ID
- `target`: Target node ID  
- `rating`: Edge weight/rating (positive/negative)
- `time`: Timestamp

**Bitcoin OTC Dataset Format:**
The Bitcoin OTC dataset follows this format where each line represents one rating:
```
SOURCE,TARGET,RATING,TIME
```
- `SOURCE`: ID of the source node (rater)
- `TARGET`: ID of the target node (ratee)  
- `RATING`: Rating score from -10 (total distrust) to +10 (total trust)
- `TIME`: Unix timestamp of when the rating was given

Example:
```csv
1,2,8,1237462018
2,3,-5,1237465108
1,3,10,1237467209
```

The model converts ratings to binary signs: positive ratings (>0) become +1, negative ratings (<0) become -1.

### Evaluation Metrics

The framework evaluates models using:

1. **AUC (Area Under Curve)**: Binary classification performance
2. **F1 Score**: Harmonic mean of precision and recall
3. **Precision@100**: Precision of top-100 predicted links

## Key Functions

### Data Loading
```python
timesteps, num_nodes = load_bitcoin_dataset_timesteps(
    file_path="data.csv.gz", 
    num_time_bins=10
)
```

### Model Training
```python
# Train baseline model
baseline_model = SE_SGformer(args)
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

for epoch in range(epochs):
    z = baseline_model(x, pos_edge_index, neg_edge_index)
    loss = baseline_model.loss(z, pos_edge_index, neg_edge_index)
    loss.backward()
    optimizer.step()
```

### Historical Context Extraction
```python
# Extract embeddings from previous timesteps
historical_embeddings = []
for hist_data in historical_timesteps:
    z_hist = model(x_hist, pos_edges_hist, neg_edges_hist)
    historical_embeddings.append(z_hist)

# Use in temporal model
z_temporal = temporal_model(x, pos_edges, neg_edges, historical_embeddings)
```

## Visualization

The framework generates comprehensive visualizations:

1. **Training Loss Curves**: Comparison of baseline vs temporal training
2. **Loss Difference**: Training improvement over epochs
3. **Performance Metrics**: AUC, F1, and Precision@100 comparison
4. **Improvement Analysis**: Absolute and percentage gains

## Advanced Configuration

### Temporal Parameters
```python
# Configure historical context extractor
context_extractor = HistoricalContextExtractor(node_dim=128)
context_extractor.decay_factor = 0.7      # Decay for older timesteps
context_extractor.recency_strength = 1.5  # Recency bias strength
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request


## Acknowledgments

- **Original SE-SGformer**: This work builds upon the SE-SGformer (Self-Explainable Signed Graph Transformer) framework proposed by Liu et al. in "Self-Explainable Graph Transformer for Link Sign Prediction" (arXiv:2408.08754, 2024). We extend their original model with temporal context awareness while preserving the core architectural innovations.
- **Bitcoin OTC Dataset**: We use the Bitcoin OTC trust weighted signed network from the Stanford Network Analysis Project (SNAP). This dataset was introduced by Kumar et al. in "Edge weight prediction in weighted signed networks" (ICDM 2016) and represents a who-trusts-whom network of Bitcoin traders with ratings from -10 to +10.
