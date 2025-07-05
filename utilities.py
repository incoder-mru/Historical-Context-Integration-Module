"""
Comprehensive utility module for evaluating and visualizing temporal graph neural networks.

This module provides essential tools for assessing the performance of temporal GNN models,
particularly focusing on link prediction tasks. The utilities enable systematic comparison
between baseline and temporal models through various metrics and visualization techniques.

Key Features:
    - Precision@K evaluation for ranking-based link prediction assessment
    - Training dynamics visualization with loss curve analysis
    - Performance comparison charts with absolute and relative improvements
    - Robust device handling for CUDA, MPS, and CPU environments
    - Memory-efficient batch processing for large graphs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from torch import Tensor

# ============================================================================
# DEVICE CONFIGURATION AND HARDWARE OPTIMIZATION
# ============================================================================

# Automatically detect and configure the optimal compute device for tensor operations
# This prioritization ensures maximum performance across different hardware configurations:
# 1. CUDA GPU: Provides massive parallel processing for large graph computations
# 2. Apple MPS: Leverages Apple Silicon GPU acceleration for Mac systems  
# 3. CPU: Fallback option ensuring universal compatibility
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ============================================================================
# PRECISION-BASED RANKING EVALUATION METRICS
# ============================================================================

def calculate_precision_at_100(model, x, pos_edge_index, neg_edge_index, batch_size=10000, historical_embeddings=None):
    """
    Calculate Precision@100 metric for link prediction evaluation.
    
    Precision@K is a critical ranking-based metric that evaluates how well a model
    can identify true positive edges among its top-K predictions. This metric is
    particularly important for temporal graph learning because:
    
    1. Real-world applications often focus on top recommendations (e.g., friend suggestions)
    2. It provides insight into model confidence and ranking quality
    3. Unlike AUC, it directly measures performance on the most confident predictions
    4. It's robust to class imbalance common in graph link prediction tasks
    
    The evaluation process simulates realistic link prediction scenarios where
    models must rank potential edges and identify the most likely connections.
    
    Args:
        model: Either a PyTorch model object or callable function that generates embeddings.
               Supports both traditional models with .eval() method and lambda functions
               for flexible evaluation scenarios.
        x (Tensor): Node feature matrix [num_nodes, feature_dim]. Contains the input
                   features for all nodes in the graph at the current timestep.
        pos_edge_index (Tensor): Positive edge indices [2, num_pos_edges]. Represents
                               confirmed connections that should receive high scores.
        neg_edge_index (Tensor): Negative edge indices [2, num_neg_edges]. Represents
                               absent connections that should receive low scores.
        batch_size (int): Maximum batch size for processing. Currently unused but
                         reserved for memory-efficient large-scale evaluation.
        historical_embeddings (Optional[List[Tensor]]): Historical node embeddings from
                                                       previous timesteps for temporal models.
                                                       None for static baseline models.
    
    Returns:
        float: Precision@100 score between 0.0 and 1.0, where 1.0 indicates perfect
               ranking with all top-100 predictions being true positive edges.
    
    Implementation Details:
        - Uses dot product similarity for edge scoring (cosine similarity without normalization)
        - Handles both temporal and static models through conditional parameter passing
        - Provides detailed logging for debugging and analysis
        - Gracefully handles cases with insufficient test edges
    """
    # Set model to evaluation mode if it's a standard PyTorch model
    # Lambda functions and custom callables don't have eval() method
    if hasattr(model, 'eval'):
        model.eval()
    
    # Ensure consistent device placement for all tensors
    device = x.device
    
    # Disable gradient computation for evaluation efficiency and memory conservation
    with torch.no_grad():
        # Generate node embeddings using appropriate model interface
        # Temporal models require historical context, baseline models don't
        if historical_embeddings is not None:
            # Temporal model: incorporate historical information for richer representations
            z = model(x, pos_edge_index, neg_edge_index, historical_embeddings)
        else:
            # Baseline model: use only current timestep information
            z = model(x, pos_edge_index, neg_edge_index)
        
        # Construct comprehensive test set from both positive and negative edges
        # This creates a realistic evaluation scenario with balanced true/false samples
        pos_edges = pos_edge_index.T  # Shape: [num_pos, 2] - confirmed connections
        neg_edges = neg_edge_index.T  # Shape: [num_neg, 2] - absent connections
        
        # Combine positive and negative edges into unified test set
        test_edges = torch.cat([pos_edges, neg_edges], dim=0)
        
        # Create corresponding binary labels: 1 for positive edges, 0 for negative edges
        true_labels = torch.cat([
            torch.ones(pos_edges.size(0), device=device),   # Positive edge labels
            torch.zeros(neg_edges.size(0), device=device)   # Negative edge labels
        ])
        
        # Log evaluation statistics for monitoring and debugging
        print(f"   Test edges: {len(test_edges)}, Labels: {true_labels.sum().item():.0f} pos, {(len(true_labels) - true_labels.sum()).item():.0f} neg")
        
        # Calculate edge scores using dot product similarity
        # Dot product captures how well two node embeddings align in the learned space
        # Higher scores indicate stronger predicted connections
        edge_scores = []
        for edge in test_edges:
            # Extract node indices for current edge
            i, j = edge[0].item(), edge[1].item()
            
            # Compute similarity score between node embeddings
            # Dot product provides unnormalized similarity measure
            score = torch.dot(z[i], z[j]).item()
            edge_scores.append(score)
        
        # Convert to tensor for efficient PyTorch operations
        edge_scores = torch.tensor(edge_scores, device=device)
        
        # Log score distribution for model behavior analysis
        print(f"   Score range: {edge_scores.min().item():.4f} to {edge_scores.max().item():.4f}")
        
        # Calculate Precision@100 metric
        k = 100  # Number of top predictions to evaluate
        
        # Handle edge case where insufficient test edges are available
        if k > len(edge_scores):
            precision_100 = 0.0
            print(f"   P@100: Not enough edges ({len(edge_scores)} < {k})")
            return precision_100
        
        # Identify top-k highest scoring edges (model's most confident predictions)
        # largest=True ensures we get the highest scores (strongest predicted connections)
        _, top_k_indices = torch.topk(edge_scores, k, largest=True)
        
        # Extract true labels for the top-k predictions
        top_k_labels = true_labels[top_k_indices]
        
        # Calculate precision as fraction of correct predictions in top-k
        # This measures how many of the model's top predictions are actually correct
        precision_100 = top_k_labels.float().mean().item()
        
        # Log final results with detailed breakdown
        print(f"   P@100: {precision_100:.4f} ({top_k_labels.sum().item():.0f}/100 correct)")
        
        return precision_100

# ============================================================================
# TRAINING DYNAMICS VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_loss_curves(baseline_metrics, temporal_metrics, title: str = None):
    """
    Visualize training loss progression for baseline and temporal models.
    
    Training loss curves provide critical insights into model learning dynamics:
    1. Convergence behavior: How quickly models reach optimal performance
    2. Stability assessment: Whether training exhibits smooth or erratic patterns
    3. Comparative analysis: Which architecture learns more effectively
    4. Overfitting detection: Identifying potential generalization issues
    
    This visualization is essential for temporal GNN research because temporal
    models often exhibit different learning dynamics than static baselines due to
    their increased complexity and historical information processing.
    
    Args:
        baseline_metrics (List[float]): Training loss values for baseline model
                                       across epochs. Should contain loss values
                                       from each training epoch in sequential order.
        temporal_metrics (List[float]): Training loss values for temporal model
                                       across epochs. Must have same length as
                                       baseline_metrics for proper comparison.
        title (str, optional): Custom title prefix for the plot. If None, uses
                              generic title. Useful for distinguishing different
                              experimental conditions or datasets.
    
    Visual Elements:
        - Red line: Baseline model performance for easy identification
        - Blue line: Temporal model performance with distinct color contrast
        - Grid overlay: Facilitates precise value reading and trend analysis
        - Legend: Clear model identification for interpretation
        - Professional styling: Publication-ready appearance with proper spacing
    """
    # Create figure with appropriate size for detailed analysis
    plt.figure(figsize=(10, 6))
    
    # Generate epoch indices for x-axis (0, 1, 2, ..., num_epochs-1)
    epochs = list(range(len(baseline_metrics)))
    
    # Plot baseline model loss curve in red for clear distinction
    # Thicker line (linewidth=2) ensures visibility in publications
    plt.plot(epochs, baseline_metrics, label='Baseline', color='red', linewidth=2)
    
    # Plot temporal model loss curve in blue for contrast
    plt.plot(epochs, temporal_metrics, label='Temporal', color='blue', linewidth=2)
    
    # Configure axis labels with clear, descriptive text
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Set title with conditional formatting for flexibility
    plt.title(f"{title} Training Loss Comparison" if title else "Training Loss Comparison")
    
    # Add legend for model identification
    plt.legend()
    
    # Enable grid for precise value reading with subtle transparency
    plt.grid(True, alpha=0.3)
    
    # Optimize layout to prevent label cutoff and ensure professional appearance
    plt.tight_layout()
    plt.show()


def plot_loss_difference(baseline_metrics, temporal_metrics, title=None):
    """
    Visualize the performance gap between temporal and baseline models over training.
    
    Loss difference analysis reveals when and how much temporal models outperform
    baselines during training. This metric is crucial for understanding:
    
    1. Training efficiency: How quickly temporal advantages emerge
    2. Consistency: Whether improvements are sustained throughout training
    3. Magnitude: Quantifying the actual benefit of temporal information
    4. Convergence patterns: How the performance gap evolves over time
    
    Negative values indicate temporal model superiority (lower loss), while
    positive values suggest baseline advantages. The zero line serves as a
    reference point for equal performance.
    
    Args:
        baseline_metrics (List[float]): Training loss sequence for baseline model.
                                       Each value represents loss at corresponding epoch.
        temporal_metrics (List[float]): Training loss sequence for temporal model.
                                       Must align with baseline_metrics epochs.
        title (str, optional): Plot title prefix for experimental context identification.
    
    Interpretation Guide:
        - Negative values: Temporal model performs better (lower loss)
        - Positive values: Baseline model performs better
        - Zero crossing: Point where models achieve equal performance
        - Trend direction: Whether gap is widening or narrowing over time
    """
    # Initialize figure with standard dimensions for detailed analysis
    plt.figure(figsize=(10, 6))
    
    # Create epoch sequence for x-axis alignment
    epochs = list(range(len(baseline_metrics)))
    
    # Calculate pointwise loss difference: temporal_loss - baseline_loss
    # Negative values indicate temporal model superiority (desired outcome)
    loss_diff = [t - b for b, t in zip(baseline_metrics, temporal_metrics)]
    
    # Plot difference curve in green to represent "improvement" theme
    plt.plot(epochs, loss_diff, color='green', linewidth=2)
    
    # Add horizontal reference line at zero for equal performance identification
    # Dashed style and transparency make it subtle but visible
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Configure descriptive axis labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference (Temporal - Baseline)')
    
    # Set contextual title
    plt.title(f"{title} Loss Improvement Over Training" if title else "Loss Improvement Over Training")
    
    # Enable grid for precise value interpretation
    plt.grid(True, alpha=0.3)
    
    # Optimize layout presentation
    plt.tight_layout()
    plt.show()

# ============================================================================
# COMPREHENSIVE PERFORMANCE COMPARISON VISUALIZATIONS
# ============================================================================

def plot_auc_f1_comparison(baseline_results, temporal_results, title: str = None):
    """
    Compare AUC and F1-Score performance between baseline and temporal models.
    
    AUC and F1-Score represent complementary evaluation perspectives:
    
    AUC (Area Under ROC Curve):
    - Measures overall ranking quality across all decision thresholds
    - Robust to class imbalance, common in link prediction tasks
    - Indicates how well models separate positive and negative edges
    - Higher values (closer to 1.0) indicate better discrimination ability
    
    F1-Score:
    - Harmonic mean of precision and recall at optimal threshold
    - Balances false positive and false negative considerations
    - More sensitive to classification threshold selection
    - Provides insight into practical deployment performance
    
    This dual-metric visualization is essential for temporal GNN evaluation because
    different applications may prioritize ranking quality vs. classification accuracy.
    
    Args:
        baseline_results (dict): Performance dictionary containing 'auc' and 'f1' keys
                                with corresponding float values for baseline model.
        temporal_results (dict): Performance dictionary containing 'auc' and 'f1' keys
                                with corresponding float values for temporal model.
        title (str, optional): Plot title prefix for experimental context labeling.
    
    Visual Design:
        - Side-by-side bars: Easy comparison between models for each metric
        - Color coding: Light coral (baseline) vs. light blue (temporal) for distinction
        - Value annotations: Precise scores displayed on bars for exact comparison
        - Professional styling: Publication-ready appearance with clear formatting
    """
    # Create appropriately sized figure for detailed metric comparison
    plt.figure(figsize=(10, 6))
    
    # Define metrics for evaluation and extract corresponding scores
    metrics = ['AUC', 'F1']
    baseline_scores = [baseline_results['auc'], baseline_results['f1']]
    temporal_scores = [temporal_results['auc'], temporal_results['f1']]
    
    # Configure bar positioning for side-by-side comparison
    x = np.arange(len(metrics))  # Metric positions
    width = 0.35  # Bar width for proper spacing
    
    # Create baseline bars with light coral color for warm, distinctive appearance
    bars1 = plt.bar(x - width/2, baseline_scores, width, label='Baseline', 
                   color='lightcoral', alpha=0.8)
    
    # Create temporal bars with light blue color for cool, contrasting appearance
    bars2 = plt.bar(x + width/2, temporal_scores, width, label='Temporal', 
                   color='lightblue', alpha=0.8)
    
    # Add precise value annotations on baseline bars for exact comparison
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Add precise value annotations on temporal bars
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Configure axis labels and formatting
    plt.ylabel('Score')
    plt.title(f"{title} Performance Comparison (AUC & F1)" if title else "Performance Comparison (AUC & F1)")
    plt.xticks(x, metrics)
    plt.legend()
    
    # Add subtle grid for value estimation
    plt.grid(True, alpha=0.3)
    
    # Optimize layout for presentation
    plt.tight_layout()
    plt.show()


def plot_precision_100_comparison(baseline_results, temporal_results, title: str = None):
    """
    Visualize Precision@100 comparison between baseline and temporal models.
    
    Precision@100 is a specialized ranking metric crucial for temporal graph learning:
    
    Why Precision@100 Matters:
    1. Real-world relevance: Most applications focus on top recommendations
    2. Practical deployment: Users typically only consider highest-confidence predictions
    3. Quality assessment: Measures model's ability to prioritize true positives
    4. Threshold independence: Evaluates ranking quality without classification cutoffs
    
    This metric is particularly important for temporal GNNs because historical
    information should improve the model's confidence in identifying the most
    likely connections, which this metric directly measures.
    
    Args:
        baseline_results (dict): Results dictionary containing 'precision_at_100' key
                                with float value representing baseline model performance.
        temporal_results (dict): Results dictionary containing 'precision_at_100' key
                                with float value representing temporal model performance.
        title (str, optional): Plot title prefix for experimental identification.
    
    Interpretation:
        - Values range from 0.0 to 1.0
        - Higher values indicate better ranking quality
        - 1.0 means all top-100 predictions are correct
        - Significant improvements (>0.05) suggest meaningful temporal advantages
    """
    # Create focused figure for single-metric comparison
    plt.figure(figsize=(8, 6))
    
    # Extract precision values for comparison
    precision_baseline = baseline_results['precision_at_100']
    precision_temporal = temporal_results['precision_at_100']
    
    # Configure bar positioning for side-by-side display
    width = 0.35
    x = [0]  # Single bar group for focused comparison
    
    # Create baseline bar with consistent color scheme
    bars1 = plt.bar(x[0] - width/2, precision_baseline, width, label='Baseline', 
                   color='lightcoral', alpha=0.8)
    
    # Create temporal bar with contrasting color
    bars2 = plt.bar(x[0] + width/2, precision_temporal, width, label='Temporal', 
                   color='lightblue', alpha=0.8)
    
    # Add precise value annotation for baseline bar
    plt.annotate(f'{precision_baseline:.3f}',
                xy=(bars1[0].get_x() + bars1[0].get_width() / 2, precision_baseline),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')
    
    # Add precise value annotation for temporal bar
    plt.annotate(f'{precision_temporal:.3f}',
                xy=(bars2[0].get_x() + bars2[0].get_width() / 2, precision_temporal),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')
    
    # Configure axis and title formatting
    plt.ylabel('Precision@100')
    plt.title(f"{title} Precision@100 Comparison" if title else "Precision@100 Comparison")
    plt.xticks([0], ['Precision@100'])
    plt.legend()
    
    # Add subtle grid for value estimation
    plt.grid(True, alpha=0.3)
    
    # Optimize layout
    plt.tight_layout()
    plt.show()

# ============================================================================
# IMPROVEMENT ANALYSIS AND QUANTIFICATION VISUALIZATIONS
# ============================================================================

def plot_absolute_improvements(baseline_results, temporal_results, title: str = None):
    """
    Visualize absolute performance improvements from temporal modeling.
    
    Absolute improvement analysis quantifies the raw benefit of incorporating
    temporal information into graph neural networks. This visualization answers:
    
    1. Which metrics benefit most from temporal modeling?
    2. What is the magnitude of improvement across different evaluation criteria?
    3. Are there any metrics where temporal modeling performs worse?
    4. How consistent are the improvements across different evaluation aspects?
    
    The color-coded visualization immediately highlights positive (green) and
    negative (red) changes, providing quick assessment of temporal model benefits.
    
    Args:
        baseline_results (dict): Baseline performance containing 'auc', 'f1', and
                                'precision_at_100' keys with corresponding float values.
        temporal_results (dict): Temporal performance with same key structure as
                                baseline_results for consistent comparison.
        title (str, optional): Plot title prefix for experimental context identification.
    
    Interpretation Guide:
        - Green bars: Positive improvements (temporal model better)
        - Red bars: Negative changes (baseline model better)  
        - Bar height: Magnitude of absolute change
        - Values near zero: Minimal difference between approaches
    """
    # Create standard figure for multi-metric comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate absolute improvements across all metrics
    # Positive values indicate temporal model superiority
    auc_improvement = temporal_results['auc'] - baseline_results['auc']
    f1_improvement = temporal_results['f1'] - baseline_results['f1']
    precision_improvement = temporal_results['precision_at_100'] - baseline_results['precision_at_100']
    
    # Define metrics and corresponding improvement values
    metrics = ['AUC', 'F1', 'Precision@100']
    improvements = [auc_improvement, f1_improvement, precision_improvement]
    
    # Color-code bars based on improvement direction
    # Green for positive improvements, red for negative changes
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    # Create improvement bars with dynamic coloring
    bars = plt.bar(metrics, improvements, color=colors, alpha=0.7)
    
    # Add precise improvement values as annotations on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        # Position annotation at bar center, with special handling for zero-height bars
        plt.annotate(f'{imp:+.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height/2 if height != 0 else 0.001),
                    ha='center', va='center', fontweight='bold', color='white')
    
    # Configure axis labels and title
    plt.ylabel('Absolute Improvement')
    plt.title(f"{title} Performance Gains (Temporal - Baseline)" if title else "Performance Gains (Temporal - Baseline)")
    
    # Add horizontal reference line at zero for neutral performance
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Enable grid for precise value reading
    plt.grid(True, alpha=0.3)
    
    # Optimize layout
    plt.tight_layout()
    plt.show()


def plot_percentage_improvements(baseline_results, temporal_results, title=None):
    """
    Visualize percentage-based performance improvements from temporal modeling.
    
    Percentage improvements provide relative context for temporal model benefits,
    which is crucial for understanding the practical significance of changes:
    
    1. Relative impact: A 0.01 improvement in AUC might be more significant than
       a 0.10 improvement in Precision@100, depending on baseline performance
    2. Scaling effects: Higher baseline scores make absolute improvements harder
    3. Practical significance: Percentage changes help assess real-world impact
    4. Cross-metric comparison: Enables fair comparison across different scales
    
    This analysis is essential for temporal GNN research because it contextualizes
    improvements relative to the difficulty of achieving them.
    
    Args:
        baseline_results (dict): Baseline performance metrics dictionary with 'auc',
                                'f1', and 'precision_at_100' keys and float values.
        temporal_results (dict): Temporal model performance with identical structure
                                to baseline_results for percentage calculation.
        title (str, optional): Plot title prefix for experimental context.
    
    Mathematical Formula:
        percentage_improvement = (temporal_score - baseline_score) / baseline_score * 100
    
    Interpretation:
        - Positive percentages: Temporal model improvements
        - Negative percentages: Baseline model advantages
        - Large percentages (>10%): Substantial improvements
        - Small percentages (<5%): Marginal differences
    """
    # Create figure for percentage-based comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate absolute improvements as foundation for percentage calculation
    auc_improvement = temporal_results['auc'] - baseline_results['auc']
    f1_improvement = temporal_results['f1'] - baseline_results['f1']
    precision_improvement = temporal_results['precision_at_100'] - baseline_results['precision_at_100']
    
    # Calculate percentage improvements with zero-division protection
    # Handle edge cases where baseline performance might be zero
    auc_improvement_pct = (auc_improvement / baseline_results['auc'] * 100) if baseline_results['auc'] > 0 else 0
    f1_improvement_pct = (f1_improvement / baseline_results['f1'] * 100) if baseline_results['f1'] > 0 else 0
    precision_improvement_pct = (precision_improvement / baseline_results['precision_at_100'] * 100) if baseline_results['precision_at_100'] > 0 else 0
    
    # Organize metrics and corresponding percentage improvements
    metrics = ['AUC', 'F1', 'Precision@100']
    improvements_pct = [auc_improvement_pct, f1_improvement_pct, precision_improvement_pct]
    
    # Apply color coding based on improvement direction
    colors = ['green' if x > 0 else 'red' for x in improvements_pct]
    
    # Create percentage improvement bars
    bars = plt.bar(metrics, improvements_pct, color=colors, alpha=0.7)
    
    # Add percentage annotations on bars for precise values
    for bar, pct in zip(bars, improvements_pct):
        height = bar.get_height()
        # Position annotation at bar center with special zero-height handling
        plt.annotate(f'{pct:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height/2 if height != 0 else 0.5),
                    ha='center', va='center', fontweight='bold', color='white')
    
    # Configure axis labels and title
    plt.ylabel('Percentage Improvement (%)')
    plt.title(f"{title} Relative Performance Gains" if title else "Relative Performance Gains")
    
    # Add horizontal reference line for neutral performance
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Enable grid for value interpretation
    plt.grid(True, alpha=0.3)
    
    # Optimize layout presentation
    plt.tight_layout()
    plt.show()