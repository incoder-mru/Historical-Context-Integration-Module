"""
This module implements a comprehensive evaluation framework for signed network 
link prediction heuristics. Signed networks contain both positive and negative 
relationships, requiring specialized approaches that account for relationship polarity.

The framework evaluates three categories of heuristics:
1. Baseline heuristics: Simple majority class prediction for comparison
2. Signed-aware heuristics: Traditional methods adapted for signed networks
3. Signed-specific heuristics: Methods designed specifically for signed networks

Key Features:
- Comprehensive evaluation metrics (AUC, F1, Precision@K)
- Visual performance comparison with type-based categorization
- Support for temporal graph analysis across multiple timesteps
- Statistical analysis of network properties and performance patterns
"""

# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch
from data_loader import load_dataset_timesteps

def test_signed_heuristics(file_path: str, target_timestep: int = -1, num_time_bins: int = 8,
                                            show_plot: bool = True, k_values: list = [100], title: str = None):
    """
    Comprehensive evaluation of signed network link prediction heuristics.
    
    This function evaluates multiple heuristic approaches for predicting positive vs 
    negative links in signed networks. Unlike unsigned networks where only link existence
    is predicted, signed networks require predicting relationship polarity.
    
    The evaluation framework tests heuristics across three categories:
    - Baseline: Simple majority class prediction for performance floor
    - Signed-aware: Traditional heuristics adapted to consider edge signs
    - Signed-specific: Methods designed specifically for signed network properties
    
    Args:
        file_path: Path to temporal graph dataset file containing signed edges
        target_timestep: Timestep to evaluate (-1 for last timestep)
        num_time_bins: Total number of temporal bins in the dataset
        show_plot: Whether to generate performance visualization plots
        k_values: List of k values for Precision@K evaluation
        title: Custom title prefix for plots and output
        
    Returns:
        Dictionary containing:
        - Performance metrics for all heuristics
        - Best performing heuristic identification
        - Network statistics and structural properties
        - Comparative analysis by heuristic type
    """
    print("="*80)
    print("SIGNED NETWORK HEURISTICS")
    print("="*80)

    def calculate_precision_at_k_heuristic(scores, true_labels, k_values):
        """
        Calculate Precision@K metric for heuristic performance evaluation.
        
        Precision@K measures how many of the top-k highest scoring predictions
        are actually positive edges. This metric is crucial for signed networks
        where users care most about the highest-confidence positive predictions.
        
        Args:
            scores: Tensor of heuristic scores for test edges
            true_labels: Binary tensor (1=positive edge, 0=negative edge)
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary mapping 'precision_at_k' to calculated precision values
        """
        precision_results = {}
        
        for k in k_values:
            if k > len(scores):
                precision_results[f'precision_at_{k}'] = 0.0
                continue
            
            # Get top-k highest scoring edges
            _, top_k_indices = torch.topk(scores, k, largest=True)
            top_k_labels = true_labels[top_k_indices]
            
            # Calculate precision
            precision_k = top_k_labels.float().mean().item()
            precision_results[f'precision_at_{k}'] = precision_k
        
        return precision_results
    
    # Load temporal graph data with multiple timesteps
    timesteps, num_nodes = load_dataset_timesteps(file_path, num_time_bins)
    
    if len(timesteps) < 1:
        print("Need at least 1 timestep")
        return None
    
    # Select target timestep for evaluation
    if target_timestep == -1:
        target_timestep = len(timesteps) - 1
    
    target_data = timesteps[target_timestep]
    
    print(f"\nTest setup:")
    print(f"  Target timestep: {target_data['timestep']}")
    print(f"  Positive edges: {target_data['num_pos']}")
    print(f"  Negative edges: {target_data['num_neg']}")
    print(f"  Total edges: {target_data['num_edges']}")
    print(f"  Total nodes: {num_nodes}")
    
    # ========================================================================
    # DATA PREPARATION: Create test set and adjacency matrices
    # ========================================================================
    
    # Create test edges (combine positive and negative for evaluation)
    pos_edges = target_data['pos_edge_index'].T  # [num_pos, 2]
    neg_edges = target_data['neg_edge_index'].T  # [num_neg, 2]
    
    # True labels: 1 for positive edges, 0 for negative edges
    test_edges = torch.cat([pos_edges, neg_edges], dim=0)
    true_labels = torch.cat([
        torch.ones(pos_edges.size(0), device=pos_edges.device),
        torch.zeros(neg_edges.size(0), device=neg_edges.device)
    ])
    
    print(f"  Test edges: {len(test_edges)}")
    
    # Build adjacency matrices for different heuristic calculations
    
    # Unsigned adjacency matrix: all edges have weight 1 (ignores signs)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj_matrix[target_data['pos_edge_index'][0], target_data['pos_edge_index'][1]] = 1
    adj_matrix[target_data['pos_edge_index'][1], target_data['pos_edge_index'][0]] = 1
    adj_matrix[target_data['neg_edge_index'][0], target_data['neg_edge_index'][1]] = 1
    adj_matrix[target_data['neg_edge_index'][1], target_data['neg_edge_index'][0]] = 1
    
    # Signed adjacency matrix: positive edges = +1, negative edges = -1
    signed_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    signed_adj[target_data['pos_edge_index'][0], target_data['pos_edge_index'][1]] = 1
    signed_adj[target_data['pos_edge_index'][1], target_data['pos_edge_index'][0]] = 1
    signed_adj[target_data['neg_edge_index'][0], target_data['neg_edge_index'][1]] = -1
    signed_adj[target_data['neg_edge_index'][1], target_data['neg_edge_index'][0]] = -1
    
    # Positive-only adjacency matrix: only positive edges have weight 1
    pos_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    pos_adj[target_data['pos_edge_index'][0], target_data['pos_edge_index'][1]] = 1
    pos_adj[target_data['pos_edge_index'][1], target_data['pos_edge_index'][0]] = 1
    
    # ========================================================================
    # DEGREE CALCULATIONS: Multiple degree centrality measures
    # ========================================================================
    
    # Calculate various degree measures for heuristic computations
    all_edge_index = torch.cat([target_data['pos_edge_index'], target_data['neg_edge_index']], dim=1)
    total_degrees = torch.bincount(all_edge_index.flatten(), minlength=num_nodes).float()
    pos_degrees = torch.bincount(target_data['pos_edge_index'].flatten(), minlength=num_nodes).float()
    neg_degrees = torch.bincount(target_data['neg_edge_index'].flatten(), minlength=num_nodes).float()
    signed_degrees = pos_degrees - neg_degrees  # Net positive degree (status measure)
    
    # Initialize results tracking for all heuristics
    all_results = []
    
    # ========================================================================
    # BASELINE HEURISTIC: Majority Class Prediction
    # ========================================================================
    print(f"\n{'='*60}")
    print("Majority Class Prediction")
    print("="*60)
    
    # Simple baseline: always predict the majority class (positive or negative)
    # This provides a performance floor - any useful heuristic should beat this
    pos_ratio = target_data['num_pos'] / target_data['num_edges']
    majority_class = 1 if pos_ratio > 0.5 else 0
    
    majority_predictions = torch.full((len(test_edges),), float(majority_class))
    
    # Add small noise to enable AUC calculation (AUC requires score variation)
    if len(np.unique(true_labels.cpu().numpy())) > 1:
        majority_predictions_noisy = majority_predictions + torch.randn_like(majority_predictions) * 1e-6
        majority_auc = roc_auc_score(true_labels.cpu().numpy(), majority_predictions_noisy.cpu().numpy())
    else:
        majority_auc = 0.5
    
    majority_f1 = f1_score(true_labels.cpu().numpy(), majority_predictions.cpu().numpy())
    majority_precision_k = calculate_precision_at_k_heuristic(majority_predictions_noisy, true_labels, k_values)
    
    print(f"Majority Class: AUC = {majority_auc:.4f}, F1 = {majority_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {majority_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'majority_class',
        'description': f'Always predict majority class ({"positive" if majority_class == 1 else "negative"})',
        'auc': majority_auc,
        'f1': majority_f1,
        'type': 'baseline'
    }
    result_dict.update(majority_precision_k)
    all_results.append(result_dict)

    # ========================================================================
    # SIGNED PREFERENTIAL ATTACHMENT
    # ========================================================================
    print(f"\n{'='*60}")
    print("Signed Preferential Attachment")
    print("="*60)
    
    # Hypothesis: Nodes with high positive degree prefer to form positive connections
    # This extends traditional preferential attachment to signed networks by considering
    # only positive degree when calculating attachment probabilities
    source_pos_degrees = pos_degrees[test_edges[:, 0]]
    target_pos_degrees = pos_degrees[test_edges[:, 1]]
    pos_degree_products = source_pos_degrees * target_pos_degrees
    
    # Normalize scores to [0,1] range for consistent comparison
    if pos_degree_products.max() > pos_degree_products.min():
        signed_attachment_predictions = (pos_degree_products - pos_degree_products.min()) / (pos_degree_products.max() - pos_degree_products.min())
    else:
        signed_attachment_predictions = torch.full_like(pos_degree_products, 0.5)
    
    signed_attachment_binary = (signed_attachment_predictions > signed_attachment_predictions.median()).float()
    signed_attachment_auc = roc_auc_score(true_labels.cpu().numpy(), signed_attachment_predictions.cpu().numpy())
    signed_attachment_f1 = f1_score(true_labels.cpu().numpy(), signed_attachment_binary.cpu().numpy())
    signed_attachment_precision_k = calculate_precision_at_k_heuristic(signed_attachment_predictions, true_labels, k_values)
    
    print(f"Signed Preferential Attachment: AUC = {signed_attachment_auc:.4f}, F1 = {signed_attachment_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {signed_attachment_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'signed_preferential_attachment',
        'description': 'High positive-degree nodes prefer positive connections',
        'auc': signed_attachment_auc,
        'f1': signed_attachment_f1,
        'type': 'signed_specific'
    }
    result_dict.update(signed_attachment_precision_k)
    all_results.append(result_dict)

    # ========================================================================
    # POSITIVE COMMON NEIGHBORS (SIGNED-AWARE)
    # ========================================================================
    print(f"\n{'='*60}")
    print("Positive Common Neighbors (Signed-Aware)")
    print("="*60)
    
    # Hypothesis: Nodes with many common positive neighbors are likely to form positive edges
    # This adapts traditional common neighbors by only considering positive relationships,
    # based on social psychology principle that "friends of friends are friends"
    pos_triangle_counts = []
    for edge in test_edges:
        i, j = edge[0].item(), edge[1].item()
        # Count shared positive neighbors between nodes i and j
        common_pos_neighbors = (pos_adj[i] * pos_adj[j]).sum().item()
        pos_triangle_counts.append(common_pos_neighbors)
    
    pos_triangle_counts = torch.tensor(pos_triangle_counts, dtype=torch.float)
    
    # Normalize for consistent scoring
    if pos_triangle_counts.max() > pos_triangle_counts.min():
        pos_triangle_predictions = pos_triangle_counts / pos_triangle_counts.max()
    else:
        pos_triangle_predictions = torch.full_like(pos_triangle_counts, 0.5)
    
    pos_triangle_binary = (pos_triangle_predictions > pos_triangle_predictions.median()).float()
    pos_triangle_auc = roc_auc_score(true_labels.cpu().numpy(), pos_triangle_predictions.cpu().numpy())
    pos_triangle_f1 = f1_score(true_labels.cpu().numpy(), pos_triangle_binary.cpu().numpy())
    pos_triangle_precision_k = calculate_precision_at_k_heuristic(pos_triangle_predictions, true_labels, k_values)
    
    print(f"Positive Common Neighbors: AUC = {pos_triangle_auc:.4f}, F1 = {pos_triangle_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {pos_triangle_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'positive_common_neighbors',
        'description': 'More positive common neighbors = more likely positive edge',
        'auc': pos_triangle_auc,
        'f1': pos_triangle_f1,
        'type': 'signed_aware'
    }
    result_dict.update(pos_triangle_precision_k)
    all_results.append(result_dict)
    
    # ========================================================================
    # SIGNED STATUS THEORY
    # ========================================================================
    print(f"\n{'='*60}")
    print("Signed Status Theory")
    print("="*60)
    
    # Hypothesis: Nodes with similar status (net positive degree) form positive edges
    # Status theory suggests that social relationships form based on perceived status,
    # where status is measured by net positive connections minus negative connections
    node_status = signed_degrees
    
    status_scores = []
    for edge in test_edges:
        i, j = edge[0].item(), edge[1].item()
        
        # Calculate status difference between nodes
        status_diff = abs(node_status[i] - node_status[j])
        
        # Convert to similarity measure (inverse of difference)
        # Similar status nodes are more likely to form positive connections
        if node_status.max() > node_status.min():
            max_diff = (node_status.max() - node_status.min()).item()
            status_similarity = 1 - (status_diff / max_diff) if max_diff > 0 else 1
        else:
            status_similarity = 1
        
        status_scores.append(status_similarity)
    
    status_scores = torch.tensor(status_scores, dtype=torch.float)
    status_binary = (status_scores > status_scores.median()).float()
    
    # Handle case where all status scores are identical
    if status_scores.std() > 1e-6:
        status_auc = roc_auc_score(true_labels.cpu().numpy(), status_scores.cpu().numpy())
    else:
        status_auc = 0.5
    
    status_f1 = f1_score(true_labels.cpu().numpy(), status_binary.cpu().numpy())
    status_precision_k = calculate_precision_at_k_heuristic(status_scores, true_labels, k_values)
    
    print(f"Signed Status Theory: AUC = {status_auc:.4f}, F1 = {status_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {status_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'signed_status_theory',
        'description': 'Similar status nodes more likely to have positive edges',
        'auc': status_auc,
        'f1': status_f1,
        'type': 'signed_specific'
    }
    result_dict.update(status_precision_k)
    all_results.append(result_dict)
    

    # ========================================================================
    # SIGNED JACCARD SIMILARITY
    # ========================================================================
    print(f"\n{'='*60}")
    print("Signed Jaccard Similarity")
    print("="*60)
    
    # Hypothesis: Nodes with similar positive neighborhoods form positive connections
    # Jaccard similarity adapted for signed networks by considering only positive neighbors
    # High overlap in positive connections suggests similar preferences and likely positive edge
    signed_jaccard_scores = []
    for edge in test_edges:
        i, j = edge[0].item(), edge[1].item()
        
        # Get positive neighbors only for both nodes
        pos_neighbors_i = set(torch.where(pos_adj[i] > 0)[0].tolist())
        pos_neighbors_j = set(torch.where(pos_adj[j] > 0)[0].tolist())
        
        # Calculate Jaccard similarity: intersection / union
        intersection = len(pos_neighbors_i & pos_neighbors_j)
        union = len(pos_neighbors_i | pos_neighbors_j)
        
        jaccard = intersection / union if union > 0 else 0
        signed_jaccard_scores.append(jaccard)
    
    signed_jaccard_scores = torch.tensor(signed_jaccard_scores, dtype=torch.float)
    signed_jaccard_binary = (signed_jaccard_scores > signed_jaccard_scores.median()).float()
    
    # Handle case where all similarity scores are identical
    if signed_jaccard_scores.std() > 1e-6:
        signed_jaccard_auc = roc_auc_score(true_labels.cpu().numpy(), signed_jaccard_scores.cpu().numpy())
    else:
        signed_jaccard_auc = 0.5
    
    signed_jaccard_f1 = f1_score(true_labels.cpu().numpy(), signed_jaccard_binary.cpu().numpy())
    signed_jaccard_precision_k = calculate_precision_at_k_heuristic(signed_jaccard_scores, true_labels, k_values)
    
    print(f"Signed Jaccard Similarity: AUC = {signed_jaccard_auc:.4f}, F1 = {signed_jaccard_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {signed_jaccard_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'signed_jaccard_similarity',
        'description': 'Similar positive neighborhood overlap = more likely positive edge',
        'auc': signed_jaccard_auc,
        'f1': signed_jaccard_f1,
        'type': 'signed_aware'
    }
    result_dict.update(signed_jaccard_precision_k)
    all_results.append(result_dict)
    

    # ========================================================================
    # SIGNED ADAMIC-ADAR INDEX
    # ========================================================================
    print(f"\n{'='*60}")
    print("Signed Adamic-Adar Index")
    print("="*60)
    
    # Hypothesis: Common positive neighbors with low positive degree are more informative
    # Adamic-Adar gives higher weight to rare shared connections, adapted for signed networks
    # A rare positive mutual friend is more predictive than a highly connected one
    signed_adamic_adar_scores = []
    for edge in test_edges:
        i, j = edge[0].item(), edge[1].item()
        
        # Get positive neighbors for both nodes
        pos_neighbors_i = torch.where(pos_adj[i] > 0)[0]
        pos_neighbors_j = torch.where(pos_adj[j] > 0)[0]
        
        # Find common positive neighbors
        common = set(pos_neighbors_i.tolist()) & set(pos_neighbors_j.tolist())
        
        # Calculate Adamic-Adar score using positive degrees only
        score = 0
        for neighbor in common:
            pos_degree_neighbor = pos_degrees[neighbor].item()
            if pos_degree_neighbor > 1:  # Avoid log(0)
                # Weight by inverse log of positive degree - rare connections matter more
                score += 1 / np.log(pos_degree_neighbor)
        
        signed_adamic_adar_scores.append(score)
    
    signed_adamic_adar_scores = torch.tensor(signed_adamic_adar_scores, dtype=torch.float)
    
    # Normalize scores for consistent comparison
    if signed_adamic_adar_scores.max() > signed_adamic_adar_scores.min():
        signed_adamic_adar_predictions = (signed_adamic_adar_scores - signed_adamic_adar_scores.min()) / (signed_adamic_adar_scores.max() - signed_adamic_adar_scores.min())
    else:
        signed_adamic_adar_predictions = torch.full_like(signed_adamic_adar_scores, 0.5)
    
    signed_adamic_adar_binary = (signed_adamic_adar_predictions > signed_adamic_adar_predictions.median()).float()
    signed_adamic_adar_auc = roc_auc_score(true_labels.cpu().numpy(), signed_adamic_adar_predictions.cpu().numpy())
    signed_adamic_adar_f1 = f1_score(true_labels.cpu().numpy(), signed_adamic_adar_binary.cpu().numpy())
    signed_adamic_adar_precision_k = calculate_precision_at_k_heuristic(signed_adamic_adar_predictions, true_labels, k_values)
    
    print(f"Signed Adamic-Adar Index: AUC = {signed_adamic_adar_auc:.4f}, F1 = {signed_adamic_adar_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {signed_adamic_adar_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'signed_adamic_adar',
        'description': 'Weighted positive common neighbors (rare positive neighbors count more)',
        'auc': signed_adamic_adar_auc,
        'f1': signed_adamic_adar_f1,
        'type': 'signed_aware'
    }
    result_dict.update(signed_adamic_adar_precision_k)
    all_results.append(result_dict)
    
    # ========================================================================
    # SIGNED CLUSTERING COEFFICIENT
    # ========================================================================
    print(f"\n{'='*60}")
    print("Signed Clustering Coefficient")
    print("="*60)
    
    # Hypothesis: Nodes in balanced clusters are more likely to form positive edges
    # Balance theory: triangles with even number of negative edges are stable
    # Calculate proportion of balanced triangles for each node as clustering measure
    signed_clustering_coeffs = torch.zeros(num_nodes)
    for node in range(num_nodes):
        # Get all neighbors (positive and negative)
        neighbors = torch.where(torch.abs(signed_adj[node]) > 0)[0]
        k = len(neighbors)
        
        if k > 1:
            # Count balanced triangles among neighbors
            balanced_triangles = 0
            total_triangles = 0
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    neighbor_i = neighbors[i]
                    neighbor_j = neighbors[j]
                    
                    # Check if there's an edge between neighbors
                    if torch.abs(signed_adj[neighbor_i, neighbor_j]) > 0:
                        # Check if triangle is balanced using balance theory
                        sign_node_i = signed_adj[node, neighbor_i]
                        sign_node_j = signed_adj[node, neighbor_j]
                        sign_i_j = signed_adj[neighbor_i, neighbor_j]
                        
                        # Triangle is balanced if product of signs is positive
                        triangle_product = sign_node_i * sign_node_j * sign_i_j
                        if triangle_product > 0:
                            balanced_triangles += 1
                        total_triangles += 1
            
            # Signed clustering coefficient (proportion of balanced triangles)
            if total_triangles > 0:
                signed_clustering_coeffs[node] = balanced_triangles / total_triangles
    
    # Use average signed clustering coefficient of both endpoints
    signed_cluster_scores = []
    for edge in test_edges:
        i, j = edge[0].item(), edge[1].item()
        avg_clustering = (signed_clustering_coeffs[i] + signed_clustering_coeffs[j]) / 2
        signed_cluster_scores.append(avg_clustering.item())
    
    signed_cluster_scores = torch.tensor(signed_cluster_scores, dtype=torch.float)
    signed_cluster_binary = (signed_cluster_scores > signed_cluster_scores.median()).float()
    
    # Handle case where all clustering scores are identical
    if signed_cluster_scores.std() > 1e-6:
        signed_cluster_auc = roc_auc_score(true_labels.cpu().numpy(), signed_cluster_scores.cpu().numpy())
    else:
        signed_cluster_auc = 0.5
    
    signed_cluster_f1 = f1_score(true_labels.cpu().numpy(), signed_cluster_binary.cpu().numpy())
    signed_cluster_precision_k = calculate_precision_at_k_heuristic(signed_cluster_scores, true_labels, k_values)
    
    print(f"Signed Clustering Coefficient: AUC = {signed_cluster_auc:.4f}, F1 = {signed_cluster_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {signed_cluster_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'signed_clustering_coefficient',
        'description': 'Nodes in balanced clusters more likely to form positive edges',
        'auc': signed_cluster_auc,
        'f1': signed_cluster_f1,
        'type': 'signed_specific'
    }
    result_dict.update(signed_cluster_precision_k)
    all_results.append(result_dict)
    

    # ========================================================================
    # SIGNED 2-HOP PATHS
    # ========================================================================
    print(f"\n{'='*60}")
    print("Signed 2-Hop Paths")
    print("="*60)
    
    # Hypothesis: More positive 2-hop paths indicate likely positive direct connection
    # Matrix multiplication A² gives 2-hop path counts with sign preservation
    # Positive values indicate net positive indirect influence between nodes
    signed_adj_squared = torch.mm(signed_adj, signed_adj)
    
    signed_katz_scores = []
    for edge in test_edges:
        i, j = edge[0].item(), edge[1].item()
        # Number of positive 2-hop paths between i and j
        score = signed_adj_squared[i, j].item()
        signed_katz_scores.append(max(0, score))  # Only count positive paths
    
    signed_katz_scores = torch.tensor(signed_katz_scores, dtype=torch.float)
    
    # Normalize scores for consistent comparison
    if signed_katz_scores.max() > signed_katz_scores.min():
        signed_katz_predictions = (signed_katz_scores - signed_katz_scores.min()) / (signed_katz_scores.max() - signed_katz_scores.min())
    else:
        signed_katz_predictions = torch.full_like(signed_katz_scores, 0.5)
    
    signed_katz_binary = (signed_katz_predictions > signed_katz_predictions.median()).float()
    signed_katz_auc = roc_auc_score(true_labels.cpu().numpy(), signed_katz_predictions.cpu().numpy())
    signed_katz_f1 = f1_score(true_labels.cpu().numpy(), signed_katz_binary.cpu().numpy())
    signed_katz_precision_k = calculate_precision_at_k_heuristic(signed_katz_predictions, true_labels, k_values)
    
    print(f"Signed 2-Hop Paths: AUC = {signed_katz_auc:.4f}, F1 = {signed_katz_f1:.4f}")
    for k in k_values:
        print(f"  Precision@{k} = {signed_katz_precision_k[f'precision_at_{k}']:.4f}")
    
    result_dict = {
        'heuristic': 'signed_2hop_paths',
        'description': 'More positive 2-hop paths = more likely positive connection',
        'auc': signed_katz_auc,
        'f1': signed_katz_f1,
        'type': 'signed_specific'
    }
    result_dict.update(signed_katz_precision_k)
    all_results.append(result_dict)
    

    # ========================================================================
    # RESULTS COMPARISON AND ANALYSIS
    # ========================================================================
    """
    Comprehensive analysis and comparison of all evaluated signed network heuristics.
    
    This section aggregates performance results from all heuristics and provides
    comparative analysis to identify the most effective approaches. The analysis
    includes ranking by performance metrics, categorical comparisons, and 
    identification of the best-performing methods for signed link prediction.
    """
    print(f"\n{'='*60}")
    print("SIGNED NETWORK HEURISTICS COMPARISON RESULTS")
    print("="*60)
    
    # Sort by AUC performance - primary metric for heuristic comparison
    # AUC provides robust performance assessment across all classification thresholds
    sorted_results = sorted(all_results, key=lambda x: x['auc'], reverse=True)
    
    # Display performance ranking table with key metrics
    # Provides quick overview of relative heuristic effectiveness
    print(f"{'Rank':<4} {'Heuristic':<30} {'Type':<15} {'AUC':<8} {'F1':<8}")
    print("-" * 70)
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank:<4} {result['heuristic']:<30} {result['type']:<15} {result['auc']:<8.4f} {result['f1']:<8.4f}")
    
    # Identify best performing heuristic - top of sorted list
    # This represents the most effective approach for this specific network
    best_heuristic = sorted_results[0]
    print(f"\n Best performing heuristic: {best_heuristic['heuristic']}")
    print(f"   Type: {best_heuristic['type']}")
    print(f"   AUC: {best_heuristic['auc']:.4f}, F1: {best_heuristic['f1']:.4f}")
    
    # Compare performance by heuristic category
    # Reveals whether signed-specific methods outperform traditional approaches
    print(f"\n Performance by Type:")
    for heuristic_type in ['baseline', 'signed_aware', 'signed_specific']:
        type_results = [r for r in all_results if r['type'] == heuristic_type]
        if type_results:
            # Calculate aggregate statistics for each category
            avg_auc = np.mean([r['auc'] for r in type_results])
            avg_f1 = np.mean([r['f1'] for r in type_results])
            print(f"   {heuristic_type:<15}: Avg AUC = {avg_auc:.4f}, Avg F1 = {avg_f1:.4f} ({len(type_results)} heuristics)")

    # VISUALIZATION
    """
    Performance visualization section creates comprehensive charts showing:
    - AUC performance ranking with type-based color coding
    - F1 score comparison across all heuristics  
    - Precision@K analysis for practical application insights
    
    Charts use consistent styling and color coding to facilitate comparison
    across different metrics and heuristic categories.
    """
    if show_plot:
        print(f"\n{'='*60}")
        print("SIGNED NETWORK HEURISTICS PERFORMANCE VISUALIZATION")
        print("="*60)
        
        # Prepare data for plotting - extract metrics and metadata
        heuristic_names = [r['heuristic'] for r in sorted_results]
        auc_scores = [r['auc'] for r in sorted_results]
        f1_scores = [r['f1'] for r in sorted_results]
        heuristic_types = [r['type'] for r in sorted_results]
        
        # Clean up heuristic names for better display readability
        display_names = []
        for name in heuristic_names:
            clean_name = name.replace('_', ' ').title()
            clean_name = clean_name.replace('2Hop', '2-Hop')
            display_names.append(clean_name)
        
        # Color mapping for different heuristic types
        # Enables visual distinction between baseline, signed-aware, and signed-specific methods
        type_colors = {
            'baseline': '#FF6B6B',       # Red
            'signed_aware': '#45B7D1',   # Blue
            'signed_specific': '#96CEB4' # Green
        }
        colors = [type_colors[t] for t in heuristic_types]
        
        # Plot 1: AUC Scores with Type Coloring
        """
        AUC performance visualization shows overall discriminative ability
        of each heuristic across all classification thresholds.
        """
        plt.figure(figsize=(16, 10))
        y_pos = np.arange(len(display_names))
        bars = plt.barh(y_pos, auc_scores, color=colors, alpha=0.8)
        
        # Add score labels on bars for precise value reading
        for i, (bar, score) in enumerate(zip(bars, auc_scores)):
            width = bar.get_width()
            plt.annotate(f'{score:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontweight='bold', fontsize=9)
        
        # Configure chart layout and styling
        plt.yticks(y_pos, display_names, fontsize=10)
        plt.xlabel('AUC Score', fontsize=12)
        plt.title(f'{title} Signed Network Heuristics Performance: AUC Scores\nTimestep {target_data["timestep"]} '
                 f'({target_data["num_edges"]:,} edges, {num_nodes:,} nodes)', 
                 fontweight='bold', fontsize=14)
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add legend for heuristic type identification
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=type_colors[t], label=t.replace('_', ' ').title()) 
                          for t in type_colors.keys()]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Try to save the plot with error handling
        try:
            plot_filename = f"signed_heuristics_auc_t{target_data['timestep']}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Signed AUC scores plot saved: {plot_filename}")
        except Exception as e:
            print(f"Could not save AUC plot: {e}")
            print("Displaying plot only...")
        
        plt.show()
        
        # Plot 2: F1 Scores with Type Coloring
        """
        F1 score visualization shows balanced precision-recall performance,
        particularly important for imbalanced signed network datasets.
        """
        plt.figure(figsize=(16, 10))
        y_pos = np.arange(len(display_names))
        bars = plt.barh(y_pos, f1_scores, color=colors, alpha=0.8)
        
        # Add score labels on bars for precise value reading
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            width = bar.get_width()
            plt.annotate(f'{score:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontweight='bold', fontsize=9)
        
        # Configure chart with consistent styling
        plt.yticks(y_pos, display_names, fontsize=10)
        plt.xlabel('F1 Score', fontsize=12)
        plt.title(f'{title} Signed Network Heuristics Performance: F1 Scores\nTimestep {target_data["timestep"]} '
                 f'({target_data["num_edges"]:,} edges, {num_nodes:,} nodes)', 
                 fontweight='bold', fontsize=14)
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add legend for heuristic type identification
        legend_elements = [Patch(facecolor=type_colors[t], label=t.replace('_', ' ').title()) 
                          for t in type_colors.keys()]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Try to save the plot with error handling
        try:
            plot_filename = f"signed_heuristics_f1_t{target_data['timestep']}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Signed F1 scores plot saved: {plot_filename}")
        except Exception as e:
            print(f"Could not save F1 plot: {e}")
            print("Displaying plot only...")
        
        plt.show()
        
        # Plot 3: Precision@K Bar Charts (Multiple subplots)
        """
        Precision@K analysis shows performance for top-ranked predictions,
        critical for practical applications where users care most about
        highest-confidence positive edge predictions.
        """
        fig, axes = plt.subplots(len(k_values), 1, figsize=(16, 10))
        if len(k_values) == 1:
            axes = [axes]  # Make it a list for consistency
        
        # Create subplot for each k value
        for idx, k in enumerate(k_values):
            precision_scores = [result[f'precision_at_{k}'] for result in sorted_results]
            
            y_pos = np.arange(len(display_names))
            bars = axes[idx].barh(y_pos, precision_scores, color=colors, alpha=0.8)
            
            # Add score labels on bars for precise value reading
            for i, (bar, score) in enumerate(zip(bars, precision_scores)):
                width = bar.get_width()
                axes[idx].annotate(f'{score:.3f}',
                                  xy=(width, bar.get_y() + bar.get_height() / 2),
                                  xytext=(5, 0), textcoords="offset points",
                                  ha='left', va='center', fontweight='bold', fontsize=9)
            
            # Configure subplot layout
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(display_names, fontsize=10)
            axes[idx].set_xlabel(f'Precision@{k} Score', fontsize=12)
            axes[idx].set_title(f'{title} Precision@{k} Scores - Timestep {target_data["timestep"]}', 
                               fontweight='bold', fontsize=13)
            axes[idx].set_xlim(0, 1)
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Add legend only on the first subplot to avoid repetition
            if idx == 0:
                legend_elements = [Patch(facecolor=type_colors[t], label=t.replace('_', ' ').title()) 
                                  for t in type_colors.keys()]
                axes[idx].legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Try to save the precision charts
        try:
            plot_filename = f"signed_heuristics_precision_bars_t{target_data['timestep']}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Signed Precision@K bar charts saved: {plot_filename}")
        except Exception as e:
            print(f"Could not save precision bar charts: {e}")
        
        plt.show()

    # Return comprehensive results dictionary
    """
    Final results aggregation provides complete evaluation summary including:
    - Network statistics and properties
    - Performance results for all heuristics
    - Best performing method identification
    - Signed network specific metrics and analysis
    """
    return {
        'timestep': target_data['timestep'],
        'num_edges': target_data['num_edges'],
        'num_nodes': num_nodes,
        'num_pos': target_data['num_pos'],
        'num_neg': target_data['num_neg'],
        'results': all_results,
        'best_heuristic': best_heuristic,
        'signed_network_stats': {
            'positive_ratio': target_data['num_pos'] / target_data['num_edges'],
            'negative_ratio': target_data['num_neg'] / target_data['num_edges'],
            'avg_total_degree': total_degrees.mean().item(),
            'avg_positive_degree': pos_degrees.mean().item(),
            'avg_negative_degree': neg_degrees.mean().item(),
            'avg_signed_degree': signed_degrees.mean().item(),
        }
    }