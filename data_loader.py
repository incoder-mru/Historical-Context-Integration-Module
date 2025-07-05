"""
This module generates synthetic temporal signed graphs using Erdős-Rényi random graph models
and provides functionality to load, process, and visualize temporal graph datasets.

The primary purpose is to create realistic synthetic datasets that mimic the behavior of
real-world signed networks (like Bitcoin trust networks) where edges have positive or
negative signs and evolve over time through addition, removal, and sign changes.

Key Components:
- Erdős-Rényi temporal graph generation with configurable parameters
- Temporal evolution modeling with edge persistence, sign flips, and network dynamics
- Data loading and timestep splitting for both synthetic and real datasets
- Visualization tools for analyzing temporal patterns in signed networks
"""

import pandas as pd
import numpy as np
import torch
import gzip
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, Dict
from dataclasses import dataclass

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically detect and configure the best available compute device
# Priority: CUDA GPU > Apple Metal Performance Shaders (MPS) > CPU
# This ensures optimal performance across different hardware configurations
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class TemporalConfig:
    """
    Configuration parameters that control the temporal evolution of signed graphs.
    
    These parameters determine how the graph structure and edge signs change over time,
    modeling realistic network dynamics found in social and trust networks.
    
    Attributes:
        num_timesteps: Total number of time periods to simulate
        base_time: Starting Unix timestamp for the temporal sequence
        time_interval: Time elapsed between consecutive timesteps (in seconds)
        edge_persistence: Probability that an existing edge survives to the next timestep
        sign_flip_prob: Probability that an edge changes sign (positive↔negative)
        new_edge_prob: Rate at which new edges are introduced (unused in current implementation)
        edge_death_prob: Probability that an existing edge is removed
        activity_variation: Controls temporal activity fluctuations (0=constant, 1=high variation)
    """
    num_timesteps: int = 10
    base_time: int = 1000000000  # Unix timestamp base
    time_interval: int = 86400   # Seconds between timesteps (1 day)
    edge_persistence: float = 0.4  # Probability edge survives to next timestep
    sign_flip_prob: float = 0.02   # Probability of sign flip
    new_edge_prob: float = 0.3     # Rate of new edges
    edge_death_prob: float = 0.2   # Rate of edge removal
    activity_variation: float = 0.5  # Activity variation (0=constant, 1=high variation)


@dataclass
class GraphConfig:
    """
    Configuration parameters that define the structural properties of Erdős-Rényi graphs.
    
    These parameters control the initial graph topology and the distribution of
    positive versus negative edges, mimicking real-world signed network characteristics.
    
    Attributes:
        num_nodes: Total number of nodes (users/entities) in the network
        er_prob: Edge probability for Erdős-Rényi model (controls network sparsity)
        positive_ratio: Fraction of edges that are positive (trust vs distrust)
    """
    num_nodes: int = 100
    er_prob: float = 0.005  # Edge probability (sparse like real networks)
    positive_ratio: float = 0.85  # Ratio of positive to negative edges


# ============================================================================
# TEMPORAL GRAPH GENERATOR CLASS
# ============================================================================

class ERTemporalGraphGenerator:
    """
    Generator for temporal signed Erdős-Rényi graphs that evolve over time.
    
    This class creates synthetic datasets that mimic real-world signed networks
    by modeling how trust/distrust relationships form, persist, change, and dissolve
    over time. The generator uses stochastic processes to simulate realistic
    network dynamics including:
    - Edge persistence (some relationships endure)
    - Sign flipping (trust can become distrust and vice versa)
    - Network growth and shrinkage (new connections form, old ones break)
    - Activity variations (some periods are more active than others)
    
    The resulting datasets are suitable for testing temporal graph neural networks
    and link prediction algorithms on signed networks.
    """
    
    def __init__(self, graph_config: GraphConfig, temporal_config: TemporalConfig):
        """
        Initialize the generator with structural and temporal configuration parameters.
        
        Args:
            graph_config: Parameters defining the graph structure and edge sign distribution
            temporal_config: Parameters controlling temporal evolution dynamics
        """
        self.graph_config = graph_config
        self.temporal_config = temporal_config
        
    def generate_base_graph(self) -> nx.Graph:
        """
        Generate the initial Erdős-Rényi graph structure for timestep 0.
        
        Creates a random graph where each possible edge exists with probability er_prob.
        This forms the foundation topology that will evolve over subsequent timesteps.
        The Erdős-Rényi model produces realistic sparse networks similar to real-world
        social and trust networks.
        
        Returns:
            NetworkX Graph object representing the initial network topology
        """
        return nx.erdos_renyi_graph(
            self.graph_config.num_nodes, 
            self.graph_config.er_prob
        )
    
    def assign_edge_signs(self, G: nx.Graph) -> Dict[Tuple[int, int], int]:
        """
        Assign positive (+1) or negative (-1) signs to all edges in the graph.
        
        Signs represent the nature of relationships: positive edges indicate trust,
        friendship, or positive sentiment, while negative edges represent distrust,
        antagonism, or negative sentiment. The distribution follows the configured
        positive_ratio to mimic real networks where positive relationships typically
        outnumber negative ones.
        
        Args:
            G: NetworkX graph with edges needing sign assignment
            
        Returns:
            Dictionary mapping edge tuples to their assigned signs {(u,v): sign}
        """
        edge_signs = {}
        
        for edge in G.edges():
            # Randomly assign signs based on the configured positive/negative ratio
            if np.random.random() < self.graph_config.positive_ratio:
                sign = 1  # Positive edge (trust, friendship, positive sentiment)
            else:
                sign = -1  # Negative edge (distrust, antagonism, negative sentiment)
            edge_signs[edge] = sign
            
        return edge_signs
    
    def evolve_graph(self, current_edges: Dict[Tuple[int, int], int], 
                     timestep: int) -> Dict[Tuple[int, int], int]:
        """
        Evolve the graph structure and edge signs for the next timestep.
        
        This method implements the core temporal dynamics by modeling three key processes:
        1. Edge persistence: Some existing edges survive with possible sign changes
        2. Edge addition: New relationships form based on activity levels
        3. Edge removal: Some existing relationships dissolve
        
        The evolution includes realistic temporal patterns like activity fluctuations
        and periodic trends that create more dynamic and realistic synthetic data.
        
        Args:
            current_edges: Dictionary of current edges and their signs
            timestep: Current timestep index (used for temporal patterns)
            
        Returns:
            Dictionary of edges and signs for the next timestep
        """
        new_edges = {}
        
        # Model activity variation: some timesteps have higher/lower network activity
        # This creates realistic temporal patterns where network usage fluctuates
        activity_multiplier = 1.0 + (np.random.random() - 0.5) * 2 * self.temporal_config.activity_variation
        activity_multiplier = max(0.2, min(2.0, activity_multiplier))  # Clamp to reasonable bounds
        
        # PHASE 1: Edge Persistence and Sign Evolution
        # Determine which existing edges survive to the next timestep
        persistence_rate = self.temporal_config.edge_persistence * activity_multiplier
        for edge, sign in current_edges.items():
            if np.random.random() < persistence_rate:
                # Edge survives - check if sign flips (trust→distrust or vice versa)
                if np.random.random() < self.temporal_config.sign_flip_prob:
                    new_sign = -sign  # Flip sign (relationship polarity changes)
                else:
                    new_sign = sign  # Keep existing sign (relationship polarity stable)
                new_edges[edge] = new_sign
        
        # PHASE 2: New Edge Formation
        # Add new edges based on activity level and temporal patterns
        n = self.graph_config.num_nodes
        base_edge_rate = 0.001  # Low base rate maintains realistic network sparsity
        
        # Apply temporal patterns: sinusoidal variation creates periodic activity cycles
        time_factor = 1.0 + 0.5 * np.sin(timestep * 0.5)  # Periodic temporal patterns
        target_new_edges = int(n * n * base_edge_rate * activity_multiplier * time_factor)
        target_new_edges = max(1, min(target_new_edges, n * (n-1) // 4))  # Reasonable bounds
        
        # Generate new edges through random sampling
        attempts = 0
        max_attempts = target_new_edges * 3  # Prevent infinite loops
        
        while len(new_edges) - len([e for e in new_edges if e in current_edges]) < target_new_edges and attempts < max_attempts:
            # Randomly select two distinct nodes for potential edge
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            
            if i != j:  # Avoid self-loops
                edge = tuple(sorted([i, j]))  # Canonical edge representation
                
                if edge not in new_edges:
                    # Assign sign to new edge based on configured ratio
                    if np.random.random() < self.graph_config.positive_ratio:
                        sign = 1   # New positive relationship
                    else:
                        sign = -1  # New negative relationship
                    
                    new_edges[edge] = sign
            
            attempts += 1
        
        # PHASE 3: Edge Removal (Death)
        # Remove some edges to model relationship dissolution
        death_rate = self.temporal_config.edge_death_prob / activity_multiplier
        edges_to_remove = []
        for edge in new_edges:
            if np.random.random() < death_rate:
                edges_to_remove.append(edge)
        
        # Actually remove the selected edges
        for edge in edges_to_remove:
            del new_edges[edge]
        
        return new_edges
    
    def generate_temporal_dataset(self) -> pd.DataFrame:
        """
        Generate the complete temporal signed graph dataset across all timesteps.
        
        This method orchestrates the entire dataset generation process:
        1. Creates the initial graph structure and assigns edge signs
        2. Iteratively evolves the graph through all timesteps
        3. Records all edges with their timestamps for downstream analysis
        4. Provides progress feedback and summary statistics
        
        The resulting dataset contains all temporal edges suitable for training
        and evaluating temporal graph neural networks and link prediction models.
        
        Returns:
            Pandas DataFrame with columns: source, target, rating, time
        """
        print(f"Generating Erdős-Rényi temporal graph...")
        print(f"Nodes: {self.graph_config.num_nodes}, Timesteps: {self.temporal_config.num_timesteps}")
        
        # Generate the initial graph structure and assign signs
        base_graph = self.generate_base_graph()
        current_edges = self.assign_edge_signs(base_graph)
        
        # Storage for all temporal edge data across timesteps
        all_data = []
        
        # Generate data for each timestep in the temporal sequence
        for t in range(self.temporal_config.num_timesteps):
            # Calculate timestamp for this timestep
            timestamp = (self.temporal_config.base_time + 
                        t * self.temporal_config.time_interval)
            
            # Record all edges at this timestep
            for (source, target), rating in current_edges.items():
                all_data.append({
                    'source': source,      # Source node ID
                    'target': target,      # Target node ID  
                    'rating': rating,      # Edge sign (+1 or -1)
                    'time': timestamp      # Unix timestamp
                })
            
            print(f"Timestep {t+1}: {len(current_edges)} edges")
            
            # Evolve graph for next timestep (except for the final timestep)
            if t < self.temporal_config.num_timesteps - 1:
                current_edges = self.evolve_graph(current_edges, t)
        
        # Convert to DataFrame for easy manipulation and analysis
        df = pd.DataFrame(all_data)
        
        # Print comprehensive dataset statistics
        print(f"Generated dataset: {len(df)} total edges")
        print(f"Positive edges: {len(df[df['rating'] == 1])} ({len(df[df['rating'] == 1])/len(df)*100:.1f}%)")
        print(f"Negative edges: {len(df[df['rating'] == -1])} ({len(df[df['rating'] == -1])/len(df)*100:.1f}%)")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str, compress: bool = True):
        """
        Save the generated dataset to disk in CSV format.
        
        Supports optional gzip compression to reduce file size for large datasets.
        The CSV format ensures compatibility with various analysis tools and frameworks.
        Headers are omitted to match the expected format for downstream processing.
        
        Args:
            df: DataFrame containing the temporal graph data
            filename: Output filename (will add .gz if compression enabled)
            compress: Whether to apply gzip compression to reduce file size
        """
        # Automatically add compression extension if not present
        if compress and not filename.endswith('.gz'):
            filename += '.gz'
        
        # Save with appropriate compression
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt') as f:
                df.to_csv(f, index=False, header=False)  # No headers for compatibility
        else:
            df.to_csv(filename, index=False, header=False)
        
        print(f"Saved dataset to: {filename}")


# ============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# ============================================================================

def create_er_config(num_nodes: int = 100, num_timesteps: int = 10, 
                     edge_prob: float = 0.005) -> Tuple[GraphConfig, TemporalConfig]:
    """
    Factory function to create standard Erdős-Rényi graph configurations.
    
    Provides sensible default parameters that work well for most use cases
    while allowing customization of the most important parameters (nodes, timesteps, sparsity).
    The defaults are tuned to create realistic sparse networks similar to
    Bitcoin and other real-world signed networks.
    
    Args:
        num_nodes: Number of nodes in the network
        num_timesteps: Number of temporal snapshots to generate
        edge_prob: Edge probability controlling network density/sparsity
        
    Returns:
        Tuple of (GraphConfig, TemporalConfig) with configured parameters
    """
    graph_config = GraphConfig(
        num_nodes=num_nodes,
        er_prob=edge_prob,
        positive_ratio=0.85  # Bitcoin-like positive/negative ratio
    )
    
    temporal_config = TemporalConfig(
        num_timesteps=num_timesteps,
        edge_persistence=0.4,      # Moderate persistence for dynamic networks
        sign_flip_prob=0.02,       # Low sign flip rate (relationships are somewhat stable)
        new_edge_prob=0.3,         # Moderate new edge formation rate
        edge_death_prob=0.2,       # Moderate edge removal rate
        activity_variation=0.5     # Moderate activity fluctuations
    )
    
    return graph_config, temporal_config

def generate_er_dataset(filename: str = "synthetic_er.csv.gz") -> pd.DataFrame:
    """
    Generate a large-scale Erdős-Rényi temporal dataset with Bitcoin-like characteristics.
    
    Creates a substantial dataset (5000 nodes, 8 timesteps) that mimics the scale
    and dynamics of real cryptocurrency trust networks. The parameters are specifically
    tuned to create high temporal variation, realistic sparsity, and appropriate
    positive/negative edge ratios found in Bitcoin transaction networks.
    
    The function includes intelligent sampling to control dataset size while
    maintaining temporal patterns and statistical properties across all timesteps.
    
    Args:
        filename: Output filename for the generated dataset
        
    Returns:
        Generated DataFrame containing the temporal signed graph data
    """
    # Configuration for Bitcoin-scale networks
    num_nodes = 5000
    num_timesteps = 8
    edge_prob = 0.0008  # Very sparse to match Bitcoin's sparse transaction patterns

    # Create base configuration
    graph_config, temporal_config = create_er_config(num_nodes, num_timesteps, edge_prob)
    
    # Adjust temporal parameters to match Bitcoin's high temporal variation
    temporal_config.edge_persistence = 0.15      # Low persistence creates high variation
    temporal_config.new_edge_prob = 0.4          # High new edge rate for dynamic growth
    temporal_config.edge_death_prob = 0.3        # High death rate balances growth
    temporal_config.activity_variation = 0.8     # High variation mimics Bitcoin activity patterns
    
    # Generate the dataset
    generator = ERTemporalGraphGenerator(graph_config, temporal_config)
    df = generator.generate_temporal_dataset()
    
    # Apply intelligent sampling if dataset is too large
    # This maintains temporal patterns while controlling computational requirements
    total_edges = len(df)
    target_edges = 35000  # Target size for manageable computation
    
    if total_edges > target_edges * 1.5:
        print(f"Sampling down from {total_edges} to ~{target_edges} edges")
        
        sampled_data = []
        # Sample proportionally from each timestep to maintain temporal patterns
        for time in sorted(df['time'].unique()):
            timestep_df = df[df['time'] == time]
            timestep_target = int(len(timestep_df) * target_edges / total_edges)
            timestep_target = max(100, timestep_target)  # Ensure minimum timestep size
            
            if len(timestep_df) > timestep_target:
                sampled_timestep = timestep_df.sample(n=timestep_target, random_state=42)
            else:
                sampled_timestep = timestep_df
            
            sampled_data.append(sampled_timestep)
        
        df = pd.concat(sampled_data, ignore_index=True)
    
    # Save the generated dataset
    generator.save_dataset(df, filename)
    
    # Print comprehensive summary statistics
    print(f"Total edges: {len(df)}")
    print(f"Nodes: {num_nodes}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    
    # Per-timestep statistics for validation
    for i, time in enumerate(sorted(df['time'].unique()), 1):
        timestep_df = df[df['time'] == time]
        pos_count = len(timestep_df[timestep_df['rating'] == 1])
        neg_count = len(timestep_df[timestep_df['rating'] == -1])
        print(f"Timestep {i}: {len(timestep_df)} edges ({pos_count}+, {neg_count}-)")
    
    return df

# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================

# Device configuration for tensor operations (duplicated for standalone usage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset_timesteps(file_path, num_time_bins=10):
    """
    Load and process temporal graph datasets by splitting them into discrete timesteps.
    
    This function handles the complete pipeline for preparing temporal graph data:
    1. Loads data from CSV/compressed files with automatic format detection
    2. Creates consistent node ID mappings for graph neural network compatibility
    3. Splits temporal data into equal-duration time bins
    4. Separates positive and negative edges for signed graph processing
    5. Converts data to PyTorch tensors on the appropriate device
    6. Filters out empty or single-sign timesteps for model stability
    
    The resulting timestep data structure is optimized for temporal graph neural
    networks and link prediction tasks on signed networks.
    
    Args:
        file_path: Path to the dataset file (supports .csv and .csv.gz)
        num_time_bins: Number of equal-duration timesteps to create
        
    Returns:
        Tuple of (timesteps_list, num_nodes) where timesteps_list contains
        dictionaries with processed data for each valid timestep
    """
    print(f"Loading dataset: {file_path}")
    
    # Load data with automatic compression detection
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, header=None, names=['source', 'target', 'rating', 'time'])
    else:
        df = pd.read_csv(file_path, header=None, names=['source', 'target', 'rating', 'time'])
    
    print(f"Raw data: {len(df)} edges")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    
    # Create consistent node ID mapping for graph neural network compatibility
    # This ensures node IDs are contiguous integers starting from 0
    unique_nodes = sorted(set(df['source']) | set(df['target']))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    # Apply node mapping and standardize edge attributes
    df['source_mapped'] = df['source'].map(node_mapping)
    df['target_mapped'] = df['target'].map(node_mapping)
    df['edge_attr'] = df['rating'].apply(lambda x: 1 if x > 0 else -1)  # Normalize to ±1
    
    # Create equal-duration time bins for temporal analysis
    min_time, max_time = df['time'].min(), df['time'].max()
    time_bin_edges = np.linspace(min_time, max_time, num_time_bins + 1)
    
    timesteps = []
    
    # Process each time bin to create timestep data structures
    for i in range(num_time_bins):
        bin_start, bin_end = time_bin_edges[i], time_bin_edges[i + 1]
        
        # Handle bin boundaries (include endpoint for final bin)
        if i == num_time_bins - 1:
            bin_df = df[(df['time'] >= bin_start) & (df['time'] <= bin_end)]
        else:
            bin_df = df[(df['time'] >= bin_start) & (df['time'] < bin_end)]
        
        # Skip empty time bins
        if len(bin_df) == 0:
            continue
        
        # Separate positive and negative edges for signed graph processing
        pos_df = bin_df[bin_df['edge_attr'] == 1]
        neg_df = bin_df[bin_df['edge_attr'] == -1]
        
        # Skip timesteps with only one edge type (causes training instability)
        if len(pos_df) == 0 or len(neg_df) == 0:
            continue
        
        # Convert to PyTorch tensors with optimized array creation
        # This avoids the numpy array list warning by using proper array construction
        pos_edges_array = np.array([pos_df['source_mapped'].values, pos_df['target_mapped'].values])
        pos_edge_index = torch.from_numpy(pos_edges_array).long().to(device)
        
        neg_edges_array = np.array([neg_df['source_mapped'].values, neg_df['target_mapped'].values])
        neg_edge_index = torch.from_numpy(neg_edges_array).long().to(device)
        
        # Create comprehensive timestep data structure
        timestep_data = {
            'timestep': i + 1,                    # Human-readable timestep number
            'time_range': (bin_start, bin_end),   # Actual time boundaries
            'pos_edge_index': pos_edge_index,     # Positive edges as tensor
            'neg_edge_index': neg_edge_index,     # Negative edges as tensor
            'num_edges': len(bin_df),             # Total edges in this timestep
            'num_pos': len(pos_df),               # Count of positive edges
            'num_neg': len(neg_df)                # Count of negative edges
        }
        
        timesteps.append(timestep_data)
        print(f"Timestep {i+1}: {len(bin_df)} edges ({len(pos_df)}+, {len(neg_df)}-)")
    
    return timesteps, len(unique_nodes)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_total_edges_over_time(timesteps, save_path=None, figsize=(10, 6), title=None):
    """
    Visualize the total number of edges across all timesteps.
    
    This plot reveals temporal patterns in network activity, showing periods of
    high and low connectivity. Useful for understanding dataset characteristics
    and validating temporal evolution patterns in synthetic data.
    
    The visualization includes annotations for minimum and maximum activity periods
    to highlight temporal extremes and help identify interesting time periods.
    
    Args:
        timesteps: List of timestep dictionaries from load_dataset_timesteps
        save_path: Optional path to save the figure
        figsize: Figure dimensions as (width, height) tuple
        title: Optional title prefix for the plot
        
    Returns:
        Matplotlib figure object for further customization
    """
    if not timesteps:
        print("No timesteps to visualize")
        return
    
    # Extract temporal data for plotting
    timestep_nums = [ts['timestep'] for ts in timesteps]
    total_edges = [ts['num_edges'] for ts in timesteps]
    
    # Create professional-looking plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(timestep_nums, total_edges, 'b-o', linewidth=2.5, markersize=8, 
            markerfacecolor='lightblue', markeredgecolor='blue')
    
    ax.set_title(f"{title} Total Edges Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add informative annotations for extreme values
    min_idx = total_edges.index(min(total_edges))
    max_idx = total_edges.index(max(total_edges))
    
    ax.annotate(f'Min: {total_edges[min_idx]}', 
                xy=(timestep_nums[min_idx], total_edges[min_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.annotate(f'Max: {total_edges[max_idx]}', 
                xy=(timestep_nums[max_idx], total_edges[max_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Optional save functionality
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig

def plot_positive_vs_negative_edges(timesteps, save_path=None, figsize=(10, 6), title=None):
    """
    Visualize the evolution of positive versus negative edges over time.
    
    This plot reveals the temporal dynamics of signed relationships, showing
    how trust (positive edges) and distrust (negative edges) evolve differently
    over time. Essential for understanding the balance of sentiment in signed
    networks and validating that synthetic data maintains realistic proportions.
    
    The dual-line plot makes it easy to compare trends and identify periods
    where one type of relationship dominates or where both change together.
    
    Args:
        timesteps: List of timestep dictionaries from load_dataset_timesteps
        save_path: Optional path to save the figure
        figsize: Figure dimensions as (width, height) tuple
        title: Optional title prefix for the plot
        
    Returns:
        Matplotlib figure object for further customization
    """
    if not timesteps:
        print("No timesteps to visualize")
        return
    
    # Extract signed edge data for comparative plotting
    timestep_nums = [ts['timestep'] for ts in timesteps]
    pos_edges = [ts['num_pos'] for ts in timesteps]
    neg_edges = [ts['num_neg'] for ts in timesteps]
    
    # Create comparative plot with distinct styling for each edge type
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(timestep_nums, pos_edges, 'g-o', label='Positive', linewidth=2.5, markersize=8)
    ax.plot(timestep_nums, neg_edges, 'r-o', label='Negative', linewidth=2.5, markersize=8)
    
    ax.set_title(f"{title} Positive vs Negative Edges Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Optional save functionality
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig