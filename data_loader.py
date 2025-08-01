"""
This module generates synthetic temporal signed graphs using Watts-Strogatz and Barabási-Albert 
graph models and provides functionality to load, process, and visualize temporal graph datasets.

The primary purpose is to create realistic synthetic datasets that mimic the behavior of
real-world signed networks (like Bitcoin trust networks) where edges have positive or
negative signs and evolve over time through addition, removal, and sign changes.

Key Components:
- Watts-Strogatz temporal graph generation with small-world properties
- Barabási-Albert temporal graph generation with preferential attachment
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
from typing import Tuple, Dict, List, Set
from dataclasses import dataclass
import random
from collections import defaultdict

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
    num_timesteps: int = 20
    base_time: int = 1000000000
    time_interval: int = 86400
    
    edge_persistence: float = 0.4
    sign_flip_prob: float = 0.02
    new_edge_prob: float = 0.3
    edge_death_prob: float = 0.2
    activity_variation: float = 0.5
    triadic_closure_prob: float = 0.15  # Probability of forming triangles
    community_homophily: float = 0.7    # Preference for intra-community edges
    sentiment_propagation: float = 0.1   # Rate of sentiment spread along edges
    node_activity_correlation: float = 0.6  # How much node activity affects edge formation
    degree_attachment_strength: float = 0.3  # Strength of preferential attachment
    temporal_clustering_strength: float = 0.4  # Tendency for edges to cluster in time


@dataclass
class GraphConfig:
    """
    Configuration parameters that define the structural properties of graphs.
    
    These parameters control the initial graph topology and the distribution of
    positive versus negative edges, mimicking real-world signed network characteristics.
    
    Attributes:
        num_nodes: Total number of nodes (users/entities) in the network
        positive_ratio: Fraction of edges that are positive (trust vs distrust)
        
        # WS-specific parameters
        ws_k: Each node is initially connected to k nearest neighbors in ring topology
        ws_p: Probability of rewiring each edge (0=regular, 1=random)
        
        # BA-specific parameters
        ba_m: Number of edges to attach from new node in BA model
        ba_seed_nodes: Initial connected nodes for BA model
        
        # Enhanced parameters
        num_communities: Number of communities for enhanced dynamics
        community_strength: How strongly nodes cluster in communities
    """
    num_nodes: int = 2000
    positive_ratio: float = 0.88
    
    # WS-specific parameters
    ws_k: int = 6  # Each node connected to k nearest neighbors
    ws_p: float = 0.1  # Rewiring probability (0.1 creates small-world)
    
    # BA-specific parameters
    ba_m: int = 4  # Number of edges to attach from new node in BA model
    ba_seed_nodes: int = 5  # Initial connected nodes for BA model
    
    # Enhanced parameters
    num_communities: int = 5  # Number of communities for enhanced dynamics
    community_strength: float = 0.8  # How strongly nodes cluster in communities
    
# ============================================================================
# WATTS-STROGATZ TEMPORAL GRAPH GENERATOR CLASS
# ============================================================================

class WSTemporalGraphGenerator:
    """Watts-Strogatz small-world network generator with rich temporal dynamics."""
    
    def __init__(self, graph_config: GraphConfig, temporal_config: TemporalConfig):
        """Initialize the WS generator with configuration parameters."""
        self.graph_config = graph_config
        self.temporal_config = temporal_config
        
        # Initialize community structure - assign nodes to communities in a ring-aware manner
        # This respects the initial ring structure while adding community dynamics
        nodes_per_community = self.graph_config.num_nodes // self.graph_config.num_communities
        self.node_communities = []
        
        for i in range(self.graph_config.num_nodes):
            # Assign communities in blocks to maintain some locality
            community = (i // nodes_per_community) % self.graph_config.num_communities
            self.node_communities.append(community)
        
        self.node_communities = np.array(self.node_communities)
        self.node_activities = np.random.random(self.graph_config.num_nodes)
        
        # Track clustering coefficient for small-world maintenance
        self.target_clustering = None
    
    def generate_base_graph(self) -> nx.Graph:
        """Generate initial WS small-world graph."""
        # Use NetworkX's built-in Watts-Strogatz generator
        G = nx.watts_strogatz_graph(
            n=self.graph_config.num_nodes,
            k=self.graph_config.ws_k,
            p=self.graph_config.ws_p,
            seed=None
        )
        
        # Store the target clustering coefficient to maintain small-world properties
        self.target_clustering = nx.average_clustering(G)
        print(f"Initial WS graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Clustering coefficient: {self.target_clustering:.3f}")
        print(f"Average path length: {nx.average_shortest_path_length(G):.3f}")
        
        return G
    
    def assign_edge_signs(self, G: nx.Graph) -> Dict[Tuple[int, int], int]:
        """Assign edge signs with small-world and community-based bias."""
        edge_signs = {}
        
        for edge in G.edges():
            i, j = edge
            
            # Enhanced positive bias to reduce negative edge ratio
            base_positive_prob = self.graph_config.positive_ratio
            
            # Small-world networks: local edges (high clustering) tend to be more positive
            # Check if this edge contributes to local clustering
            common_neighbors = len(list(nx.common_neighbors(G, i, j)))
            clustering_bonus = min(0.1, common_neighbors * 0.02)  # Bonus for clustered edges
            
            # Community homophily effect with stronger positive bias
            if self.node_communities[i] == self.node_communities[j]:
                community_bonus = 0.08  # Within-community positive bias
            else:
                community_bonus = -0.03  # Small penalty for cross-community
            
            positive_prob = base_positive_prob + clustering_bonus + community_bonus
            positive_prob = np.clip(positive_prob, 0.75, 0.96)  # Keep in reasonable bounds
            
            sign = 1 if np.random.random() < positive_prob else -1
            edge_signs[edge] = sign
            
        return edge_signs
    
    def evolve_graph(self, current_edges: Dict[Tuple[int, int], int], timestep: int) -> Dict[Tuple[int, int], int]:
        """Evolve WS graph while maintaining small-world properties."""
        new_edges = {}
        
        # Natural activity variation that can create ups and downs
        base_activity = 1.0 + (np.random.random() - 0.5) * self.temporal_config.activity_variation * 0.6
        activity_multiplier = np.clip(base_activity, 0.8, 1.25)
        
        # PHASE 1: Edge Persistence with small-world bias
        # Local edges (contributing to clustering) have higher persistence
        persistence_rate = min(0.78, self.temporal_config.edge_persistence * 1.6 * activity_multiplier)
        
        for edge, sign in current_edges.items():
            i, j = edge
            
            # Check if edge contributes to local clustering (has common neighbors)
            current_edge_set = set(current_edges.keys())
            i_neighbors = {n for (a, b) in current_edge_set for n in [a, b] if (a == i or b == i) and n != i}
            j_neighbors = {n for (a, b) in current_edge_set for n in [a, b] if (a == j or b == j) and n != j}
            common_neighbors = len(i_neighbors.intersection(j_neighbors))
            
            # Clustering edges persist longer to maintain small-world properties
            clustering_bonus = min(0.15, common_neighbors * 0.03)
            edge_persistence = min(0.9, persistence_rate + clustering_bonus)
            
            if np.random.random() < edge_persistence:
                # Simple sign flip
                if np.random.random() < self.temporal_config.sign_flip_prob:
                    new_sign = -sign
                else:
                    new_sign = sign
                new_edges[edge] = new_sign
        
        # PHASE 2: New Edge Formation with small-world preference
        n = self.graph_config.num_nodes
        current_edge_count = len(current_edges)
        
        # Target to replace lost edges plus upward bias for recovery
        edges_lost = current_edge_count - len(new_edges)
        replacement_target = edges_lost + np.random.randint(-30, 150)  # Upward biased variation
        target_new = max(50, int(replacement_target * activity_multiplier))  # Minimum new edges
        target_new = min(target_new, n * 8)  # Higher cap
        
        # Strategy: Mix of random edges and triadic closure for small-world maintenance
        attempts = 0
        max_attempts = target_new * 4
        new_edge_count = 0
        triadic_attempts = int(target_new * 0.4)  # 40% of new edges try triadic closure
        
        # First, attempt triadic closure to maintain clustering
        triadic_added = 0
        current_edge_set = set(new_edges.keys())
        
        for _ in range(triadic_attempts):
            if triadic_added >= triadic_attempts:
                break
                
            # Pick a random existing edge
            if current_edge_set:
                edge_list = list(current_edge_set)
                random_edge = random.choice(edge_list)
                i, j = random_edge
                
                # Find neighbors of both nodes
                i_neighbors = {n for (a, b) in current_edge_set for n in [a, b] if (a == i or b == i) and n != i}
                j_neighbors = {n for (a, b) in current_edge_set for n in [a, b] if (a == j or b == j) and n != j}
                
                # Try to close triangles
                for neighbor in i_neighbors:
                    if neighbor != j:
                        potential_edge = tuple(sorted([j, neighbor]))
                        if potential_edge not in current_edge_set and potential_edge not in new_edges:
                            # Triadic closure with positive bias
                            sign_i_neighbor = current_edges.get(tuple(sorted([i, neighbor])), 
                                                             new_edges.get(tuple(sorted([i, neighbor])), 1))
                            sign_i_j = current_edges.get(tuple(sorted([i, j])), 
                                                       new_edges.get(tuple(sorted([i, j])), 1))
                            
                            # Structural balance: positive if both paths are same sign
                            if sign_i_neighbor * sign_i_j > 0:
                                triadic_sign = 1 if np.random.random() < 0.9 else -1
                            else:
                                triadic_sign = -1 if np.random.random() < 0.7 else 1
                            
                            new_edges[potential_edge] = triadic_sign
                            current_edge_set.add(potential_edge)
                            triadic_added += 1
                            new_edge_count += 1
                            break
        
        # Then add random edges for the remainder
        while new_edge_count < target_new and attempts < max_attempts:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            
            if i != j:
                edge = tuple(sorted([i, j]))
                
                if edge not in new_edges:
                    # Small-world positive bias with community effects
                    if self.node_communities[i] == self.node_communities[j]:
                        positive_prob = min(0.95, self.graph_config.positive_ratio * 1.15)
                    else:
                        positive_prob = max(0.75, self.graph_config.positive_ratio * 0.95)
                    
                    sign = 1 if np.random.random() < positive_prob else -1
                    new_edges[edge] = sign
                    current_edge_set.add(edge)
                    new_edge_count += 1
            
            attempts += 1
        
        # PHASE 3: Minimal Edge Removal to maintain small-world structure
        additional_death_rate = 0.01  # Very low to preserve structure
        edges_to_remove = []
        
        # Preferentially remove edges that don't contribute to clustering
        for edge in list(new_edges.keys()):
            i, j = edge
            
            # Check clustering contribution
            i_neighbors = {n for (a, b) in current_edge_set for n in [a, b] if (a == i or b == i) and n != i and n != j}
            j_neighbors = {n for (a, b) in current_edge_set for n in [a, b] if (a == j or b == j) and n != j and n != i}
            common_neighbors = len(i_neighbors.intersection(j_neighbors))
            
            # Edges with no common neighbors are more likely to be removed
            if common_neighbors == 0 and np.random.random() < additional_death_rate * 2:
                edges_to_remove.append(edge)
            elif np.random.random() < additional_death_rate * 0.5:  # Very low rate for clustering edges
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            del new_edges[edge]
        
        return new_edges
    
    def generate_temporal_dataset(self) -> pd.DataFrame:
        """Generate the complete WS temporal dataset."""
        print(f"Generating Watts-Strogatz small-world temporal graph...")
        print(f"Nodes: {self.graph_config.num_nodes}, Timesteps: {self.temporal_config.num_timesteps}")
        print(f"WS parameters: k={self.graph_config.ws_k}, p={self.graph_config.ws_p}")
        
        # Generate initial WS graph
        base_graph = self.generate_base_graph()
        current_edges = self.assign_edge_signs(base_graph)
        
        all_data = []
        
        for t in range(self.temporal_config.num_timesteps):
            timestamp = (self.temporal_config.base_time + 
                        t * self.temporal_config.time_interval)
            
            for (source, target), rating in current_edges.items():
                all_data.append({
                    'source': source,
                    'target': target,
                    'rating': rating,
                    'time': timestamp
                })
            
            # Print edge counts with positive/negative breakdown
            pos_count = sum(1 for r in current_edges.values() if r == 1)
            neg_count = sum(1 for r in current_edges.values() if r == -1)
            print(f"Timestep {t+1}: {len(current_edges)} edges ({pos_count}+, {neg_count}-)")
            
            if t < self.temporal_config.num_timesteps - 1:
                current_edges = self.evolve_graph(current_edges, t)
        
        df = pd.DataFrame(all_data)
        
        print(f"Generated WS dataset: {len(df)} total edges")
        print(f"Positive edges: {len(df[df['rating'] == 1])} ({len(df[df['rating'] == 1])/len(df)*100:.1f}%)")
        print(f"Negative edges: {len(df[df['rating'] == -1])} ({len(df[df['rating'] == -1])/len(df)*100:.1f}%)")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str, compress: bool = True):
        """Save the generated dataset to disk in CSV format."""
        if compress and not filename.endswith('.gz'):
            filename += '.gz'
        
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt') as f:
                df.to_csv(f, index=False, header=False)
        else:
            df.to_csv(filename, index=False, header=False)
        
        print(f"Saved dataset to: {filename}")

# ============================================================================
# BA TEMPORAL GRAPH GENERATOR CLASS
# ============================================================================

class BATemporalGraphGenerator:
    """Barabási-Albert temporal graph generator with preferential attachment."""
    
    def __init__(self, graph_config: GraphConfig, temporal_config: TemporalConfig):
        """Initialize BA generator with configuration parameters."""
        self.graph_config = graph_config
        self.temporal_config = temporal_config
        
        # Track node degrees for preferential attachment
        self.positive_degrees = np.zeros(self.graph_config.num_nodes)
        self.total_degrees = np.zeros(self.graph_config.num_nodes)
        self.node_birth_time = np.full(self.graph_config.num_nodes, -1)
        
    def generate_base_graph(self) -> nx.Graph:
        """Generate initial BA graph using preferential attachment."""
        G = nx.Graph()
        
        # Start with small complete graph
        seed_nodes = min(self.graph_config.ba_seed_nodes, self.graph_config.num_nodes)
        for i in range(seed_nodes):
            for j in range(i + 1, seed_nodes):
                G.add_edge(i, j)
            self.node_birth_time[i] = 0
        
        # Add remaining nodes with preferential attachment
        for new_node in range(seed_nodes, self.graph_config.num_nodes):
            # Calculate attachment probabilities based on current degrees
            existing_nodes = list(G.nodes())
            degrees = np.array([G.degree(node) for node in existing_nodes])
            
            if degrees.sum() > 0:
                probabilities = degrees / degrees.sum()
            else:
                probabilities = np.ones(len(existing_nodes)) / len(existing_nodes)
            
            # Select m nodes to connect to
            m = min(self.graph_config.ba_m, len(existing_nodes))
            chosen_nodes = np.random.choice(
                existing_nodes, 
                size=m, 
                replace=False, 
                p=probabilities
            )
            
            # Add edges to chosen nodes
            for target in chosen_nodes:
                G.add_edge(new_node, target)
            
            self.node_birth_time[new_node] = 0
        
        return G
    
    def assign_edge_signs(self, G: nx.Graph) -> Dict[Tuple[int, int], int]:
        """Assign edge signs with preferential attachment bias."""
        edge_signs = {}
        
        for edge in G.edges():
            u, v = edge
            
            # BA networks: high-degree nodes tend to have more positive connections
            degree_u = G.degree(u)
            degree_v = G.degree(v)
            avg_degree = (degree_u + degree_v) / 2
            
            # Higher degree nodes have higher positive probability
            max_degree = max(dict(G.degree()).values()) if G.nodes() else 1
            degree_factor = avg_degree / max_degree if max_degree > 0 else 0.5
            
            positive_prob = self.graph_config.positive_ratio + (degree_factor * 0.1)
            positive_prob = min(0.95, positive_prob)
            
            sign = 1 if np.random.random() < positive_prob else -1
            edge_signs[edge] = sign
            
            # Update degree tracking
            if sign > 0:
                self.positive_degrees[u] += 1
                self.positive_degrees[v] += 1
            
            self.total_degrees[u] += 1
            self.total_degrees[v] += 1
        
        return edge_signs
    
    def _preferential_attachment_probabilities(self, current_edges: Dict[Tuple[int, int], int], 
                                             use_positive: bool = True) -> np.ndarray:
        """Calculate preferential attachment probabilities."""
        if use_positive:
            degrees = self.positive_degrees.copy()
        else:
            degrees = self.total_degrees.copy()
        
        # Add small constant to avoid zero probabilities
        degrees = degrees + 1
        
        # Normalize to probabilities
        return degrees / degrees.sum()
    
    def evolve_graph(self, current_edges: Dict[Tuple[int, int], int], timestep: int) -> Dict[Tuple[int, int], int]:
        """Evolve BA graph with preferential attachment dynamics."""
        new_edges = {}
        
        # Reset degree counts
        self.positive_degrees.fill(0)
        self.total_degrees.fill(0)
        
        # Natural activity variation around 1.0
        activity_multiplier = 1.0 + (np.random.random() - 0.5) * self.temporal_config.activity_variation * 0.3
        activity_multiplier = max(0.9, min(1.1, activity_multiplier))
        
        # PHASE 1: Balanced Edge Persistence for BA networks
        enhanced_persistence = min(0.8, self.temporal_config.edge_persistence * 1.6)
        persistence_rate = enhanced_persistence * activity_multiplier
        
        for edge, sign in current_edges.items():
            if np.random.random() < persistence_rate:
                # Sign flip probability (lower in BA networks)
                if np.random.random() < self.temporal_config.sign_flip_prob * 0.8:
                    new_sign = -sign
                else:
                    new_sign = sign
                    
                new_edges[edge] = new_sign
                
                # Update degree counts
                u, v = edge
                if new_sign > 0:
                    self.positive_degrees[u] += 1
                    self.positive_degrees[v] += 1
                self.total_degrees[u] += 1
                self.total_degrees[v] += 1
        
        # PHASE 2: Controlled New Edge Formation for BA
        n = self.graph_config.num_nodes
        current_edge_count = len(current_edges)
        
        # Target to replace lost edges plus controlled variation
        edges_lost = current_edge_count - len(new_edges)
        replacement_target = edges_lost + np.random.randint(-30, 80)  # Controlled variation
        target_new = max(0, int(replacement_target * activity_multiplier))
        target_new = min(target_new, n * 8)  # Higher cap for BA due to preferential attachment
        
        attempts = 0
        max_attempts = target_new * 5
        new_edge_count = 0
        
        while new_edge_count < target_new and attempts < max_attempts:
            # Simplified preferential attachment
            if np.sum(self.total_degrees) > 0:
                # Use degree-based selection
                prob_total = (self.total_degrees + 1) / np.sum(self.total_degrees + 1)
                i = np.random.choice(n, p=prob_total)
                j = np.random.choice(n, p=prob_total)
            else:
                # Fallback to random selection
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
            
            if i != j:
                edge = tuple(sorted([i, j]))
                
                if edge not in new_edges:
                    # BA networks: preferential attachment increases positive probability
                    degree_i = self.positive_degrees[i] + 1
                    degree_j = self.positive_degrees[j] + 1
                    degree_factor = min(1.0, (degree_i + degree_j) / 20.0)  # Normalize degree effect
                    
                    positive_prob = self.graph_config.positive_ratio + (degree_factor * 0.1)
                    positive_prob = min(0.95, positive_prob)
                    
                    sign = 1 if np.random.random() < positive_prob else -1
                    new_edges[edge] = sign
                    new_edge_count += 1
                    
                    # Update degree counts
                    if sign > 0:
                        self.positive_degrees[i] += 1
                        self.positive_degrees[j] += 1
                    self.total_degrees[i] += 1
                    self.total_degrees[j] += 1
            
            attempts += 1
        
        # PHASE 3: Minimal Additional Edge Removal for BA networks
        additional_death_rate = 0.01  # Very minimal for BA networks
        edges_to_remove = []
        
        for edge in list(new_edges.keys()):
            u, v = edge
            # Higher degree nodes protected from removal
            avg_degree = (self.total_degrees[u] + self.total_degrees[v]) / 2
            if avg_degree < 3 and np.random.random() < additional_death_rate:  # Only remove low-degree edges
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            u, v = edge
            sign = new_edges[edge]
            
            # Update degree counts when removing edges
            if sign > 0:
                self.positive_degrees[u] = max(0, self.positive_degrees[u] - 1)
                self.positive_degrees[v] = max(0, self.positive_degrees[v] - 1)
            self.total_degrees[u] = max(0, self.total_degrees[u] - 1)
            self.total_degrees[v] = max(0, self.total_degrees[v] - 1)
            
            del new_edges[edge]
        
        return new_edges
    
    def generate_temporal_dataset(self) -> pd.DataFrame:
        """Generate the complete BA temporal dataset."""
        print(f"Generating Barabási-Albert temporal graph...")
        print(f"Nodes: {self.graph_config.num_nodes}, Timesteps: {self.temporal_config.num_timesteps}")
        print(f"BA parameter m: {self.graph_config.ba_m}")
        
        # Generate initial BA graph
        base_graph = self.generate_base_graph()
        current_edges = self.assign_edge_signs(base_graph)
        
        all_data = []
        
        for t in range(self.temporal_config.num_timesteps):
            timestamp = (self.temporal_config.base_time + 
                        t * self.temporal_config.time_interval)
            
            for (source, target), rating in current_edges.items():
                all_data.append({
                    'source': source,
                    'target': target,
                    'rating': rating,
                    'time': timestamp
                })
            
            # Print edge counts with positive/negative breakdown
            pos_count = sum(1 for r in current_edges.values() if r == 1)
            neg_count = sum(1 for r in current_edges.values() if r == -1)
            print(f"Timestep {t+1}: {len(current_edges)} edges ({pos_count}+, {neg_count}-)")
            
            if t < self.temporal_config.num_timesteps - 1:
                current_edges = self.evolve_graph(current_edges, t)
        
        df = pd.DataFrame(all_data)
        
        print(f"Generated BA dataset: {len(df)} total edges")
        print(f"Positive edges: {len(df[df['rating'] == 1])} ({len(df[df['rating'] == 1])/len(df)*100:.1f}%)")
        print(f"Negative edges: {len(df[df['rating'] == -1])} ({len(df[df['rating'] == -1])/len(df)*100:.1f}%)")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str, compress: bool = True):
        """Save the generated dataset to disk in CSV format."""
        if compress and not filename.endswith('.gz'):
            filename += '.gz'
        
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt') as f:
                df.to_csv(f, index=False, header=False)
        else:
            df.to_csv(filename, index=False, header=False)
        
        print(f"Saved dataset to: {filename}")

# ============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# ============================================================================

def create_ws_config(num_nodes: int = 2000, num_timesteps: int = 20, 
                     ws_k: int = 6, ws_p: float = 0.1) -> Tuple[GraphConfig, TemporalConfig]:
    """Factory function to create enhanced WS graph configurations."""
    graph_config = GraphConfig(
        num_nodes=num_nodes,
        positive_ratio=0.88,  # Higher positive ratio for realistic signed networks
        # WS-specific parameters
        ws_k=ws_k,
        ws_p=ws_p,
        # Enhanced parameters
        num_communities=max(3, num_nodes // 100),  # More communities for small-world structure
        community_strength=0.8
    )
    
    temporal_config = TemporalConfig(
        num_timesteps=num_timesteps,
        edge_persistence=0.4,
        sign_flip_prob=0.02,
        new_edge_prob=0.3,
        edge_death_prob=0.2,
        activity_variation=0.5,
        # Enhanced dynamics optimized for small-world networks
        triadic_closure_prob=0.15,      # Higher for clustering maintenance
        community_homophily=0.7,        # Strong within-community preference
        sentiment_propagation=0.1,
        node_activity_correlation=0.6,
        degree_attachment_strength=0.3,  # Moderate for WS networks
        temporal_clustering_strength=0.4 # Important for small-world properties
    )
    
    return graph_config, temporal_config

def create_ba_config(num_nodes: int = 2000, num_timesteps: int = 20, 
                     ba_m: int = 4) -> Tuple[GraphConfig, TemporalConfig]:
    """Factory function to create BA graph configurations."""
    graph_config = GraphConfig(
        num_nodes=num_nodes,
        positive_ratio=0.85,
        # BA-specific parameters
        ba_m=ba_m,
        ba_seed_nodes=max(5, ba_m + 1),
        # Enhanced parameters (less important for BA)
        num_communities=1,  # BA networks naturally form communities
        community_strength=0.3
    )
    
    temporal_config = TemporalConfig(
        num_timesteps=num_timesteps,
        edge_persistence=0.5,  # Higher persistence due to preferential attachment
        sign_flip_prob=0.015,  # Lower sign flip rate
        new_edge_prob=0.4,     # Higher new edge rate
        edge_death_prob=0.15,  # Lower death rate
        activity_variation=0.6,
        # Enhanced dynamics (tuned for BA)
        triadic_closure_prob=0.2,   # Higher triadic closure
        community_homophily=0.5,    # Less important for BA
        sentiment_propagation=0.08,
        node_activity_correlation=0.7,
        degree_attachment_strength=0.8,  # Strong preferential attachment
        temporal_clustering_strength=0.3
    )
    
    return graph_config, temporal_config

def generate_ws_dataset(filename: str = "synthetic_ws.csv.gz") -> pd.DataFrame:
    """Generate Watts-Strogatz temporal dataset with small-world dynamics."""
    num_nodes = 2000  # Fixed to correct value
    num_timesteps = 20  # Fixed to correct value
    ws_k = 6  # Each node connected to 6 nearest neighbors initially
    ws_p = 0.1  # 10% rewiring probability for small-world properties

    graph_config, temporal_config = create_ws_config(num_nodes, num_timesteps, ws_k, ws_p)
    
    # Enhanced parameters for recovery patterns and better positive ratio
    temporal_config.edge_persistence = 0.4  # Moderate persistence for dynamics
    temporal_config.new_edge_prob = 0.3      # Not used in new logic
    temporal_config.edge_death_prob = 0.2    # Not used in new logic
    temporal_config.activity_variation = 0.5  # Increased for more ups and downs
    temporal_config.triadic_closure_prob = 0.15  # Important for clustering maintenance
    
    generator = WSTemporalGraphGenerator(graph_config, temporal_config)
    df = generator.generate_temporal_dataset()
    
    # Use full dataset
    print(f"Using full WS dataset: {len(df)} total edges")
    
    generator.save_dataset(df, filename)
    return df

def generate_ba_dataset(filename: str = "synthetic_ba.csv.gz") -> pd.DataFrame:
    """Generate BA temporal dataset with preferential attachment dynamics."""
    num_nodes = 2000  # Fixed to correct value
    num_timesteps = 20  # Fixed to correct value
    ba_m = 4  # Increased from 3 for more connectivity

    graph_config, temporal_config = create_ba_config(num_nodes, num_timesteps, ba_m)
    
    # Balanced parameters for steady BA networks with controlled variation
    temporal_config.edge_persistence = 0.65  # Moderate-high persistence
    temporal_config.new_edge_prob = 0.5      # Not used in new logic
    temporal_config.edge_death_prob = 0.1    # Not used in new logic
    temporal_config.activity_variation = 0.15  # Very low variation for stability
    temporal_config.degree_attachment_strength = 0.9  # Strong preferential attachment
    
    generator = BATemporalGraphGenerator(graph_config, temporal_config)
    df = generator.generate_temporal_dataset()
    
    # Use full dataset
    print(f"Using full BA dataset: {len(df)} total edges")
    
    generator.save_dataset(df, filename)
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
