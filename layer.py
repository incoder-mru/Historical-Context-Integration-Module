# ORIGINAL SE-SGFORMER COMPONENTS 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class CentralityEncoding(nn.Module):
    def __init__(self, max_degree: int, node_dim: int):
        super().__init__()
        self.max_degree = max_degree
        self.node_dim = node_dim
        self.z_pos = nn.Parameter(torch.randn((max_degree, node_dim)))
        self.z_neg = nn.Parameter(torch.randn((max_degree, node_dim)))

    def forward(self, x: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        positive_degrees = torch.bincount(pos_edge_index[0], minlength=num_nodes)
        negative_degrees = torch.bincount(neg_edge_index[0], minlength=num_nodes)
        positive_degrees = self.decrease_to_max_value(positive_degrees, self.max_degree - 1)
        negative_degrees = self.decrease_to_max_value(negative_degrees, self.max_degree - 1)
        x += self.z_pos[positive_degrees] + self.z_neg[negative_degrees]
        return x

    def decrease_to_max_value(self, x, max_value):
        x[x > max_value] = max_value
        return x


class RWEncoding(nn.Module):
    def __init__(self, num: int):
        super().__init__()
        self.graph_weights = nn.Parameter(torch.randn(num, 1, 1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        num_node = feature.size(1)
        weights = self.graph_weights.repeat(1, num_node, num_node).to(feature.device)
        weighted_matrix = feature * weights
        spatial_matrix = torch.sum(weighted_matrix, dim=0)
        return spatial_matrix


class ADJEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_edge_index: Tensor, neg_edge_index: Tensor, num_nodes: int) -> torch.Tensor:
        device = pos_edge_index.device
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
        adj_matrix[pos_edge_index[0], pos_edge_index[1]] = 1
        adj_matrix[pos_edge_index[1], pos_edge_index[0]] = 1
        adj_matrix[neg_edge_index[0], neg_edge_index[1]] = -1
        adj_matrix[neg_edge_index[1], neg_edge_index[0]] = -1
        row_sum = adj_matrix.sum(dim=1, keepdim=True)
        epsilon = 1e-10
        normalized_adj_matrix = adj_matrix / (row_sum + epsilon)
        return normalized_adj_matrix


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, spatial_matrix: torch.Tensor) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        a = self.compute_a(key, query)
        a = a + adj_matrix + spatial_matrix
        softmax = torch.softmax(a, dim=-1)
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query):
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        return a


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, spatial_matrix: torch.Tensor) -> torch.Tensor:
        return self.linear(
            torch.cat([
                attention_head(x, adj_matrix, spatial_matrix) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, num_heads):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            dim_in=node_dim, dim_k=node_dim, dim_q=node_dim, num_heads=num_heads,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, spatial_matrix: torch.Tensor) -> torch.Tensor:
        x_prime = self.attention(self.ln_1(x), adj_matrix, spatial_matrix) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        return x_new


def create_dummy_spatial_features(num_nodes, num_walks=4, feature_dim=50):
    """Create dummy spatial features"""
    spatial_features = torch.randn(num_walks, num_nodes, num_nodes)
    for i in range(num_walks):
        spatial_features[i] = (spatial_features[i] + spatial_features[i].T) / 2
        spatial_features[i] += 0.1
    return spatial_features