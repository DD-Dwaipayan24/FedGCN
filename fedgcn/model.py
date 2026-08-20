"""Model definition for FedGCN (federated Graph Convolutional Network)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedGCNNet(nn.Module):
    """Multi-hop GCN that consumes precomputed neighbor aggregations.

    A standard GCN interleaves graph propagation with a learned transform at
    every layer, so training layer l normally needs every client's *current*
    hidden features from its neighbors, every round. Because propagation is
    linear, ``fedgcn.aggregate.compute_hop_features`` precomputes the l-hop
    neighborhoods of the *raw* input features once, up front. This model
    consumes those hops directly — concatenating hop 0..L and passing them
    through an MLP, in the style of SGC/SIGN — instead of doing per-layer
    message passing, so only model weights need to be federated each round,
    not graph structure.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_hops: int = 2, dropout: float = 0.5):
        super().__init__()
        self.num_hops = num_hops
        self.dropout = dropout
        self.hop_proj = nn.Linear(in_dim * (num_hops + 1), hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, hop_features):
        """`hop_features`: list of num_hops+1 tensors [N, in_dim] (hop 0 = raw features)."""
        h = torch.cat(hop_features, dim=-1)
        h = F.relu(self.hop_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h)  # raw logits; nn.CrossEntropyLoss applies softmax internally


def init_weights(model: nn.Module) -> None:
    """Xavier-init every Linear layer's weights and zero its bias."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
