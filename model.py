"""Model definition for FedGCN (federated Graph Convolutional Network)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN_model(nn.Module):
    """A multi-hop GCN with one linear layer per hop."""

    def __init__(self, num_classes, hop_size, hidden_dim: list, dropout: float = 0.5):
        super().__init__()
        self.hop_size = hop_size
        dims = hidden_dim + [num_classes]
        self.linear_layer = nn.ModuleList(
            [nn.Linear(dims[h], dims[h + 1], bias=False) for h in range(hop_size)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X, agg_op):
        hop_out = X
        # print(type(agg_op), type(hop_out))
        for h in range(self.hop_size - 1):
            agg = agg_op @ hop_out
            hop_out = self.dropout(F.relu(self.linear_layer[h](agg)))
        agg = agg_op @ hop_out
        return F.softmax(self.linear_layer[-1](agg), dim=-1)


def init_weights(model: nn.Module) -> None:
    """Xavier-init every Linear layer's weights and zero its bias."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
