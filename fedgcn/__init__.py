"""
fedgcn — federated training of Graph Convolutional Networks.

Reference: Yao, Y. et al. "FedGCN: Convergence-Communication Tradeoffs in
Federated Training of Graph Convolutional Networks." NeurIPS 2023.
https://arxiv.org/abs/2201.12433

Unlike FedAvg's CNN on independent image samples, FedGCN trains a GCN over a
single graph whose *nodes* are split across clients. Message passing needs a
node's neighbors, which may live on another client, so `aggregate` runs a
one-shot, pre-training exchange of neighbor-feature sums (see its docstring)
before the usual client/server federated loop. That loop reuses fedavg's
client/server pattern for the parts that don't change (weight averaging,
seeding) — see `train_fedgcn.py` at the repo root for the CLI entry point.
"""

from .model import FedGCNNet, init_weights
from .partition import iid_partition, community_partition
from .aggregate import compute_hop_features, normalized_adjacency
from .client import client_update
from .server import federated_average
from .utils import set_seed, evaluate

__all__ = [
    "FedGCNNet",
    "init_weights",
    "iid_partition",
    "community_partition",
    "compute_hop_features",
    "normalized_adjacency",
    "client_update",
    "federated_average",
    "set_seed",
    "evaluate",
]
