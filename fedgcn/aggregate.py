"""One-shot neighbor-feature aggregation — the core FedGCN mechanism.

A GCN layer aggregates each node's neighbor features before transforming
them. In FedAvg's setting (independent image samples) that's a non-issue;
here the graph's nodes are split across clients, so some of a node's
neighbors usually live on *other* clients. Naively fixing this by having
clients exchange neighbor features every training round would make
communication scale with the number of rounds T — exactly what federated
learning is trying to avoid.

FedGCN (Yao et al., NeurIPS 2023, https://arxiv.org/abs/2201.12433) avoids
that by observing that graph propagation is linear: the L-hop neighborhood
of the *raw* input features can be precomputed once, before training starts,
and reused unchanged for every one of the T rounds. In their Algorithm 1,
clients compute this by exchanging only partial sums/counts along boundary
edges with the server — one round of communication per hop, done once, not
once per training round.

This module computes that same result directly (all clients' data lives in
one process here), then ``train_fedgcn.py`` splits it by ownership so each
client keeps only the rows for nodes it owns — precisely what it would hold
after running the real federated protocol.
"""

import torch


def normalized_adjacency(edge_index, num_nodes: int, device=None):
    """Build the symmetric-normalized adjacency D^-1/2 (A + I) D^-1/2 (standard GCN
    propagation matrix, Kipf & Welling 2017) as a sparse tensor."""
    device = device or edge_index.device
    self_loops = torch.arange(num_nodes, device=device)
    src = torch.cat([edge_index[0], edge_index[1], self_loops])
    dst = torch.cat([edge_index[1], edge_index[0], self_loops])

    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    weights = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([dst, src])  # row=dst so that A_hat @ x aggregates INTO dst
    a_hat = torch.sparse_coo_tensor(
        indices, weights, (num_nodes, num_nodes), check_invariants=False
    )
    return a_hat.coalesce()


def compute_hop_features(edge_index, x, num_hops: int):
    """Precompute [X^0, X^1, ..., X^num_hops] where X^l is the l-hop propagated
    raw feature matrix (X^0 = x itself). Feeding these hops directly into a
    model (see ``fedgcn.model.FedGCNNet``) instead of doing per-layer message
    passing is what makes the one-shot precomputation above valid for a
    multi-layer GCN.
    """
    a_hat = normalized_adjacency(edge_index, x.size(0), device=x.device)
    hops = [x]
    h = x
    for _ in range(num_hops):
        h = torch.sparse.mm(a_hat, h)
        hops.append(h)
    return hops
