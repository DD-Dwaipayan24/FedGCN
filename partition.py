"""Node partitioning across clients for graph-structured federated learning.

Unlike FedAvg's image samples, graph nodes are connected: a uniform random
split severs almost every edge between clients, which is an unrealistic
non-IID split for a graph (real deployments — e.g. hospitals, social
platforms — hold locally *connected* subgraphs, not random node samples).
"""

import numpy as np


def iid_partition(num_nodes: int, num_clients: int, seed: int = 42):
    """Uniform random split of node indices across clients."""
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(num_nodes)
    return [chunk.tolist() for chunk in np.array_split(idxs, num_clients)]


def community_partition(edge_index, num_nodes: int, num_clients: int, seed: int = 42):
    """Multi-source BFS clustering: a lightweight, dependency-free stand-in for
    the Louvain community partition used in the FedGCN paper's non-IID
    setting. Growing each client's node set from a random seed node along
    real edges keeps every client's subgraph locally connected, which is what
    makes it a realistic non-IID split for a graph.
    """
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)

    owner = np.full(num_nodes, -1, dtype=int)
    seeds = rng.choice(num_nodes, size=num_clients, replace=False)
    frontiers = [[int(s)] for s in seeds]
    counts = np.ones(num_clients, dtype=int)
    for cid, s in enumerate(seeds):
        owner[s] = cid

    target = num_nodes / num_clients
    remaining = num_nodes - num_clients
    order = list(range(num_clients))
    while remaining > 0:
        progressed = False
        rng.shuffle(order)
        for cid in order:
            if remaining == 0:
                break
            if counts[cid] >= target or not frontiers[cid]:
                continue
            next_frontier = []
            for node in frontiers[cid]:
                for nbr in adj[node]:
                    if owner[nbr] == -1:
                        owner[nbr] = cid
                        counts[cid] += 1
                        next_frontier.append(nbr)
                        remaining -= 1
                        progressed = True
            frontiers[cid] = next_frontier
        if not progressed:
            break  # remaining nodes are unreachable (disconnected components)

    # Nodes no BFS frontier reached (disconnected components, or every
    # frontier died out before hitting its target size): hand out round-robin
    # to whichever client is currently smallest.
    for node in np.where(owner == -1)[0]:
        cid = int(np.argmin(counts))
        owner[node] = cid
        counts[cid] += 1

    return [np.where(owner == cid)[0].tolist() for cid in range(num_clients)]
