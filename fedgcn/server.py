"""Server-side aggregation for FedGCN.

FedGCN and FedAvg differ in what a client trains on (a local subgraph with
precomputed neighbor hops vs. an independent image shard) and in the one-shot
graph preprocessing step (see `fedgcn.aggregate`), but the server-side
weight-combination step is identical — a sample-weighted average of client
weights — so it's reused directly rather than duplicated.
"""

from fedavg.server import federated_average

__all__ = ["federated_average"]
