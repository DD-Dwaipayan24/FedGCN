# Federated-Learning

Personal research experiments on federated learning algorithms:
[FedAvg](https://arxiv.org/abs/1602.05629) (McMahan et al., 2017) on MNIST,
and [FedGCN](https://arxiv.org/abs/2201.12433) (Yao et al., 2023) — federated
training of Graph Convolutional Networks — on Planetoid citation graphs.

## Structure

```
.
├── FedAvg.py             # Self-contained FedAvg reference script (single
│                          #   file, both a plain run and the paper's B/E sweep)
├── train.py               # FedAvg CLI entry point, built on fedavg/ below
├── train_fedgcn.py         # FedGCN CLI entry point, built on fedgcn/ below
├── fedavg/                 # Reusable FedAvg building blocks
│   ├── model.py             # MnistCNN, weight init
│   ├── partition.py         # IID / non-IID client data splits (image samples)
│   ├── client.py            # Local client update (ClientUpdate)
│   ├── server.py            # Weighted-average weight aggregation
│   └── utils.py             # Seeding, evaluation metrics
├── fedgcn/                  # Reusable FedGCN building blocks
│   ├── model.py               # FedGCNNet: multi-hop GCN (SGC/SIGN-style)
│   ├── partition.py            # IID / community client node splits (graph)
│   ├── aggregate.py            # One-shot L-hop neighbor feature precompute
│   ├── client.py                # Local client update, full-batch
│   ├── server.py                 # Reuses fedavg's weight aggregation
│   └── utils.py                   # Reuses fedavg's seeding; graph-mask evaluation
├── notebooks/              # Exploratory notebooks
├── outputs/                 # Trained checkpoints & plots (git-ignored)
└── data/                     # MNIST / Planetoid, auto-downloaded (git-ignored)
```

`FedAvg.py` and `train.py` implement the same FedAvg algorithm; `FedAvg.py`
is the original monolithic script kept as a standalone reference, `train.py`
is the modular version built from the `fedavg` package. `fedgcn` follows the
same client/server/model/partition/utils shape so the two algorithms stay
easy to compare, but it is not just a renamed FedAvg — see
[FedGCN vs. FedAvg](#fedgcn-vs-fedavg) below for what's actually different.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# FedAvg — modular CLI
python train.py --num_clients 100 --rounds 50 --frac 0.1 \
    --local_epochs 5 --batch_size 10 --lr 0.01 --seed 42 --plot

# FedAvg — reference script (also supports --based_on paper for the B/E sweep)
python FedAvg.py --num_clients 100 --rounds 50 --frac 0.1 \
    --local_epochs 5 --batch_size 10 --lr 0.01 --seed 42 --plot

# FedGCN — node classification on Cora
python train_fedgcn.py --num_clients 10 --rounds 50 --frac 0.4 \
    --local_epochs 3 --lr 0.01 --hops 2 --seed 42 --plot
```

By default FedAvg clients get a non-IID, sorted-by-label shard partition (as
in the original paper); pass `--iid` for a uniform random partition instead.
FedGCN clients default to a locally-connected community partition (a
dependency-free BFS-based stand-in for Louvain clustering); pass `--iid` for
a uniform random node split instead — which, for a graph, severs far more
edges and is a much harder non-IID setting.

Trained models are saved to `outputs/`; MNIST/Planetoid data is downloaded to
`data/` on first run — neither directory is version-controlled.

## FedGCN vs. FedAvg

FedAvg's clients hold independent image samples, so nothing but model
weights ever needs to cross the network. FedGCN's clients each hold a
*subgraph* of one connected graph — a GCN layer needs a node's neighbors,
and some of those neighbors live on other clients. Recomputing neighbor
features from other clients every training round would make communication
scale with the number of rounds, defeating the point of federating at all.

FedGCN's fix (`fedgcn/aggregate.py`): since graph propagation is linear, the
L-hop neighborhood of the *raw* input features can be precomputed once,
before training starts, and reused unchanged for all T rounds — in the real
protocol, clients exchange only partial neighbor-feature sums with the
server, one round per hop. `fedgcn/model.py`'s `FedGCNNet` then consumes
those precomputed hops directly (concatenated through an MLP, SGC/SIGN-style)
instead of doing per-layer message passing. Everything downstream of that —
sampling clients, local training, weighted-average aggregation — is the same
shape as `fedavg/`, and `fedgcn/server.py` and part of `fedgcn/utils.py`
reuse `fedavg`'s code directly rather than duplicating it.

## References

- McMahan, H. B., et al. "Communication-Efficient Learning of Deep Networks
  from Decentralized Data." AISTATS 2017. https://arxiv.org/abs/1602.05629
- Yao, Y., et al. "FedGCN: Convergence-Communication Tradeoffs in Federated
  Training of Graph Convolutional Networks." NeurIPS 2023.
  https://arxiv.org/abs/2201.12433
