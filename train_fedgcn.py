"""
Federated GCN (FedGCN) on citation graphs — CLI entry point
=============================================================

Reference: Yao et al., "FedGCN: Convergence-Communication Tradeoffs in
Federated Training of Graph Convolutional Networks" (NeurIPS 2023) -
https://arxiv.org/abs/2201.12433

Node classification on a Planetoid citation graph (Cora by default), with
graph nodes split across clients. Unlike train.py's FedAvg on MNIST
(independent image samples), a graph's nodes are connected, so a one-shot
neighbor-feature aggregation (see fedgcn.aggregate) runs once before the
federated loop; after that, training follows the same client-update /
server-average pattern as FedAvg.

Usage
-----
    python train_fedgcn.py --num_clients 10 --rounds 50 --frac 0.4 \
        --local_epochs 3 --lr 0.01 --hops 2 --seed 42
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from fedgcn import (
    FedGCNNet,
    client_update,
    community_partition,
    compute_hop_features,
    evaluate,
    federated_average,
    iid_partition,
    set_seed,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Federated GCN (FedGCN) on citation graphs")
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    p.add_argument("--num_clients", type=int, default=10, help="Total number of clients (K)")
    p.add_argument("--rounds", type=int, default=50, help="Communication rounds (T)")
    p.add_argument("--frac", type=float, default=0.4, help="Fraction of clients sampled per round (C)")
    p.add_argument("--local_epochs", type=int, default=3, help="Local full-batch epochs per client (E)")
    p.add_argument("--lr", type=float, default=0.01, help="Local Adam learning rate")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--hops", type=int, default=2, help="Pre-aggregated hops (L); matches the model's effective depth")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--iid", action="store_true", help="Use a uniform random node split (default: community split)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="fedgcn_model.pth", help="Filename (in outputs/) for the final global model")
    p.add_argument("--plot", action="store_true", help="Save accuracy-vs-round plot to outputs/fedgcn_accuracy.png")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root=args.data_dir, name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(device)

    if args.iid:
        client_nodes = iid_partition(data.num_nodes, args.num_clients, seed=args.seed)
    else:
        client_nodes = community_partition(data.edge_index.cpu(), data.num_nodes, args.num_clients, seed=args.seed)

    # One-shot neighbor aggregation (see fedgcn.aggregate) - done once, before
    # any training round, independent of --rounds.
    hop_features = compute_hop_features(data.edge_index, data.x, args.hops)
    client_hop_features = [[h[idxs] for h in hop_features] for idxs in client_nodes]
    client_masks = [data.train_mask[idxs] for idxs in client_nodes]

    global_model = FedGCNNet(
        in_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        num_hops=args.hops,
        dropout=args.dropout,
    ).to(device)

    num_selected = max(1, int(args.frac * args.num_clients))
    rng = np.random.default_rng(args.seed)
    history = []

    for rnd in range(1, args.rounds + 1):
        selected_clients = rng.choice(args.num_clients, size=num_selected, replace=False)

        local_states, local_counts = [], []
        for cid in tqdm(selected_clients, desc=f"Round {rnd}/{args.rounds}", leave=False):
            if client_masks[cid].sum() == 0:
                continue  # no labeled training nodes on this client
            local_model = copy.deepcopy(global_model)  # independent copy per client
            state, n = client_update(
                local_model, client_hop_features[cid], data.y[client_nodes[cid]],
                client_masks[cid], device, args.local_epochs, args.lr, args.weight_decay,
            )
            local_states.append(state)
            local_counts.append(n)

        if local_states:
            global_model.load_state_dict(federated_average(local_states, local_counts))

        metrics = evaluate(global_model, hop_features, data.y, data.val_mask, device)
        history.append(metrics)
        print(
            f"[Round {rnd:3d}/{args.rounds}] "
            f"val_acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    test_metrics = evaluate(global_model, hop_features, data.y, data.test_mask, device)
    print(
        f"[Test] acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / args.out
    torch.save(global_model.state_dict(), out_path)
    print(f"Saved final global model to {out_path}")

    if args.plot:
        import matplotlib.pyplot as plt

        accs = [m["accuracy"] for m in history]
        plt.figure()
        plt.plot(range(1, args.rounds + 1), accs, marker="o")
        plt.xlabel("Communication Round")
        plt.ylabel("Validation Accuracy")
        plt.title(f"FedGCN on {args.dataset}")
        plt.grid(True)
        plot_path = OUTPUT_DIR / "fedgcn_accuracy.png"
        plt.savefig(plot_path)
        print(f"Saved accuracy plot to {plot_path}")

    return history


if __name__ == "__main__":
    main()
