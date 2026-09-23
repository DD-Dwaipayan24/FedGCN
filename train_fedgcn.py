import copy

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm
from torch_geometric.utils import subgraph
import matplotlib.pyplot as plt

from model import GCN_model
from partition import community_partition, iid_partition
from client import client_update
from utils import evaluate, set_seed
from server import federated_average



# OUTPUT_DIR = Path(__file__).parent / "outputs"



def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    dataset = Planetoid(root=args.data_dir, name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(device)

    # 

    if args.iid:
        client_nodes = iid_partition(data.num_nodes, args.num_clients, seed=args.seed)
    else:
        client_nodes = community_partition(data.edge_index.cpu(), data.num_nodes, args.num_clients, seed=args.seed)

    # print(len(client_nodes[0]))

    # One-shot neighbor aggregation (see fedgcn.aggregate) - done once, before
    # any training round, independent of --rounds.
    hop_features = data.x
    client_hop_features = [data.x[idxs] for idxs in client_nodes]
    # print(client_hop_features.shape)
    client_edge_index = [
        subgraph(torch.as_tensor(idxs).to(device), data.edge_index, relabel_nodes=True,
                num_nodes=data.num_nodes)[0]
        for idxs in client_nodes
    ]
    client_masks = [data.train_mask[idxs] for idxs in client_nodes]
    hidden_dim = [data.x.shape[1]] + args.hidden_dims

    global_model = GCN_model(
        hidden_dim = hidden_dim,
        num_classes = dataset.num_classes,
        hop_size = args.hops,
        dropout = args.dropout,
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
            # print(client_hop_features[cid].shape)
            state, n = client_update(
                local_model, 
                client_edge_index[cid],
                client_hop_features[cid], 
                data.y[client_nodes[cid]],
                client_masks[cid], 
                device, 
                args.local_epochs, 
                args.lr, 
                args.weight_decay,
            )
            local_states.append(state)
            local_counts.append(n)

        if local_states:
            global_model.load_state_dict(federated_average(local_states, local_counts))

        metrics = evaluate(global_model, hop_features, data.edge_index, data.y, data.val_mask, device)
        history.append(metrics)
        print(
            f"[Round {rnd:3d}/{args.rounds}] "
            f"val_acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    test_metrics = evaluate(global_model, hop_features, data.edge_index, data.y, data.test_mask, device)
    print(
        f"[Test] acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )

    
    out_path = args.out
    torch.save(global_model.state_dict(), out_path)
    print(f"Saved final global model to {out_path}")

    return history

        
def plot_metric(args, history, rounds, dataset, dpi, show):
    accs = [m["accuracy"] for m in history]
    plt.figure()
    plt.plot(range(1, rounds + 1), accs, marker="o")
    plt.xlabel("Communication Round")
    plt.ylabel("Validation Accuracy")
    plt.title(f"FedGCN on {dataset}")
    plt.grid(True)
    plot_path = args.out / "fedgcn_accuracy.png"
    plt.savefig(plot_path, dpi=dpi)
    print(f"Saved accuracy plot to {plot_path}")
    if show:
        plt.show()
    plt.close()

   


