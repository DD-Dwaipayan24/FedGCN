import copy
import json
import pickle
import pandas as pd


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
from dataset import load_dataset



# OUTPUT_DIR = Path(__file__).parent / "outputs"



def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    dataset = load_dataset(args.dataset, args.data_root)
    data = dataset[0].to(device)

    if args.dataset == "Cora" or args.dataset == "Citeseer" or args.dataset == "Pubmed":
        train_mask = data.train_mask
        val_mask = data.val_mask

    if args.dataset == "Actor" or args.dataset == "Roman-empire" or args.dataset == "amazon-ratings":
        train_mask = data.train_mask[:,0]
        val_mask = data.val_mask[:,0]

    if args.dataset == "Cornell" or args.dataset == "Texas" or args.dataset == "Wisconsin":
        train_mask = torch.tensor([np.random.choice([True, False]) for _ in range(data.x.shape[0])])
        val_mask = torch.tensor([np.random.choice([True, False]) for _ in range(data.x.shape[0])])


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
    client_masks = [train_mask[idxs] for idxs in client_nodes]
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

    # if test_metrics["accuracy"] > best_accuracy:
    best_accuracy = test_metrics["accuracy"]
    best_precision = test_metrics["precision"]
    best_recall = test_metrics["recall"]
    best_f1= test_metrics["f1"]
    # patience_counter = 0

    # else:
        # patience_counter += 1
        # if patience_counter >= args.patience:
    metrics = {
        "accuracy" : best_accuracy,
        "precesion" : best_precision,
        "recall" : best_recall,
        "f1" : best_f1
    }
    # Saving the maodel performance in csv format
    df = pd.DataFrame([metrics])
    df.to_csv(f"{args.output_dir}/model_metrics.csv", index = False)

    # Saving the maodel performance in json format
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    params = {
        name: param.detach().cpu().numpy() 
        for name, param in global_model.named_parameters()
    }

    with open(f"{args.output_dir}/model_params.pkl", "wb") as f:
        pickle.dump(params, f)

            



    
    # out_path = args.out
    # torch.save(global_model.state_dict(), out_path)
    # print(f"Saved final global model to {out_path}")

    return history

        
def plot_metric(args, history, rounds, dataset, dpi):
    accs = [m["accuracy"] for m in history]
    plt.figure()
    plt.plot(range(1, rounds + 1), accs, marker="o")
    plt.xlabel("Communication Round")
    plt.ylabel("Validation Accuracy")
    plt.title(f"FedGCN on {dataset}")
    plt.grid(True)
    plot_path = args.output_dir + "fedgcn_accuracy.png"
    plt.savefig(plot_path, dpi=dpi)
    print(f"Saved accuracy plot to {plot_path}")
    # if show:
    #     plt.show()
    plt.close()

   


