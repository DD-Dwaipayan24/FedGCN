"""Client-side local training for FedGCN."""

import torch
import torch.nn as nn
from graph import agg_operator


def client_update(
        model, 
        edge_index, 
        hop_features, 
        labels, 
        node_mask, 
        device, 
        local_epochs, 
        lr, 
        weight_decay = 5e-4
    ):
    """Run `local_epochs` of full-batch gradient descent on a client's owned
    nodes, restricted to `node_mask` (e.g. the train split). Citation-graph
    clients have at most a few dozen labeled nodes each, so full-batch (no
    minibatching) is standard, unlike FedAvg's image-classification clients.

    `hop_features`: list of [N_local, in_dim] tensors (hop 0..L), already
    sliced down to this client's own nodes from the pre-aggregated features
    produced by `fedgcn.aggregate.compute_hop_features` — see that module's
    docstring for why no further graph communication is needed here.
    """
    model = model.to(device)
    hop_features = hop_features.to(device)
    labels = labels.to(device)
    node_mask = node_mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    agg_op = agg_operator(hop_features, edge_index).to(device)

    model.train()
    for _ in range(local_epochs):
        optimizer.zero_grad()
        logits = model(hop_features, agg_op)
        loss = criterion(logits[node_mask], labels[node_mask])
        loss.backward()
        optimizer.step()

    return model.state_dict(), int(node_mask.sum().item())
