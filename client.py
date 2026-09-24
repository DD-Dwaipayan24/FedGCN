"""Client-side local training for FedGCN."""

import torch
import torch.nn as nn
from graph import agg_operator
from sklearn.metrics import precision_recall_fscore_support


def client_update(
        model, 
        edge_index, 
        hop_features, 
        labels, 
        node_mask, 
        device, 
        local_epochs, 
        lr, 
        log_every,
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

    train_acc_list, train_loss_list = [], []
    epochs_run = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    agg_op = agg_operator(hop_features, edge_index).to(device)

    model.train()
    for epoch in range(local_epochs):
        optimizer.zero_grad()
        logits = model(hop_features, agg_op)
        loss = criterion(logits[node_mask], labels[node_mask])

        train_loss_list.append(loss.detach().cpu().numpy())
        # val_loss = criterion(logits[val_mask], labels[val_mask])
        # val_loss_list.append(val_loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            train_acc = (preds[node_mask] == labels[node_mask]).float().mean().item()
            # val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()

            train_acc_list.append(train_acc)
            # val_acc_list.append(val_acc)

            # val_preds = preds[val_mask].cpu().numpy()
            # val_true = y[val_mask].cpu().numpy()
            # precision, recall, f1, _ = precision_recall_fscore_support(
            #     val_true, val_preds, average="macro", zero_division=0
            # )

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | "
                # f"Val: {val_acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}"
            )

        epochs_run += 1


    return model.state_dict(), int(node_mask.sum().item())
