"""Reproducibility and evaluation helpers for FedGCN.

`set_seed` is identical to FedAvg's and reused directly; `evaluate` differs
because node classification scores a boolean node mask against precomputed
hop features, not a DataLoader of independent samples.
"""

import random
import numpy as np

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from graph import agg_operator



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



@torch.no_grad()
def evaluate(model, hop_features, edge_index, labels, mask, device):
    """Score `model` on the nodes selected by `mask` (e.g. val_mask/test_mask)."""
    model = model.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    model.eval()
    hop_features = hop_features.to(device)
    agg_op = agg_operator(hop_features, edge_index).to(device)
    logits = model(hop_features, agg_op)
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()

    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
