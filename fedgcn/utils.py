"""Reproducibility and evaluation helpers for FedGCN.

`set_seed` is identical to FedAvg's and reused directly; `evaluate` differs
because node classification scores a boolean node mask against precomputed
hop features, not a DataLoader of independent samples.
"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fedavg.utils import set_seed

__all__ = ["set_seed", "evaluate"]


@torch.no_grad()
def evaluate(model, hop_features, labels, mask, device):
    """Score `model` on the nodes selected by `mask` (e.g. val_mask/test_mask)."""
    model.eval()
    hop_features = [h.to(device) for h in hop_features]
    logits = model(hop_features)
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()

    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
