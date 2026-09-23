"""Server-side aggregation."""

import copy

import torch


def federated_average(state_dicts, sample_counts):
    """Weighted average of client weights, weighted by each client's local dataset size."""
    total = sum(sample_counts)
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        stacked = torch.zeros_like(avg_state[key], dtype=torch.float32)
        for state, n in zip(state_dicts, sample_counts):
            stacked += state[key].float() * (n / total)
        avg_state[key] = stacked.to(state_dicts[0][key].dtype)
    return avg_state
