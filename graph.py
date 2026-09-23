import torch
from torch_geometric.datasets import Planetoid
import os
import warnings
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class graph_data():
     def __init__(self, dataset_name):
          self.dataset = Planetoid(root=f'./Desktop/01_RESEARCH/04_DATASETS/{dataset_name}', name=dataset_name)
          self.data = self.dataset[0]
          self.x = self.data.x
          

          


def adjacency_matrix(X, edge_index):
    adj_mat = torch.zeros(len(X), len(X), dtype=torch.int64)     # Creating an empty list

    for i in range(len(edge_index[0])):
            adj_mat[edge_index[0][i]][edge_index[1][i]] = 1

    return adj_mat

def degree_matrix(X, edge_index):
    adj_mat = adjacency_matrix(X, edge_index)
    deg_mat = torch.zeros(adj_mat.shape)
    for i in range(len(deg_mat)):
         deg_mat[i][i] = sum(adj_mat[i])
    
    return deg_mat

def agg_operator(X, edge_index, device = 'cuda'):

    deg_mat = degree_matrix(X, edge_index).to(device)
    adj_mat = adjacency_matrix(X, edge_index).to(device)
    deg_mat_self = deg_mat + torch.eye(deg_mat.shape[0], device=device)
    adj_mat_self = adj_mat + torch.eye(adj_mat.shape[0], device=device)
    adj_mat_self = adj_mat_self.to(device)
    deg_mat_self_inv = deg_mat_self
    for i in range(len(deg_mat)):
        deg_mat_self_inv[i][i] = 1/deg_mat_self[i][i]
    
    deg_mat_self_inv = deg_mat_self_inv.to(device)
    return (deg_mat_self_inv**0.5) @ (adj_mat_self @ (deg_mat_self_inv**0.5))
    