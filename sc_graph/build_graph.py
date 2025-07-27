import torch
import numpy
import math
from sklearn.ensemble import IsolationForest
import random

from utils.utility_fn import extract_data_matrix_from_adata
from utils.utility_fn import prune_edges_with_IF_labels as prune_fn


def build_adj_graph(adata, use_rep='X_fae', k=10, data_dtype = torch.float32, device='cpu'):
    """
    """
    
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=True, 
                                                    data_dtype=data_dtype, device=device)
    edge_index = hidden_build_edge_index(feature_matrix, 
                                         k_min=0, 
                                         k_max=k, 
                                         self_edge=True, 
                                         remov_edge_prob=None, 
                                         node_IF_labels=None,
                                         device=device)
    
    adata.uns['adj_edge_index'] = edge_index.cpu().numpy()
    


def build_diffusion_graph(adata, 
                          use_rep='X_fae',  
                          k=10, 
                          self_edge = False, 
                          remov_edge_prob=None, 
                          prune=False, 
                          data_dtype = torch.float32, 
                          device='cpu'):
    """
    """
    
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=True, 
                                                    data_dtype=data_dtype, device=device)
    node_IF_labels = numpy.array(adata.obs['isolation']) if prune else None
    edge_index = hidden_build_edge_index(feature_matrix, 
                                         k_min=0, 
                                         k_max=k, 
                                         self_edge=self_edge, 
                                         remov_edge_prob=remov_edge_prob, 
                                         node_IF_labels=node_IF_labels,
                                         device=device)
    
    adata.uns['diffusion_edge_index'] = edge_index.cpu().numpy()
    


def build_graph(adata, use_rep="X_dif", k=10, self_edge = False, prune=False, data_dtype = torch.float32, device='cpu'):
    """
    """
    
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=True, 
                                                    data_dtype=data_dtype, device=device)
    node_IF_labels = numpy.array(adata.obs['isolation']) if prune else None
    edge_index = hidden_build_edge_index(feature_matrix, 
                                         k_min=0, 
                                         k_max=k, 
                                         self_edge=self_edge, 
                                         remov_edge_prob=None, 
                                         node_IF_labels=node_IF_labels, 
                                         device=device)
    edge_index = edge_index.cpu().numpy()
    
    adata.uns['edge_index'] = edge_index
    


def build_gnd_steps_graph(adata, k=10, self_edge = False, prune=False, data_dtype = torch.float32, device='cpu'):
    
    node_IF_labels = numpy.array(adata.obs['isolation']) if prune else None
    
    gnd_steps_edge_index = []
    for data_matrix in adata.uns['gnd_steps_data']:
        feature_matrix = torch.tensor(data_matrix, dtype=data_dtype, device=device)
        edge_index = hidden_build_edge_index(feature_matrix, 
                                             k_min=0, 
                                             k_max=k, 
                                             self_edge=self_edge, 
                                             remov_edge_prob=None, 
                                             node_IF_labels=node_IF_labels, 
                                             device=device)
        
        gnd_steps_edge_index.append(edge_index.cpu().numpy())
        
    adata.uns['gnd_steps_edge_index'] = gnd_steps_edge_index
        


def hidden_build_edge_index(feature_matrix, k_min=0, k_max=10, self_edge=False, remov_edge_prob=None, node_IF_labels=None, device='cpu'):
    
    num_of_nodes = feature_matrix.size()[0]
    
    edge_index = knn_graph(feature_matrix.to(device), k_min=k_min, k_max=k_max, self_edge=self_edge)
    
    if node_IF_labels is not None:
        edge_index = prune_fn(edge_index, node_IF_labels)
        
    if remov_edge_prob is not None:
        mask = torch.rand(edge_index.size(1)) > remov_edge_prob
        edge_index = edge_index[:, mask]
        
    return edge_index

def knn_graph(feature_matrix, k_min=0, k_max=10, self_edge = False):
    """
    """
   
    # Calculate the pairwise squared distances between points
    dist_matrix = torch.cdist(feature_matrix, feature_matrix, p=2)

    # Find the indices of the k nearest neighbors for each point
    if self_edge:
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, k_min: k_max]  # Exclude the point itself (at index 0)
        # construct edge index from knn_indices
        edge_index = knn_indices_to_edge_index(knn_indices)
    
    else:
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, k_min+1: k_max+1] # the point itself may not at index 0(overlapped points)
        # construct edge index from knn_indices
        edge_index = knn_indices_to_edge_index(knn_indices)

        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

    return edge_index

def knn_indices_to_edge_index(knn_indices):
    """
    """
    num_points, k = knn_indices.shape

    # Create source and target node index tensors
    src_nodes = torch.arange(num_points, device=knn_indices.device).view(-1, 1).repeat(1, k).view(-1)
    trg_nodes = knn_indices.reshape(-1)

    # Concatenate the source and target node index tensors to create the edge_index tensor
    edge_index = torch.stack([src_nodes, trg_nodes], dim=0)

    return edge_index

def edge_index_to_adj(edge_index, num_of_nodes):
    """
    construct adjacency matrix from edge index
    """
    adjacency_matrix = torch.zeros((num_of_nodes, num_of_nodes), dtype=edge_index.dtype, device=edge_index.device)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1

    return adjacency_matrix



def prune_fn(edge_index, node_IF_labels):
    
    node_IF_labels = torch.tensor(node_IF_labels, device=edge_index.device)
    
    normal_nodes = torch.where(node_IF_labels == 1)[0]
    anomalous_nodes = torch.where(node_IF_labels == -1)[0]
    
    mask_1 = torch.isin(edge_index[0,:], anomalous_nodes)
    mask_2 = torch.isin(edge_index[1,:], normal_nodes)
    
    mask = mask_1 & mask_2

    edge_index = edge_index[:,~mask]
    
    return edge_index


    