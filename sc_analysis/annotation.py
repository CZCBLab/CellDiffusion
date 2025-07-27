import torch
import numpy as np
from collections import Counter

from utils.utility_fn import *


def transfer_annotation(adata, 
                           use_rep='X_fae',
                           use_label='labels',
                           batch_key='batch',
                           ref_batch='reference', # accept a str or str list
                           data_batch='new_data', # accept a str or str list
                           n_edges_per_node=50,
                           use_mnn=False,
                           prune=False, 
                           device='cpu'):
    """
    adata: 
            Anndata object that combines Reference and New Data. 
    use_rep: 
            the embeddings used to build the label transfer graph.
    use_label: 
            For Reference, it should be cell identities to be transfer; For New Data, could be 
            labeled as "New".
    batch_key: 
            The key to distinguish Reference and New Data. 
    ref_batch: 
            the batch label for Reference used in 'batch_key'. Accept a str or str list.
    data_batch: 
            the batch label for New Data used in 'batch_key'. Accept a str or str list.
    n_edges_per_node: 
            K-value in KNN graph or MNN graph.
    use_mnn: 
            if use MNN to build graph.
    prune: 
            prune edges for isolation node.
    device: 
            'cpu' or 'cuda'.
                    )
    """
    
    reference_batch = ref_batch if isinstance(ref_batch, list) else [ref_batch,]
    new_data_batch = data_batch if isinstance(data_batch, list) else [data_batch,]
    
    batch_labels = np.array(adata.obs[batch_key])
    
    batch_labels[np.isin(batch_labels, reference_batch)]='reference'
    batch_labels[np.isin(batch_labels, new_data_batch)]='new_data'
    
    feature_matrix = extract_data_matrix_from_adata(adata, 
                                                    use_rep=use_rep, 
                                                    torch_tensor=True, 
                                                    device=device)
    
    reference_alignment = check_alignment_with_reference(batch_labels, feature_matrix)
    
    edge_index = inter_batch_edges(batch_labels, 
                                   feature_matrix, 
                                   n_edges_per_node=n_edges_per_node, 
                                   use_mnn=use_mnn)
    
    if prune:
        edge_index = prune_edges_with_IF_labels(edge_index, adata.obs['isolation'])
    
    adata.uns['annotation_edge_index'] = edge_index.cpu().numpy()   
    
    adata.obs['batch_labels'] = batch_labels 
    
    adata.obs['transfered_labels'] = adata.obs[use_label]
    
    knn_label_transfer(adata, reference_alignment=reference_alignment)
    
    adata.obs['reference_alignment'] = reference_alignment


def check_alignment_with_reference(batch_labels, feature_matrix):
    
    device = feature_matrix.device
    
    reference_nodes = torch.tensor(np.where(batch_labels == 'reference')[0], device=device)
    new_data_nodes = torch.tensor(np.where(batch_labels == 'new_data')[0], device=device)
    
    new_data_feature_matrix = feature_matrix[new_data_nodes,:]
    
    # find not good aligned nodes
        
    knn_indices = feature_to_knn_indices(feature_matrix, feature_matrix_trg=feature_matrix, 
                                             k_min=None, k_max=None)
    knn_indices_new = knn_indices[new_data_nodes,:]
    
    knn_indices_mask = torch.isin(knn_indices_new, reference_nodes)

    #new_data_nodes_mask_1 = knn_indices_mask[:, :10].int().sum(dim=1) < 11
    new_data_nodes_mask = knn_indices_mask[:, :100].int().sum(dim=1) < 5

    #new_data_nodes_mask = new_data_nodes_mask_1 & new_data_nodes_mask_2
    
    not_good_aligned_nodes = new_data_nodes[new_data_nodes_mask]
    
    
    
    # find unaligned nodes
    
    knn_indices = feature_to_knn_indices(new_data_feature_matrix, feature_matrix_trg=new_data_feature_matrix, 
                                             k_min=None, k_max=None)
    knn_indices = knn_indices[new_data_nodes_mask,:]
    
    not_good_aligned_nodes_index = torch.where(new_data_nodes_mask)[0]
    
    knn_indices_mask = torch.isin(knn_indices, not_good_aligned_nodes_index)
    
    unaligned_nodes_mask = knn_indices_mask[:, :100].int().sum(dim=1) > 80
    
    
    unaligned_nodes = not_good_aligned_nodes[unaligned_nodes_mask]
    
    
    
    
    
    reference_alignment = batch_labels.astype('object')
    reference_alignment[unaligned_nodes.cpu().numpy()]='Unaligned'
        


    return reference_alignment



def inter_batch_edges(batch_labels, feature_matrix, n_edges_per_node=50, use_mnn=True):
    
    device = feature_matrix.device
    
    reference_nodes = torch.tensor(np.where(batch_labels == 'reference')[0], device=device)
    new_data_nodes = torch.tensor(np.where(batch_labels == 'new_data')[0], device=device)
    
    reference_feature_matrix = feature_matrix[reference_nodes,:]
    new_data_feature_matrix = feature_matrix[new_data_nodes,:]
    

    knn_indices = feature_to_knn_indices(new_data_feature_matrix, 
                                         feature_matrix_trg=reference_feature_matrix, 
                                         k_min=None, k_max=n_edges_per_node)

    edge_index = knn_indices_to_edge_index(knn_indices)                              

    edge_index[0,:] = new_data_nodes[edge_index[0,:]]
    edge_index[1,:] = reference_nodes[edge_index[1,:]]
    
    if use_mnn:

        inter_batch_edge_index = torch.empty(2, 0, dtype=torch.int64, device=device)

        knn_indices = feature_to_knn_indices(reference_feature_matrix, 
                                             feature_matrix_trg=new_data_feature_matrix, 
                                             k_min=None, k_max=10*n_edges_per_node)

        edge_index_inverse = knn_indices_to_edge_index(knn_indices)                              

        edge_index_inverse[0,:] = reference_nodes[edge_index_inverse[0,:]]
        edge_index_inverse[1,:] = new_data_nodes[edge_index_inverse[1,:]]
        
        edge_index = torch.cat((edge_index, edge_index_inverse), dim=1)
        edge_index = extract_mnn_edge_index(edge_index)
        
        # keep at least 10 edges for each test cell
        knn_indices = feature_to_knn_indices(new_data_feature_matrix, 
                                         feature_matrix_trg=reference_feature_matrix, 
                                         k_min=None, k_max=50)

        edge_index_1 = knn_indices_to_edge_index(knn_indices)                              

        edge_index_1[0,:] = new_data_nodes[edge_index_1[0,:]]
        edge_index_1[1,:] = reference_nodes[edge_index_1[1,:]]
        
        edge_index = torch.cat((edge_index, edge_index_1), dim=1)
        
        # remove duplicate edges
        edge_index = edge_index_remove_repetitions(edge_index)

    return edge_index


def knn_label_transfer(adata, reference_alignment=None):
    edge_index = adata.uns['annotation_edge_index']
    batch_labels = np.array(adata.obs['batch_labels']) 
    transfered_labels = np.array(adata.obs['transfered_labels'])
    
    reference_nodes = np.where(batch_labels == 'reference')[0]
    new_data_nodes = np.where(batch_labels == 'new_data')[0]
    
    for new_node in new_data_nodes:
        linked_nodes = edge_index[1, np.where(edge_index[0,:] == new_node)[0]]
        neighbor_labels = transfered_labels[linked_nodes]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        transfered_labels[new_node] = most_common_label
        
    if reference_alignment is not None:
        transfered_labels[np.where(reference_alignment == 'Unaligned')[0]] = 'Unaligned'
    
    adata.obs['transfered_labels'] = transfered_labels
    

def edge_index_remove_repetitions(edge_index):
    # Remove repetitions
    edges = torch.cat([edge_index[0, :].unsqueeze(0), edge_index[1, :].unsqueeze(0)], dim=0).T
    # Remove repetitions and sort the edges
    unique_edges = edges.unique(dim=0)
    # Split the unique edges back into source and target tensors
    unique_edge_index = unique_edges.T.contiguous()
    
    return unique_edge_index


