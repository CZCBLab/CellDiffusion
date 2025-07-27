"""
"""

import numpy as np
from sklearn.cluster import KMeans #, AffinityPropagation
import igraph as ig
import leidenalg as la
import community as community_louvain
import networkx as nx
import sklearn.metrics as mt

from utils.utility_fn import extract_data_matrix_from_adata

def k_means(adata, use_rep='X_dif', k=10):
    
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=False)
    
    kmresult = KMeans(n_clusters=k, random_state=0).fit(feature_matrix)
    adata.obs['k_means'] = kmresult.labels_
    adata.obs['k_means'] = adata.obs['k_means'].astype('category')


def leiden(adata, resolution=0.5):
    
    # Create a simple graph
    edges = adata.uns['edge_index']
    G = ig.Graph(edges)

    # Find the partition with the Leiden algorithm
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution)
    adata.obs['leiden+'] = np.array(partition.membership)
    adata.obs['leiden+'] = adata.obs['leiden+'].astype('category')


def louvain(adata, resolution=1.0):
    
    # Create a simple graph
    edge_list=adata.uns['edge_index']

    # Create a graph
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # compute the best partition
    partition = community_louvain.best_partition(G.to_undirected(), resolution=resolution)

    num_nodes = len(G.nodes())
    labels = [0]*num_nodes
    # fill in the list with community IDs
    for node, comm in partition.items():
        labels[node] = comm

    adata.obs['louvain+'] = np.array(labels)
    adata.obs['louvain+'] = adata.obs['louvain+'].astype('category')


def att_leiden(adata, resolution=0.5, initial_membership=None):
    
    edge_index = adata.uns['edge_index']
    weights = adata.uns['adjusted_attention']
    vertices= range(adata.X.shape[0])

    G = ig.Graph()
    G.add_vertices(vertices)
    G.add_edges(edge_index.T)
    G.es['weight'] = weights

    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution, 
                                  weights='weight', initial_membership=initial_membership)

    adata.obs['att_leiden'] = np.array(partition.membership)
    adata.obs['att_leiden'] = adata.obs['att_leiden'].astype('category')


def evaluate_clustering(cluster, label):
    
    adjusted_rand_score = mt.adjusted_rand_score(cluster, label)
    normalized_mutual_info_score = mt.normalized_mutual_info_score(cluster, label)
    fowlkes_mallows_score = mt.fowlkes_mallows_score(cluster, label)
    #jaccard_score = mt.jaccard_score(cluster, label)
    
    return adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
