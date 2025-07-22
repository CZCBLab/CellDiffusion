import networkx as nx
import numpy
import matplotlib.pyplot as plt


def call_modularity(adata, use_label='att_leiden', edge_weight=True):

    weights = adata.uns['adjusted_attention'] if edge_weight else None

    modularity = hidden_call_modularity(adata.obs[use_label], adata.uns['edge_index'], weights=weights)
        
    return modularity

def call_gnd_modularity(adata, use_label='att_leiden', edge_weight=True):

    modularity_gnd = []
    if edge_weight:
        for edge_index, weights in zip(adata.uns['gnd_steps_edge_index'], adata.uns['gnd_steps_adjusted_attention']):
            modularity = hidden_call_modularity(adata.obs[use_label], edge_index, weights=weights)
            modularity_gnd.append(modularity)
    else:
        for edge_index in adata.uns['gnd_steps_edge_index']:
            modularity = hidden_call_modularity(adata.obs[use_label], edge_index, weights=None)
            modularity_gnd.append(modularity)
            
    return modularity_gnd


def hidden_call_modularity(cluster_labels, edge_index, weights=None):

    # Create the graph
    G = nx.Graph()
    
    if weights is not None:
        weighted_edges = numpy.concatenate((edge_index, weights[numpy.newaxis, :]), axis=0)
        G.add_weighted_edges_from(weighted_edges.T)
    else:
        G.add_edges_from(edge_index.T)

    # Convert the cluster labels to the appropriate format for the modularity function
    # The modularity function expects a list of sets, where each set contains the nodes in one community.
    existing_partition = []
    for i in numpy.unique(cluster_labels):
        community = {node for node, community in enumerate(cluster_labels, start=0) if community == i}
        existing_partition.append(community)

    # Compute the modularity of the existing partition
    if weights is not None:
        modularity = nx.algorithms.community.modularity(G, existing_partition, weight='weight')
    else:
        modularity = nx.algorithms.community.modularity(G, existing_partition)
        
    return modularity


def view_modularity(Weighted_modularity, save_fig=None):
    x = list(range(len(Weighted_modularity)))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=False)

    axs[0].plot(x, Weighted_modularity, color='orange')
    axs[0].set_title('Attention weighted modularity')
    axs[0].set_xlabel('gnd steps')
    axs[0].set_ylabel('Modularity')

    if save_fig is not None:
        plt.savefig(save_fig, dpi=500)

    plt.show()


def view_gnd_modularity(Weighted_modularity, Unweighted_modularity, save_fig=None):
    pass
    

