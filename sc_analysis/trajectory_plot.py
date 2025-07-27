import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.utility_fn import unique_colors

def plot_connectivities(adata, 
                        threshold = None,
                        node_scale = 20000, 
                        node_color = None,
                        edge_scale=2, 
                        edge_color='gray', 
                        node_labels=True,
                        figsize=(10, 10), 
                        font_size=12, 
                        font_weight='bold',
                        title='Connectivities',
                        save_fig=None):

    nodes=adata.uns['community']['keys']

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in adata.uns['connectivities']:
        G.add_edge(edge[0], edge[1], weight=(1/edge[2]))

    node_weight = node_weight=adata.uns['community']['size']
    node_size = []
    for node in nodes:
        node_size.append(node_weight[node] * node_scale)
    
    if threshold is not None:
        weights = [w for _, _, w in G.edges(data='weight')]

        max_weight = max(weights)
        min_weight = min(weights)
                 
        threshold = (max_weight - min_weight)*threshold + min_weight
        filtered_edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w >= threshold]
        

        filtered_G = nx.Graph()
        G2 = nx.Graph()
        G2.add_nodes_from(nodes)
        filtered_G.add_weighted_edges_from(filtered_edges)
        G2.add_weighted_edges_from(filtered_edges)

        for node in G.nodes():
            if node not in filtered_G.nodes():
                node_edges = [(u, v, w) for u, v, w in G.edges(node, data='weight')]
                if node_edges:
                    max_edge = max(node_edges, key=lambda x: x[2])
                    filtered_G.add_weighted_edges_from([max_edge])
                    G2.add_weighted_edges_from([max_edge])   
                    
        for edge in adata.uns['trajectory']:
            G2.add_edge(edge[0], edge[1], weight=(1/edge[2]))
        
        G = G2
        
    # Draw the graph
    edge_weights = [d['weight']*edge_scale for (u, v, d) in G.edges(data=True)]

    pos = adata.uns['community']['pos_dict']
    
    if node_color is not None:
        colors = node_color
    else:
        unique_values = np.array(adata.uns['community']['keys'])
        colors = unique_colors(unique_values)
        
    

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=node_labels, width=0, node_size=node_size, node_color=colors, 
            font_size=font_size, font_weight=font_weight)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_color)
    plt.title(title)
    
    plt.tight_layout()
    
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()
    

    

def plot_trajectory(adata, 
                        node_scale = 20000, 
                        node_color = None,
                        node_labels=True,
                        edge_width=2, 
                        edge_color='gray',
                        arrows=True,
                        arrowstyle='->',
                        connectionstyle="arc3,rad=0.1",
                        figsize=(10, 10), 
                        font_size=12, 
                        font_weight='bold',
                        title='Connectivities',
                        save_fig=None):

    G = nx.DiGraph()
    G.add_nodes_from(adata.uns['community']['keys'])
    G.add_weighted_edges_from(adata.uns['trajectory'])
    

    node_weight = node_weight=adata.uns['community']['size']
    node_size = []
    for node in adata.uns['community']['keys']:
        node_size.append(node_weight[node] * node_scale)
        
    pos = adata.uns['community']['pos_dict']
    
    if node_color is not None:
        colors = node_color
    else:
        unique_values = np.array(adata.uns['community']['keys'])
        colors = unique_colors(unique_values)

    plt.figure(figsize=(10, 10))
    
    nx.draw_networkx(G, pos, width=edge_width, with_labels=node_labels, 
                     arrows=arrows, arrowstyle=arrowstyle,
                     node_size=node_size, 
                     node_color=colors, edge_color=edge_color, connectionstyle=connectionstyle,
                     font_size=font_size, font_weight=font_weight)
    plt.title(title)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()
    


