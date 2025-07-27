import torch
import numpy
import pandas as pd
import math

from scipy.spatial.distance import pdist, squareform
import math



def build_trajectory(adata, 
                     use_groups = 'labels',
                     use_community='community',
                     origin_group = None,
                     use_rep = 'X_dif',
                     use_weights = 'attention',
                     traj_shape='mdo_tree',
                     device='cpu',
                     save_csv = None):
    """
    """
    
    groups = adata.obs[use_groups].astype(str)
    edge_index = adata.uns['edge_index'].copy()
    edge_weights = adata.uns[use_weights] if use_weights is not None else None
    
    if use_community is not None:
        community = adata.uns[use_community]
        
    else:
        get_community(adata, use_groups = use_groups, community_name='community')
        community = adata.uns['community']
        use_community = 'community'
    
    # shape: num_community * num_community
    community_attention_matrix = get_community_attention(community, edge_index, edge_weights, device=device)
    
    traj_list = attention_to_traj(community_attention_matrix, adata.uns[use_community]['keys'], origin_group, shape=traj_shape)
    
    connectivities = attention_to_traj(community_attention_matrix, adata.uns[use_community]['keys'], origin_group, shape=None)
    
    if save_csv is not None:
        from_list = []
        to_list = []
        weight_list = []
        for item in traj_list:
            from_list.append(item[0])
            to_list.append(item[1])
            weight_list.append(item[2])
        
        traj_dict = {'from': from_list,
                     'to': to_list,
                     'weight': weight_list}
            
        traj_df = pd.DataFrame(traj_dict)
        
        traj_df.to_csv(save_csv, index=True, header=True)
    
    adata.uns['trajectory'] = traj_list
    
    adata.uns['connectivities'] = connectivities
    
    adata.uns[use_community]['attention_matrix'] = community_attention_matrix


def get_community_attention(community, edge_index, edge_weights, device='cpu'):
    
    community_keys = community['keys']
    community_component = community['component'].copy() 
    
    for key in community_component.keys():
        community_component[key] = torch.tensor(community_component[key], device=device)
    
    
    edge_index = torch.tensor(edge_index, device=device)
    edge_weights = torch.tensor(edge_weights, device=device)
    
    community_attention_matrix = torch.zeros((len(community_keys),len(community_keys)), device=device)  
    
    i=-1
    for group_src in community_keys:
        i+=1
        j=-1
        for group_trg in community_keys:
            j+=1
            edge_src_filter = torch.isin(edge_index[0], community_component[group_src])
            edge_trg_filter = torch.isin(edge_index[1], community_component[group_trg])
            edge_filter = edge_trg_filter & edge_src_filter
            
            edge_src_filter_op = torch.isin(edge_index[1], community_component[group_src])
            edge_trg_filter_op = torch.isin(edge_index[0], community_component[group_trg])
            
            edge_filter_op = edge_trg_filter_op & edge_src_filter_op
            
            edge_filter = edge_filter | edge_filter_op
            
            if edge_weights is not None:
                filtered_attention = edge_weights[edge_filter]
            else:
                filtered_attention = edge_filter.astype(int)
            
            group_attention = torch.sum(filtered_attention)/(len(community_component[group_src])*len(community_component[group_trg]))
            
            community_attention_matrix[i,j] = group_attention
    
    
    
    community_attention_matrix = community_attention_matrix.cpu().numpy()
    numpy.fill_diagonal(community_attention_matrix, 0)
    
    return community_attention_matrix





def attention_to_traj(community_weight_matrix, community_keys, origin_group, shape='mst_tree'):
    
    num_nodes = community_weight_matrix.shape[0]
    
    community_weight_add = 0
    nn = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if community_weight_matrix[i, j] > 0:
                community_weight_add = community_weight_add + community_weight_matrix[i, j]
                nn = nn + 1

    aaa = nn*community_weight_matrix/community_weight_add
    
    attention_matrix = 1/(numpy.log(aaa + 1))
    
    
    origin = numpy.where(community_keys==origin_group)[0][0]
    
    if shape=='mst_tree':
        trajectory_list_assi = prim_mst(attention_matrix, origin)
    elif shape=='mdo_tree':
        trajectory_list_assi, mdo_dis = min_distance_to_origin(attention_matrix, origin)
    elif shape==None:
        trajectory_list_assi = adjacency_matrix_to_edge_list(attention_matrix)
    
    trajectory_list = []
    for item in trajectory_list_assi:
        trajectory_list.append((community_keys[item[0]], community_keys[item[1]], item[2]))
                    
    return trajectory_list


def min_distance_to_origin(adj_matrix, origin):
    num_nodes = adj_matrix.shape[0]
    visited = [False] * num_nodes
    mdo_dis = [float('inf')]* num_nodes
    mdo_edges = []

    # Start from the origin node
    visited[origin] = True
    mdo_dis[origin] = 0.0

    for _ in range(num_nodes - 1):
        mdo_edge = (None, None, float('inf'))
        mdo_edge_sum = (None, None, float('inf'))
        for i in range(num_nodes):
            if visited[i]:
                for j in range(num_nodes):
                    if not visited[j] and adj_matrix[i, j] > 0:
                        if (adj_matrix[i, j]+mdo_dis[i]) < mdo_edge_sum[2]:
                            mdo_edge = (i, j, adj_matrix[i, j])
                            mdo_edge_sum = (i, j, adj_matrix[i, j]+mdo_dis[i])
                            mdo_dis[j] = adj_matrix[i, j]+mdo_dis[i]
        visited[mdo_edge[1]] = True
        mdo_edges.append(mdo_edge)

    return mdo_edges, mdo_dis


def prim_mst(adj_matrix, origin):
    num_nodes = adj_matrix.shape[0]
    visited = [False] * num_nodes
    mst_edges = []

    # Start from the origin node
    visited[origin] = True

    for _ in range(num_nodes - 1):
        min_edge = (None, None, float('inf'))
        for i in range(num_nodes):
            if visited[i]:
                for j in range(num_nodes):
                    if not visited[j] and adj_matrix[i, j] > 0:
                        if adj_matrix[i, j] < min_edge[2]:
                            min_edge = (i, j, adj_matrix[i, j])
        visited[min_edge[1]] = True
        mst_edges.append(min_edge)

    return mst_edges



def adjacency_matrix_to_edge_list(adj_matrix):
    edge_list = []
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] != 0:
                edge_list.append((i, j, adj_matrix[i, j]))
                edge_list.append((j, i, adj_matrix[i, j]))

    return edge_list



def get_community(adata, use_groups='labels', community_name='community'):
    
    groups = adata.obs[use_groups].astype(str)
    community_size = {}
    community_component = {}
    community_keys = groups.unique()
    
    for group_name in community_keys:
        community_component[group_name] = numpy.where(groups == group_name)[0]
        community_size[group_name] = len(community_component[group_name])/len(groups)
    
    adata.uns[community_name] = {}
    adata.uns[community_name]['use_groups'] = use_groups
    adata.uns[community_name]['keys'] = community_keys
    adata.uns[community_name]['component'] = community_component
    adata.uns[community_name]['size'] = community_size


def community_umap(adata, use_umap='X_umap', use_community='community'):
    
    groups = adata.obs[adata.uns[use_community]['use_groups']].astype(str)
    
    feature_umap = adata.obsm[use_umap]
    
    community_umap = {}
    array_umap = []

    for group_name in adata.uns[use_community]['keys']:
        group_umap = feature_umap[numpy.where(groups == group_name)[0], :]
        community_umap[group_name] = group_umap.mean(axis=0)
        array_umap.append(community_umap[group_name])
    
    adata.uns[use_community]['pos_dict'] = community_umap
    adata.uns[use_community]['pos_array'] = tune_community_umap(numpy.array(array_umap), 
                                                               tune_pos=0.3, 
                                                               tune_pos_scale=1.0)
    for ii in range(len(adata.uns[use_community]['keys'])):
        group_name = adata.uns[use_community]['keys'][ii]
        adata.uns[use_community]['pos_dict'][group_name] = adata.uns[use_community]['pos_array'][ii,:]



def tune_community_umap(community_umap, tune_pos=0.3, tune_pos_scale=1.0):
    
    range_1 = community_umap[:, 0].max() - community_umap[:, 0].min()
    range_2 = community_umap[:, 1].max() - community_umap[:, 1].min()
    
    distances = pdist(community_umap, metric='euclidean')

    # Convert the condensed distance matrix to a square matrix
    distance_matrix = squareform(distances)

    threshold = tune_pos*numpy.mean(distances)

    # Extract node pairs with distance below the mean distance
    node_pairs_below = []

    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[1]):
            if distance_matrix[i][j] < threshold:
                node_pairs_below.append((i, j))

    for node_pairs in node_pairs_below:

        umap_1 = community_umap[node_pairs[0]]
        umap_2 = community_umap[node_pairs[1]]
        umap_dif = umap_2 - umap_1
        distance = math.sqrt(umap_dif[0]*umap_dif[0] + umap_dif[1]*umap_dif[1])
        if distance < threshold:
            difference = threshold - distance
            axis_1 = difference * umap_dif[0]/distance
            axis_2 = difference * umap_dif[1]/distance

            if axis_1 > 0:
                community_umap[community_umap[:, 0] >=umap_2[0], 0] += axis_1
            else:
                community_umap[community_umap[:, 0] <=umap_2[0], 0] += axis_1

            if axis_2 > 0:
                community_umap[community_umap[:, 1] >=umap_2[1], 1] += axis_2
            else:
                community_umap[community_umap[:, 1] <=umap_2[1], 1] += axis_2   
        else:
            pass
    
    new_range_1 = community_umap[:, 0].max() - community_umap[:, 0].min()
    new_range_2 = community_umap[:, 1].max() - community_umap[:, 1].min()
    
    community_umap[:, 0] *= tune_pos_scale*range_1/new_range_1
    community_umap[:, 1] *= tune_pos_scale*range_2/new_range_2
        
    return community_umap


