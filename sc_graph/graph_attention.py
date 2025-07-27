import numpy
import torch

from diffusion.gnd import Attention_Weight_Sum, Attention_Inner_Product, Attention_Distance


def call_attention(adata, use_rep='X_dif', attention_type=None, num_heads_diffusion=None, dropout=None, device='cpu'):
    
    nodes_features = torch.tensor(adata.obsm[use_rep], dtype=torch.float32, device=device)
    
    edge_index = torch.tensor(adata.uns['edge_index'], dtype=torch.int64, device=device)
        
    model_dict = {k: torch.tensor(v).to(device) for k, v in adata.uns['gnd_state_dict'].items()}
    
    attention, adjusted_attention = hidden_call_attention(nodes_features, edge_index, model_dict, attention_type=attention_type, num_heads_diffusion=num_heads_diffusion, dropout=dropout, graph_diffusion_args=adata.uns['graph_diffusion_args'])
    
    attention = attention.cpu().numpy()
    adjusted_attention = adjusted_attention.cpu().numpy()
    
    adata.uns['attention'] = attention
    adata.uns['adjusted_attention'] = adjusted_attention
    

def call_gnd_attention(adata, attention_type=None, num_heads_diffusion=None, dropout=None, device='cpu'):
    
    model_dict = {k: torch.tensor(v).to(device) for k, v in adata.uns['gnd_state_dict'].items()}
    
    attention_list = []
    adjusted_attention_list = []
    for ii in range(len(adata.uns['gnd_steps_data'])):
        nodes_features = torch.tensor(adata.uns['gnd_steps_data'][ii], dtype=torch.float32, device=device)
        edge_index = torch.tensor(adata.uns['gnd_steps_edge_index'][ii], dtype=torch.int64, device=device)

        attention, adjusted_attention = hidden_call_attention(nodes_features, edge_index, model_dict, attention_type=attention_type, num_heads_diffusion=num_heads_diffusion, dropout=dropout, graph_diffusion_args=adata.uns['graph_diffusion_args'])

        attention = attention.cpu().numpy()
        adjusted_attention = adjusted_attention.cpu().numpy()

        attention_list.append(attention)
        adjusted_attention_list.append(adjusted_attention)
        
    adata.uns['gnd_steps_attention'] = attention_list
    adata.uns['gnd_steps_adjusted_attention'] = adjusted_attention_list
        


def hidden_call_attention(nodes_features, edge_index, model_dict, 
                          attention_type=None, num_heads_diffusion=None, dropout=None, graph_diffusion_args=None):
    
    num_features = nodes_features.shape[1]
    nodes_features = nodes_features.view(-1, 1, num_features)
    data = (nodes_features, edge_index)
    
    if num_heads_diffusion is None:
        num_heads_diffusion = graph_diffusion_args["num_heads_diffusion"] 
    if attention_type is None:
        attention_type = graph_diffusion_args["attention_type"] 
    if dropout is None:
        dropout = graph_diffusion_args["dropout"]
    
    # attention shape = (E, NH, 1)
    if attention_type == "sum":
        attention = att_weight_sum(data, model_dict, num_features, num_heads_diffusion)
    elif attention_type == "prod":
        attention = att_inner_product(data, model_dict, num_features, num_heads_diffusion)
    elif attention_type == "dist":
        attention = att_distance(data, model_dict, num_features, num_heads_diffusion)
    else:
        raise Exception(f'No such attention type {attention_type}.')
        
    nodes_features = nodes_features.view(-1, num_features)
    
    heads_attention = attention.squeeze(-1)
    attention = heads_attention.mean(dim=1,keepdim=False)
    adjusted_attention = adjust_attention(nodes_features, edge_index, attention)
    
    return attention, adjusted_attention


def att_weight_sum(data, model_dict, num_features, num_heads):
    
    scoring_fn_target = model_dict['diffusion.gnd_layer.attention_layer.scoring_fn_target']
    scoring_fn_source = model_dict['diffusion.gnd_layer.attention_layer.scoring_fn_source']
    
    attention_layer = Attention_Weight_Sum(num_features_diffusion = num_features, 
                                           num_of_heads = num_heads, 
                                           recover=True, 
                                           scoring_fn_target=scoring_fn_target, 
                                           scoring_fn_source=scoring_fn_source)
        
    attention, other = attention_layer(data)
    
    return attention

        
def att_inner_product(data, model_dict, num_features, num_heads):
    
    metric_weights = model_dict['diffusion.gnd_layer.attention_layer.metric_weights']
    
    attention_layer = Attention_Weight_Sum(num_features_diffusion = num_features, 
                                           num_of_heads = num_heads, 
                                           recover=True, 
                                           metric_weights=metric_weights)
        
    attention, other = attention_layer(data)
    
    return attention
        

def att_distance(data, model_dict, num_features, num_heads):
    
    edge_dims_weights = model_dict['diffusion.gnd_layer.attention_layer.edge_dims_weights']
    distance_dims_weights = model_dict['diffusion.gnd_layer.attention_layer.distance_dims_weights']
    
    attention_layer = Attention_Weight_Sum(num_features_diffusion = num_features, 
                                           num_of_heads = num_heads, 
                                           recover=True, 
                                           edge_dims_weights=edge_dims_weights, 
                                           distance_dims_weights=distance_dims_weights)
        
    attention, other = attention_layer(data)
    
    return attention


def adjust_attention(nodes_features, edge_index, attention):
    
    num_of_nodes = nodes_features.shape[0]
    
    edge_trg = edge_index[0]
    
    adjusted_attention = attention
    for i in range(num_of_nodes):
        count = edge_trg.eq(i).sum().item()
        adjusted_attention = adjusted_attention.where(edge_trg!=i, adjusted_attention*count)
        
    return adjusted_attention
    


    
    