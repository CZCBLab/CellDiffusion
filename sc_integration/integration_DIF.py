"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.gnd import GND
from utils import info_log
from utils.utility_fn import extract_data_matrix_from_adata

from sc_integration.integration_graph import build_integration_graph, build_integration_loss_adj


def integration_diffusion(adata, 
                            use_rep='X_fae', 
                            save_key='X_dif', 
                            max_epoch=2000, 
                            lr=1e-3, 
                            device='cpu',
                            num_features_diffusion=50,
                            num_heads_diffusion=8,
                            num_steps_diffusion=8, 
                            time_increment_diffusion=0.2,
                            attention_type = 'sum', 
                            activation=nn.ELU(),
                            data_dtype = torch.float32,
                            dropout=0.0, 
                            log_diffusion=False,
                            encoder=None, 
                            decoder=[300],
                            save_model = True,
                            load_model_state = False,
                            loss_reduction = "sum"):
    
    diffusion_args = {"use_rep": use_rep,
                       "num_features_diffusion": num_features_diffusion,
                       "num_heads_diffusion": num_heads_diffusion,
                       "num_steps_diffusion": num_steps_diffusion, 
                       "time_increment_diffusion": time_increment_diffusion,
                       "attention_type": attention_type, 
                       "dropout": dropout, 
                       "log_diffusion": log_diffusion,
                       "encoder": encoder, 
                       "decoder": decoder,
                       "save_model": save_model,
                       "load_model_state": load_model_state}
    
    
    info_log.print('--------> Starting data integration ...')
    
    # data
    feature_matrix = extract_data_matrix_from_adata(adata, 
                                                    use_rep=use_rep, 
                                                    torch_tensor=True, 
                                                    data_dtype=data_dtype, 
                                                    device=device)
        
    edge_index = torch.tensor(adata.uns['integration_edge_index'], dtype=torch.int64, device=device)
    
    num_of_nodes = feature_matrix.shape[0]
    
    node_batch_mt = torch.tensor(adata.obsm['node_batch_mt'], device=device)
    node_batch_mask = node_batch_mt.to(torch.bool)
    
    adjacency_list = []
    
    for ii in range(node_batch_mt.size(1)):
        N_nodes_batch = node_batch_mt[:,ii].sum().item()
        batch_dict_index = adata.uns['integration_loss_dict_index'][ii]
        edge_index_now = torch.tensor(adata.uns['integration_loss_edge_index_dict'][batch_dict_index], 
                                      dtype=torch.int64)

        adjacency_now = edge_index_to_adj(edge_index_now.to(device), N_nodes_batch)
        adjacency_list.append(adjacency_now)

    D_in = feature_matrix.shape[1]
    D_out = D_in
    
    if encoder is None:
        encoder = None if D_in==num_features_diffusion else [D_in, num_features_diffusion]
    else:
        encoder = [D_in] + encoder + [num_features_diffusion]
    
    if decoder is None:
        decoder = None if D_out==num_features_diffusion else [num_features_diffusion, D_out]
    else:
        decoder = [num_features_diffusion] + decoder + [D_out]
        

    model_gae = Graph_DIF(num_features_diffusion = num_features_diffusion, 
                           num_heads_diffusion=num_heads_diffusion,
                           num_steps_diffusion= num_steps_diffusion, 
                           time_increment_diffusion=time_increment_diffusion,
                           attention_type = attention_type, 
                           activation=activation,
                           dropout=dropout, 
                           log_diffusion=log_diffusion,
                           encoder=encoder, 
                           decoder=decoder,
                           rebuild_graph=False,
                           node_batch_one_hot=node_batch_mt).to(device)

    if load_model_state:
        try: 
            state_dict_torch = {k: torch.tensor(v).to(device) for k, v in adata.uns['gnd_state_dict'].items()}
            model_gae.load_state_dict(state_dict_torch)
        except:
            print("Graph autoencoder failed to load model state.")
                            
    optimizer = torch.optim.Adam(model_gae.parameters(), lr=lr)

    for epoch in range(max_epoch):
        model_gae.train()
        optimizer.zero_grad()
        
        data = (feature_matrix, edge_index)

        out_nodes_features, recon_adj_list, last_embedding = model_gae(data)
        
        loss_list = []
        for ii in range(len(adjacency_list)):
            target_now = torch.tensor(adjacency_list[ii].to(device), dtype = recon_adj_list[ii].dtype)
            loss_now = F.binary_cross_entropy_with_logits(recon_adj_list[ii], target_now, reduction=loss_reduction)
            loss_list.append(loss_now)
            
        loss = loss_list[0]
        for ii in range(1, len(loss_list)):
            loss += loss_list[ii]
        

        # Backprop and Update
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        
        if epoch%50 == 0:
            info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")
    
    info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")

    # save model state
    if save_model:
        state_dict_numpy = {k: v.detach().cpu().numpy() for k, v in model_gae.state_dict().items()}
        adata.uns['gnd_state_dict'] = state_dict_numpy

        
    if log_diffusion:
        adata.uns['gnd_steps_data'] = []
        for it in range(len(model_gae.diffusion_step_outputs)):
            adata.uns['gnd_steps_data'].append(model_gae.diffusion_step_outputs[it].numpy())
    
    if save_key is None:
        adata.obsm['X_dif'] = last_embedding.detach().cpu().numpy()
    else:
        adata.obsm[save_key] = last_embedding.detach().cpu().numpy()
        
    adata.uns['graph_diffusion_args'] = diffusion_args
        
    #return adata 


class Graph_DIF(nn.Module):
    def __init__(self, num_features_diffusion,
                           num_heads_diffusion,
                           num_steps_diffusion, 
                           time_increment_diffusion,
                           attention_type = 'sum', 
                           activation=nn.ELU(),
                           dropout=0.0,  
                           log_diffusion=False,
                           encoder=None, 
                           decoder=None,
                           rebuild_graph=False,
                           node_batch_one_hot=None):
        super().__init__()
        
        self.log_diffusion=log_diffusion
        
        self.attention_weights = None
        self.diffusion_step_outputs = None
        self.node_batch_mask = node_batch_one_hot.to(torch.bool)
        
        self.diffusion = GND(num_features_diffusion = num_features_diffusion, 
                           num_heads_diffusion=num_heads_diffusion,
                           num_steps_diffusion= num_steps_diffusion, 
                           time_increment_diffusion=time_increment_diffusion,
                           attention_type = attention_type, 
                           activation=activation,
                           dropout=dropout, 
                           log_diffusion=log_diffusion,
                           encoder=encoder, 
                           decoder=decoder,
                           rebuild_graph=rebuild_graph)

        self.decode = InnerProductDecoder(0, act=lambda x: x)
        #self.decode = InnerProductDecoder(0, act=torch.sigmoid)


    def forward(self, data):
        
        data, last_embedding = self.diffusion(data)
        
        out_nodes_features, edge_index = data
        
        recon_adj_list = []
        for ii in range(self.node_batch_mask.size(1)):
            
            out_features_now = out_nodes_features[self.node_batch_mask[:, ii],:]
            recon_adj_now = self.decode(out_features_now)
            recon_adj_list.append(recon_adj_now)

            
        if self.log_diffusion:
            self.diffusion_step_outputs = self.diffusion.diffusion_step_outputs
        
        return out_nodes_features, recon_adj_list, last_embedding
        
    
    
class InnerProductDecoder(nn.Module):
    """
    """

    def __init__(self, dropout, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
    
def edge_index_to_adj(edge_index, num_of_nodes):
    """
    construct adjacency matrix from edge index
    """
    adjacency_matrix = torch.zeros((num_of_nodes, num_of_nodes), dtype=edge_index.dtype, device=edge_index.device)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1

    return adjacency_matrix


def integration_high_throughput_mode(adata, 
                            batch_key='batch', 
                            use_rep='X_fae', 
                            max_epoch=2000, 
                            lr=1e-3, 
                            time_increment_diffusion=0.5,
                            device='cpu'):
    
    info_log.print('--------> Build diffusion graph for data integration ...')
    
    build_integration_graph(adata, 
                            batch_key=batch_key, 
                            use_rep=use_rep, 
                            n_edges_per_node=50, 
                            k_mnn=0, 
                            prune=False, 
                            device=device)
    
    info_log.print('--------> Diffusion graph is completed.')
    
    info_log.print('--------> Build KNN_adj for loss function ...')
    
    build_integration_loss_adj(adata, use_rep=use_rep, k=50, device=device)
    
    info_log.print('--------> KNN_adj is completed.')
    
    integration_diffusion(adata, 
                              use_rep=use_rep,
                              max_epoch=max_epoch, 
                              lr=lr, 
                              time_increment_diffusion=time_increment_diffusion,
                              num_features_diffusion=8,
                              num_heads_diffusion=6,
                              num_steps_diffusion=2,  
                              log_diffusion=False,
                              device=device)
    
    

