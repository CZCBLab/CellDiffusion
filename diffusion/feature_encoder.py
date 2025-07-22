import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import info_log
from utils.utility_fn import extract_data_matrix_from_adata


def encode_features(adata, 
                    D_encode_list=[2000, 300, 50], 
                    D_decode_list=[50, 300, 2000], 
                    max_epoch=2000, 
                    lr=1e-3, 
                    device='cpu', 
                    data_dtype = torch.float32,
                    activation=nn.ELU(), 
                    encode_last_activation=False, 
                    decode_last_activation=False):
    
    info_log.print('--------> Starting feature encoder ...')
        
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=None, torch_tensor=True, data_dtype=data_dtype)

    model_fae = Feature_AutoEncoder(D_encode_list=D_encode_list, 
                                    D_decode_list=D_decode_list, 
                                    activation=activation, 
                                    encode_last_activation=encode_last_activation, 
                                    decode_last_activation=decode_last_activation).to(device)
                            
    optimizer = torch.optim.Adam(model_fae.parameters(), lr=lr)

    for epoch in range(max_epoch):
        model_fae.train()
        optimizer.zero_grad()

        feature_matrix_encoded, feature_matrix_recover = model_fae(feature_matrix.to(device), do_only=None)
        
        target_1 = torch.tensor(feature_matrix.to(device), dtype = feature_matrix_recover.dtype)
        
        loss = F.mse_loss(feature_matrix_recover, target_1, reduction='sum')
            
        # Backprop and Update
        loss.backward()
        cur_loss = loss.item()
        
        if epoch <= (max_epoch-2):
            optimizer.step()
        
        if epoch%50 == 0:
            info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")
    
    info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")
    
    
    adata.obsm['X_fae'] = feature_matrix_encoded.detach().cpu().numpy()
    
    
    #return adata


class Feature_AutoEncoder(nn.Module):

    def __init__(self, D_encode_list, D_decode_list, activation=nn.ELU(), 
                 encode_last_activation=False, decode_last_activation=False):
        
        super().__init__()
        self.activation = activation
        
        ## Encoder
        
        linear_layers = []
        for i in range(len(D_encode_list)-1):
            N_in = D_encode_list[i]
            N_out = D_encode_list[i+1]
            layer = nn.Linear(N_in, N_out, bias=False)
            
            # The default TF initialization
            nn.init.xavier_uniform_(layer.weight)
            
            linear_layers.append(layer)
            linear_layers.append(self.activation)
            
        if encode_last_activation==False: # Remove the last activation layer in the autoencoder
            linear_layers = linear_layers[:-1]
            
        self.encoder = nn.Sequential(*linear_layers,)
        
        ## Decoder
        
        linear_layers = []   
        for i in range(len(D_decode_list)-1):
            N_in = D_decode_list[i]
            N_out = D_decode_list[i+1]
            layer = nn.Linear(N_in, N_out, bias=False)
            
            # The default TF initialization
            nn.init.xavier_uniform_(layer.weight)
            
            linear_layers.append(layer)
            linear_layers.append(self.activation)
            
        if decode_last_activation==False: # Remove the last activation layer in the autoencoder
            linear_layers = linear_layers[:-1]
            
        self.decoder = nn.Sequential(*linear_layers,)
             
    def forward(self,data, do_only=None): 
        
        if do_only=='encode':
            data_encoded = self.encoder(data)
            
            return data_encoded
        
        elif do_only=='decode':    
            data_recover = self.decoder(data)
            
            return data_recover
            
        else:
            data_encoded = self.encoder(data)
            data_recover = self.decoder(data_encoded )

            return data_encoded, data_recover
    
    
    
    
    
    
    
