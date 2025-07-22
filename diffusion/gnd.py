# # The Code of Graph Neural Diffusion Networks-PyTorch

import torch
import torch.nn as nn

class GND(nn.Module):
    """
    """
    def __init__(self, num_features_diffusion, num_heads_diffusion=8,
                 num_steps_diffusion=8, time_increment_diffusion=0.2,
                 attention_type = 'sum', activation=nn.ELU(),
                 dropout=0.0,  
                 log_diffusion=False,
                 encoder=None, decoder=None,
                 rebuild_graph=False):
        super().__init__()

        self.num_features_diffusion = num_features_diffusion
        self.num_steps_diffusion = num_steps_diffusion
        self.num_heads_diffusion = num_heads_diffusion
        self.rebuild_graph = rebuild_graph

        self.log_diffusion = log_diffusion
        self.diffusion_step_outputs = []# Record output each diffusion steps
        
        self.gnd_input = None
        
        self.activation = activation
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        
        
        # Since the diffusion process keeps node feature's dimension according to paper arXiv:2106.10934v2,
        # we use the encoder(decoder) to change node feature's dimension from the input(difffusion) dimension
        # to the diffusion(output) dimension if it is needed.
        if encoder is not None:
            assert encoder[-1] == self.num_features_diffusion, f'"encoder" should be a vector with the length represents '\
                    f'the number of encoder layers and the components represents the dimension of features in each layer. '\
                    f'Note the first component should be the dimension of the input features and the last component '\
                    f'should equal to "num_features_diffusion".'
            self.encoder = AutoEncoder(encoder, activation = self.activation,last_activation=False)
        if decoder is not None:
            assert decoder[0] == self.num_features_diffusion, f'"decoder" should be a vector with the length represents '\
                    f'the number of decoder layers and the components represents the dimension of features in each layer. '\
                    f'Note the first component should equal to "num_features_diffusion" and the last component '\
                    f'should be the dimension of the output features.'
            self.decoder = AutoEncoder(decoder, activation = self.activation,last_activation=False)

        # Load graph diffusion networks layer
        self.gnd_layer = GNDLayer(
                 num_features_diffusion=self.num_features_diffusion, 
                 num_of_heads = num_heads_diffusion, 
                 time_increment_diffusion = time_increment_diffusion,
                 attention_type = attention_type,
                 activation=self.activation,
                 dropout_prob = dropout, 
                 log_attention_weights=False,
                 rebuild_graph=rebuild_graph)


    def forward(self, data):
        """
        data is just a (in_nodes_features, edge_index) tuple if self.rebuild_graph==False else 
        (in_nodes_features, rebuild_graph_args), where rebuild_graph_args is an args dict.
        """
        #
        # Step 1: Data pre-processing and encoding
        #
        
        # Change the node feature dimesion: [N, NFIN] -> [N, NFDIF], where N is the number of nodes, NFIN is
        # the dimension of input node feature and NFDIF is the dimension of features in the diffusion process.
        data = self.encoder(data)
        
        if self.rebuild_graph:
            feature_matrix, rebuild_graph_args = data

            k_min = rebuild_graph_args['k_min']
            k_max = rebuild_graph_args['k_max']
            remov_edge_prob = rebuild_graph_args['remov_edge_prob']
            
            data = (feature_matrix, (k_min, k_max, remov_edge_prob))
        
        #
        # Step 2: Diffusion Process
        #

        # Do diffusion. Feature dimension keeped: [N, NFDIF] -> [N, NFDIF].
        for it in range(self.num_steps_diffusion):
            
            # Log diffusion output data
            if self.log_diffusion:
                if len(self.diffusion_step_outputs) == (self.num_steps_diffusion+1):
                    self.diffusion_step_outputs = []  # clear self.diffusion_step_outputs
                
                self.diffusion_step_outputs.append(data[0].detach().cpu())
                
            data = self.gnd_layer(data)
            
        
        self.diffusion_step_outputs.append(data[0].detach().cpu())


        #
        # Step 3: Data decoding and outputting
        #
        
        # record input data with dimension NDIF
        last_embeding, a = data
        
        # Change the node feature dimesion: [N, NFDIF] -> [N, NFOUT], where NFOUT is the dimension of output features.
        data = self.decoder(data)
        
        return data, last_embeding
    
    #
    # Helper functions
    #

class AutoEncoder(nn.Module):

    def __init__(self, num_features_list, activation, last_activation, pre_activation=False):
        
        super().__init__()
        self.activation = activation
        
        linear_layers = []
        if pre_activation: # Add an activation layer in front of autoencoder
            linear_layers.append(self.activation)
            
        for i in range(len(num_features_list)-1):
            N_in = num_features_list[i]
            N_out = num_features_list[i+1]
            layer = nn.Linear(N_in, N_out, bias=False)
            
            # The default TF initialization
            nn.init.xavier_uniform_(layer.weight)
            
            linear_layers.append(layer)
            linear_layers.append(self.activation)
            
        if last_activation==False: # Remove the last activation layer in the autoencoder
            linear_layers = linear_layers[:-1]
            
        self.autoencoder = nn.Sequential(*linear_layers,)
             
    def forward(self,data): 
        in_nodes_features, edge_index = data
        out_nodes_features = self.autoencoder(in_nodes_features)

        return out_nodes_features, edge_index



class GNDLayer(torch.nn.Module):
    """
    This is built based on pytorch-GAT code:
    https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py, 
    where the Implementation #3 is used.
    
    We here introduce three different attention types. You can find the definitions in 
    class Attention_Weight_Sum, class Attention_Inner_Product and class Attention_Distance.

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    # The feature flow from source to target
    src_nodes_dim = 1  # position of source nodes in edge index
    trg_nodes_dim = 0  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_features_diffusion, num_of_heads, time_increment_diffusion,
                 attention_type="sum", activation=nn.ELU(),
                 dropout_prob=0.6, log_attention_weights=False, rebuild_graph=False):

        super().__init__()

        self.num_features_diffusion = num_features_diffusion
        self.num_of_heads = num_of_heads
        self.time_increment_diffusion = time_increment_diffusion
        
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.rebuild_graph = rebuild_graph

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        
        # Load attention calculation networks layer
        self.attention_layer = self.get_attention(attention_type)

        
    def forward(self, data):
        
        if self.rebuild_graph:
            in_nodes_features, rebuild_args = data  # unpack data
            edge_index = features_to_edge_index_knn_no_self_edge(in_nodes_features, rebuild_args)
            
        else:
            in_nodes_features, edge_index = data  # unpack data
        
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        
        #
        # Step 1: Data pre-processing
        #

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features 
        in_nodes_features = self.dropout(in_nodes_features)
            
        # shape: (N, FOUT) -> (N, 1, FOUT)
        in_nodes_features = in_nodes_features.view(-1, 1, self.num_features_diffusion)
        
        #
        # Step 2: Edge attention calculation
        #

        # shape = ((N, 1, FOUT), (2, E)), E - number of edges in the graph
        data = (in_nodes_features, edge_index)
        
        # attention shape = (E, NH, 1), nodes_features_proj_source shape = (E, NH, FOUT), E - number of edges in the graph
        attentions_per_edge, nodes_features_proj_source = self.attention_layer(data)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_source_weighted = nodes_features_proj_source * attentions_per_edge

        # This part sums up weighted neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        nodes_features_aggregate = self.aggregate_neighbors(nodes_features_proj_source_weighted, edge_index, in_nodes_features, num_of_nodes)
        
        # shape: (N, NH, FOUT) -> (N, 1, FOUT)
        nodes_features_aggregated = nodes_features_aggregate.mean(dim=1,keepdim=True)

        #
        # Step 4: Propagation
        #

        # propagate neighbors aggregate features to nodes
        # shape = (N, NFOUT)
        out_nodes_features = self.propagate(in_nodes_features, nodes_features_aggregated)
        
        
        
        #
        # Step 5: log attention weights
        #
        
        if self.log_attention_weights:  # potentially log for later visualization
            self.attention_weights = attentions_per_edge
            
        
        if self.rebuild_graph:
            return out_nodes_features, rebuild_args
        
        else:
            return out_nodes_features, edge_index

    #
    # Helper functions
    #

    def aggregate_neighbors(self, nodes_features_proj_source_weighted, edge_index, in_nodes_features, num_of_nodes):
        # size = (E, NH, FOUT)
        size = list(nodes_features_proj_source_weighted.shape)  # convert to list otherwise assignment is not possible
        # size = (N, NH, FOUT)
        size[self.nodes_dim] = num_of_nodes 
        # shape = (N, NH, FOUT)
        aggregated_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_source_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        aggregated_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_source_weighted)

        return aggregated_nodes_features

    
    def propagate(self, in_nodes_features, nodes_features_aggregated):
        
        if isinstance(self.time_increment_diffusion, (int, float)):
            out_nodes_features = self.time_increment_diffusion*nodes_features_aggregated + (1.0-self.time_increment_diffusion)* in_nodes_features
        else:
            tau = torch.tensor(self.time_increment_diffusion, dtype=in_nodes_features.dtype, device=in_nodes_features.device) 
            repeat_times = list(in_nodes_features.shape)
            repeat_times[0]=1
            tau = tau.repeat(*repeat_times).reshape(in_nodes_features.shape)
            
            out_nodes_features = tau*nodes_features_aggregated + (1.0-tau)* in_nodes_features
            
            
            
            
        out_nodes_features = out_nodes_features.view(-1, self.num_features_diffusion)

        return out_nodes_features # if self.activation is None else self.activation(out_nodes_features)
    
    
    def get_attention(self, attention_type):
    
        if attention_type == "sum":
            attention_layer = Attention_Weight_Sum(
                                            num_features_diffusion=self.num_features_diffusion, 
                                            num_of_heads=self.num_of_heads)
        elif attention_type == "prod":
            attention_layer = Attention_Inner_Product(
                                            num_features_diffusion=self.num_features_diffusion, 
                                            num_of_heads=self.num_of_heads)
        elif attention_type == "dist":
            attention_layer = Attention_Distance(
                                            num_features_diffusion=self.num_features_diffusion, 
                                            num_of_heads=self.num_of_heads)
        else:
            raise Exception(f'"attention_type" should be one of ("sum", "prod", "dist") but got "{attention_type}".')
            
        return attention_layer


#
# Three attention types
#

class Attention_Weight_Sum(torch.nn.Module):
    """
    The same attention with that in pytorch-GAT code:
    https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py, 
    which is first defined in paper: arXiv:1710.10903.
    
    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 1  # position of source nodes in edge index
    trg_nodes_dim = 0  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_features_diffusion, num_of_heads, recover=False, scoring_fn_target=None, scoring_fn_source=None):

        super().__init__()
    
        if recover:
            self.scoring_fn_target = scoring_fn_target
            self.scoring_fn_source = scoring_fn_source
    
        else:
            self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_features_diffusion))
            self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_features_diffusion))
            self.init_params()
        
        
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2, no need to expose every setting
        
        
    def forward(self,data):
        
        # shape = ((N, 1, FOUT), (2, E))
        nodes_features, edge_index = data # unpack data
        
        num_of_nodes = nodes_features.shape[self.nodes_dim]
        
        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, 1, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores_lifted shape = (E, NH), src_nodes_features_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_lifted, src_nodes_features_lifted = self.lift(scores_source, scores_target, nodes_features, edge_index)
        # shape = (E, NH)
        scores_per_edge = self.leakyReLU(scores_lifted)
        
        # shape = (E, NH, 1)
        attentions_per_edge = neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim)
        
        
        return attentions_per_edge, src_nodes_features_lifted

        
    def lift(self, scores_source, scores_target, nodes_features, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source_lifted = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        src_nodes_features_lifted = nodes_features.index_select(self.nodes_dim, src_nodes_index)
        
        scores_lifted = scores_source_lifted + scores_target_lifted

        return scores_lifted, src_nodes_features_lifted
    
    def init_params(self):
        """
        The default TF initialization: https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        """
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        
        

class Attention_Inner_Product(torch.nn.Module):
    """
    We use the inner product attention matrix which is first introduced in paper: arXiv:1706.03762v5.

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 1  # position of source nodes in edge index
    trg_nodes_dim = 0  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_features_diffusion, num_of_heads, recover=False, metric_weights=None):

        super().__init__()
        self.num_features_diffusion = num_features_diffusion
        self.num_of_heads = num_of_heads
    
    
        if recover:
            self.metric_weights = metric_weights
        else:
            self.metric_weights = nn.Parameter(torch.Tensor(1, num_of_heads, num_features_diffusion, num_features_diffusion))
            self.init_params()
        
        
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2, no need to expose every setting
        
        
    def forward(self,data):
        nodes_features, edge_index = data # unpack data
        
        num_of_nodes = nodes_features.shape[self.nodes_dim]
        
        # (N, 1, FOUT) --> (E, 1, FOUT)
        nodes_features_source, nodes_features_target = self.lift(nodes_features, edge_index)
        
        # (E, 1, FOUT) -> (E, 1, FOUT, 1)
        nodes_features_target = nodes_features_target.view(-1, 1, self.num_features_diffusion, 1)
        
        # (1, NH, FOUT, FOUT) matmul (E, 1, FOUT, 1) -> (E, NH, FOUT, 1) -> (E, NH, FOUT)
        nodes_features_target = self.metric_weights.matmul(nodes_features_target).view(-1, self.num_of_heads, self.num_features_diffusion)

        # shape = (E, 1, FOUT) * (E, NH, FOUT) -> (E, NH, FOUT) -> (E, NH) because sum squeezes the last dimension
        # (* represents element-wise (a.k.a. Hadamard) product)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        edge_inner_product_values = (nodes_features_source * nodes_features_target).sum(dim=-1)
        edge_inner_product_scores = self.leakyReLU(edge_inner_product_values)
        
        # shape = (E, NH, 1)
        attentions_per_edge = neighborhood_aware_softmax(edge_inner_product_scores, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim)                          
        
        return attentions_per_edge, nodes_features_source

        
    def lift(self, nodes_features, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        # Shape: (N, 1, NF) -> (E, 1, NF)
        nodes_features_sources = nodes_features.index_select(self.nodes_dim, src_nodes_index)
        nodes_features_target = nodes_features.index_select(self.nodes_dim, trg_nodes_index)
        
        return nodes_features_sources, nodes_features_target
    
    def init_params(self):
        """
        The default TF initialization: https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        """
        nn.init.xavier_uniform_(self.metric_weights)


class Attention_Distance(torch.nn.Module):
    """
    We construct attention matrix based on the distance between two nodes.
    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 1  # position of source nodes in edge index
    trg_nodes_dim = 0  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_features_diffusion, num_of_heads, recover=False, edge_dims_weights=None, distance_dims_weights=None):

        super().__init__()
        
        if recover:
            self.edge_dims_weights = edge_dims_weights
            self.distance_dims_weights = distance_dims_weights
    
        else:
            self.edge_dims_weights = nn.Parameter(torch.Tensor(1, num_of_heads, num_features_diffusion))
            self.distance_dims_weights = nn.Parameter(torch.Tensor(1, num_of_heads, num_features_diffusion))
            self.init_params()

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2, no need to expose every setting
        
        
    def forward(self,data):
        nodes_features, edge_index = data # unpack data
        
        num_of_nodes = nodes_features.shape[self.nodes_dim]
        
       # (N, NH, FOUT) --> edge_index lift --> (E, NH, FOUT)
        edge_vectors, nodes_features_source = self.edge_vectors(nodes_features, edge_index)

        # Apply edge dimensional weights (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (E, NH, FOUT) * (1, NH, FOUT) -> (E, NH, FOUT) 
        edge_vectors_weighted = (edge_vectors * self.edge_dims_weights)

        # calculation dimension-wise distance 
        # shape = (E, NH, FOUT) * (E, NH, FOUT) -> (E, NH, FOUT)
        edge_distance_vectors = (edge_vectors_weighted * edge_vectors_weighted)
        
        # Apply distance dimensional weights
        # shape = (E, NH, FOUT) * (1, NH, FOUT) -> (E, NH, FOUT) -> (E, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        edge_distances = (edge_distance_vectors*self.distance_dims_weights).sum(dim=-1)

        # (E, NH) -> (1,NH)
        edge_distance_mean_values = edge_distances.mean(dim=0,keepdim=True)
        
        # (E, NH) + (1,NH) -> (E, NH)
        edge_distance_scores = -1.0 * self.leakyReLU(edge_distances + edge_distance_mean_values)
        
        # shape = (E, NH, 1)
        attentions_per_edge = neighborhood_aware_softmax(edge_distance_scores, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim) 
        
        return attentions_per_edge, nodes_features_source

        
    def edge_vectors(self, nodes_features, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        Compute edge vectors.

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        nodes_features_sources = nodes_features.index_select(self.nodes_dim, src_nodes_index)
        nodes_features_target = nodes_features.index_select(self.nodes_dim, trg_nodes_index)
        edge_vectors_matrix = nodes_features_target - nodes_features_sources
        
        return edge_vectors_matrix, nodes_features_sources
    
    def init_params(self):
        """
        The default TF initialization: https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        """
        nn.init.xavier_uniform_(self.edge_dims_weights)
        nn.init.xavier_uniform_(self.distance_dims_weights)


        
# useful functions for attention calculation

def neighborhood_aware_softmax(scores_per_edge, trg_index, num_of_nodes, nodes_dim):
    """
    As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
    Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
    into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
    in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
    (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
     i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

    Note:
    Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
    and it's a fairly common "trick" used in pretty much every deep learning framework.
    Check out this link for more details:

    https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

    """
    # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
    scores_per_edge = scores_per_edge - scores_per_edge.max()
    exp_scores_per_edge = scores_per_edge.exp()  # softmax

    # Calculate the denominator. shape = (E, NH)
    neigborhood_aware_denominator = sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes, nodes_dim=nodes_dim)

    # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
    # possibility of the computer rounding a very small number all the way to 0.
    attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

    # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
    return attentions_per_edge.unsqueeze(-1)

def sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes, nodes_dim):
    # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
    trg_index_broadcasted = explicit_broadcast(trg_index, exp_scores_per_edge)

    # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
    size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
    size[nodes_dim] = num_of_nodes
    neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

    # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
    # target index)
    neighborhood_sums.scatter_add_(nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

    # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
    # all the locations where the source nodes pointed to i (as dictated by the target index)
    # shape = (N, NH) -> (E, NH)
    return neighborhood_sums.index_select(nodes_dim, trg_index)

def explicit_broadcast(this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    
    
def features_to_edge_index_knn_no_self_edge(feature_matrix, rebuild_args):
    """
    Compute the k-nearest neighbors graph from a feature matrix.

    Args:
        feature_matrix (torch.Tensor): A tensor of shape (num_points, num_features).
        k (int): The number of nearest neighbors to consider.

    Returns:
        networkx.Graph: A NetworkX graph representing the k-nearest neighbors graph.
    """
    k_min, k_max, remov_edge_prob = rebuild_args
    # Calculate the pairwise squared distances between points
    dist_matrix = torch.cdist(feature_matrix, feature_matrix, p=2)
    # Find the indices of the k nearest neighbors for each point
    knn_indices = torch.argsort(dist_matrix, dim=1)[:, k_min+1:k_max+1] # the point itself may not at index 0(overlapped points)
    # construct edge index from knn_indices
    edge_index = knn_indices_to_edge_index(knn_indices)
    
    # randomly remove edges
    if remov_edge_prob is not None:
        mask = torch.rand(edge_index.size(1)) > remov_edge_prob
        edge_index = edge_index[:, mask]
    
    # remove left self-edges
    # Compute the mask
    mask = edge_index[0, :] != edge_index[1, :]
    # Filter the tensor using the mask
    edge_index = edge_index[:, mask]

    
    return edge_index


def knn_indices_to_edge_index(knn_indices):
    """
    Convert a knn_indices tensor to an edge_index tensor.

    Args:
        knn_indices (torch.Tensor): A tensor of shape (num_points, k) containing the indices of 
                                    the k-nearest neighbors for each point.

    Returns:
        torch.Tensor: An edge_index tensor of shape (2, num_edges) representing the edges in the graph.
    """
    num_points, k = knn_indices.shape

    # Create source and target node index tensors
    src_nodes = torch.arange(num_points, device=knn_indices.device).view(-1, 1).repeat(1, k).view(-1)
    trg_nodes = knn_indices.reshape(-1)

    # Concatenate the source and target node index tensors to create the edge_index tensor
    edge_index = torch.stack([src_nodes, trg_nodes], dim=0)

    return edge_index

