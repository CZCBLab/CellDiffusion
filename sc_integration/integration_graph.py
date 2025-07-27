import torch
import numpy
import pandas as pd
from functools import partial
import numpy as np
from sklearn.cluster import KMeans

from utils.utility_fn import *


def build_integration_graph(adata, 
                            batch_key='batch', 
                            use_rep='X_fae', 
                            n_edges_per_node=50, 
                            k_mnn=50, 
                            prune=False, 
                            device='cpu'):
    
    adata.obsm['node_batch_mt'] = pd.get_dummies(adata.obs[batch_key]).to_numpy()
    
    
    
    adata.obsm['X_ho'] = get_harmony_embeddings(adata, use_rep = use_rep)
    
    
    N_nodes = adata.obsm[use_rep].shape[0]
    
    
    if k_mnn != 0:
        
        node_batch_mt = adata.obsm['node_batch_mt']
    
        # numpy array
        edge_index_1 = get_inter_batch_knn_edge_index(node_batch_mt, adata.obsm[use_rep], k=k_mnn)

        # torch tensor
        edge_index_1 = batch_aware_limit_outgoing_edges(torch.tensor(edge_index_1, device=device), 
                                                        torch.tensor(node_batch_mt, device=device), 
                                                        max_edges=n_edges_per_node)


        #
        ## Fuzzy clustering alignment & mnn
        #
        
        # numpy array
        edge_index_2 = get_inter_batch_knn_edge_index(node_batch_mt, adata.obsm['X_ho'], k=k_mnn)

        
        # torch tensor
        edge_index_2 = remove_common_edges(edge_index=torch.tensor(edge_index_2, device=device), 
                                           edge_index_reference=edge_index_1)

        node_outgoing_counts_batch_mt = trg_batch_aware_node_edges_count(edge_index_1, 
                                                torch.tensor(node_batch_mt, device=device))
        
        node_batch_max_edges_mt = n_edges_per_node - node_outgoing_counts_batch_mt

        # torch tensor
        edge_index_2 = batch_and_node_aware_limit_outgoing_edges(edge_index_2, 
                                                                 torch.tensor(node_batch_mt, device=device), 
                                                                 node_batch_max_edges_mt)



        # torch tensor
        edge_index_12 = torch.cat((edge_index_1, edge_index_2), dim=1)
        edge_index_12 = limit_outgoing_edges(edge_index_12, max_edges = n_edges_per_node)
        
        
        #
        ## knn
        #

        # numpy array
        edge_index_3 = build_knn_edge_index_ckdtree(adata.obsm['X_ho'], k=n_edges_per_node)
        
        # torch tensor
        edge_index_3 = remove_common_edges(edge_index=torch.tensor(edge_index_3, device=device), 
                                           edge_index_reference=edge_index_12)


        outgoing_counts_12, incoming_counts_12 = node_edges_count(edge_index_12, N_nodes)
        node_aware_max_edges_12 = n_edges_per_node - outgoing_counts_12
        
        # torch tensor
        edge_index_3 = node_aware_limit_outgoing_edges(edge_index_3, 
                                                       node_aware_max_edges_12)
        
        # torch tensor
        edge_index = torch.cat((edge_index_12, edge_index_3), dim=1)
    
    else:
        # numpy array
        edge_index = build_knn_edge_index_ckdtree(adata.obsm['X_ho'], k=n_edges_per_node)
        # torch tensor
        edge_index = torch.tensor(edge_index, device=device)
        
    
    if prune:
        edge_index = prune_edges_with_IF_labels(edge_index, adata.obs['isolation'])

    
    outgoing_counts, incoming_counts = node_edges_count(edge_index, N_nodes)
    
    
    adata.uns['integration_edge_index'] = edge_index.cpu().numpy()
    
    adata.obs['incoming_counts'] = incoming_counts.cpu().numpy()
    adata.obs['outgoing_counts'] = outgoing_counts.cpu().numpy()          


def build_integration_loss_adj(adata, use_rep='X_fae', k=50, device='cpu'):
    
    node_batch_mt = extract_data_matrix_from_adata(adata, 
                                                    use_rep='node_batch_mt', 
                                                    torch_tensor=True, 
                                                    device=device)
    
    feature_matrix = extract_data_matrix_from_adata(adata, 
                                                    use_rep=use_rep, 
                                                    torch_tensor=True, 
                                                    device=device)
    N_nodes = feature_matrix.shape[0]
    
    this_array = []
    self_batch_edge_index_dict = {}
    for ii in range(node_batch_mt.shape[1]):
        batch_label = node_batch_mt[:, ii]
        batch_nodes = torch.where(batch_label == 1)[0]
        
        batch_features = feature_matrix[batch_nodes, :]
        knn_indices = feature_to_knn_indices(batch_features, 
                                             k_min=None, 
                                             k_max=k,
                                             self_included=True)
        
        edge_index = knn_indices_to_edge_index(knn_indices)
        
        label = 'batch_' + str(ii)
        
        self_batch_edge_index_dict[label] = edge_index.cpu().numpy()
        this_array.append(label)
        
    adata.uns['integration_loss_edge_index_dict'] = self_batch_edge_index_dict
    adata.uns['integration_loss_dict_index'] = this_array


def get_inter_batch_knn_edge_index(node_batch_mt, feature_matrix, k=50):
    
    N_nodes = node_batch_mt.shape[0]
        
    inter_batch_edge_index = numpy.empty((2, 0), dtype=np.int64)

    for ii in range(node_batch_mt.shape[1]):    # Go through all batches
        batch_trg = node_batch_mt[:, ii]
        nodes_trg = numpy.where(batch_trg == 1)[0]
        features_trg = feature_matrix[nodes_trg, :]

        nodes_src = numpy.where(batch_trg == 0)[0]
        features_src = feature_matrix[nodes_src, :]


        edge_index = build_mnn_edge_index_ckdtree(features_src, features_trg, k=k)                             

        edge_index[0,:] = nodes_src[edge_index[0,:]]
        edge_index[1,:] = nodes_trg[edge_index[1,:]]

        inter_batch_edge_index = np.concatenate((inter_batch_edge_index, edge_index), axis=1)

    return inter_batch_edge_index

            

def get_harmony_embeddings(adata, use_rep = 'X_fae'):

    data_mat = adata.obsm[use_rep]
    meta_data = adata.obs
    vars_use = 'batch'

    ho = run_harmony(data_mat, meta_data, vars_use)
    
    return ho.Z_corr.T


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta = None,
    lamb = None,
    sigma = 0.1, 
    nclust = None,
    tau = 0,
    block_size = 0.05, 
    max_iter_harmony = 10,
    max_iter_kmeans = 20,
    epsilon_cluster = 1e-5,
    epsilon_harmony = 1e-4, 
    plot_convergence = False,
    verbose = True,
    reference_values = None,
    cluster_prior = None,
    random_state = 0,
    cluster_fn = 'kmeans'
):
    """Run Harmony.
    """

    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells" 

    if nclust is None:
        nclust = np.min([np.round(N / 30.0), 100]).astype(int)

    if type(sigma) is float and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(theta, float) or isinstance(theta, int):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(lamb, float) or isinstance(lamb, int):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n)

    assert len(lamb) == np.sum(phi_n), \
        "each batch variable must have a lambda"

    # Number of items in each category.
    N_b = phi.sum(axis = 1)
    # Proportion of items in each category.
    Pr_b = N_b / N

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    lamb_mat = np.diag(np.insert(lamb, 0, 0))

    phi_moe = np.vstack((np.repeat(1, N), phi))

    np.random.seed(random_state)

    ho = Harmony(
        data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose,
        random_state, cluster_fn
    )

    return ho

class Harmony(object):
    def __init__(
            self, Z, Phi, Phi_moe, Pr_b, sigma,
            theta, max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state=None, cluster_fn='kmeans'
    ):
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi             = Phi
        self.Phi_moe         = Phi_moe
        self.N               = self.Z_corr.shape[1]
        self.Pr_b            = Pr_b
        self.B               = self.Phi.shape[0] # number of batch variables
        self.d               = self.Z_corr.shape[0]
        self.window_size     = 3
        self.epsilon_kmeans  = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb            = lamb
        self.sigma           = sigma
        self.sigma_prior     = sigma
        self.block_size      = block_size
        self.K               = K                # number of clusters
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose         = verbose
        self.theta           = theta

        self.objective_harmony        = []
        self.objective_kmeans         = []
        self.objective_kmeans_dist    = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross   = []
        self.kmeans_rounds  = []

        self.allocate_buffers()
        if cluster_fn == 'kmeans':
            cluster_fn = partial(Harmony._cluster_kmeans, random_state=random_state)
        self.init_cluster(cluster_fn)
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self):
        return self.Z_corr

    def allocate_buffers(self):
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat    = np.zeros((self.K, self.N))
        self.O           = np.zeros((self.K, self.B))
        self.E           = np.zeros((self.K, self.B))
        self.W           = np.zeros((self.B + 1, self.d))
        self.Phi_Rk      = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(data, K, random_state):
        # Start with cluster centroids
        model = KMeans(n_clusters=K, init='k-means++',
                       n_init=10, max_iter=25, random_state=random_state)
        model.fit(data)
        km_centroids, km_labels = model.cluster_centers_, model.labels_
        return km_centroids

    def init_cluster(self, cluster_fn):
        self.Y = cluster_fn(self.Z_cos.T, self.K).T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:,None]
        self.R -= np.max(self.R, axis = 0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis = 0)
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R , self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:,np.newaxis])
        # Cross Entropy
        x = (self.R * self.sigma[:,np.newaxis])
        y = np.tile(self.theta[:,np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            # STEP 1: Clustering
            self.cluster()
            # STEP 2: Regress out covariates
            # self.moe_correct_ridge()
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K,
                self.Phi_Rk, self.Phi_moe, self.lamb
            )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            if converged:
                break

        return 0

    def cluster(self):
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            # print("kmeans {}".format(i))
            # STEP 1: Update Y
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0

    def update_R(self):
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:,None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        # Update cells in blocks
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            # STEP 1: Remove cells
            self.E -= np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:,b], self.Phi[:,b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:,b] = self._scale_dist[:,b]
            self.R[:,b] = np.multiply(
                self.R[:,b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta),
                    self.Phi[:,b]
                )
            )
            self.R[:,b] = self.R[:,b] / np.linalg.norm(self.R[:,b], ord=1, axis=0)
            # STEP 3: Put cells back
            self.E += np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:,b], self.Phi[:,b].T)
        return 0

    def check_convergence(self, i_type):
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True


def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y

def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i,:])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0,:] = 0 # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk
