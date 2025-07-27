"""

"""

import numpy as np
from utils import info_log
import matplotlib.pyplot as plt
from utils.utility_fn import extract_data_matrix_from_adata

def fast_preprocess(adata):
    info_log.print('--------> Do fast preprocessing for single cell data.')

    adata = scale_and_log_transformation(adata, scale=True, scale_factor = 10000, log=True)
    adata = find_highly_variable_genes(adata, n_top_genes=2000, method='product', plot=True)



def scale_and_log_transformation(adata, scale=True, scale_factor = 10000, log=True):
    
    data = np.array(extract_data_matrix_from_adata(adata, use_rep=None, torch_tensor=False))
    
    if scale:
        info_log.print(f'---------> Scale data: scale_factor = {scale_factor}.')
        for row in range(data.shape[0]):
            data[row] /= np.sum(data[row])/scale_factor
    
    if log:
        info_log.print(f'---------> Log_transformation.')
        data = np.log(data + 1)
    
    adata.X = data
    



def find_highly_variable_genes(adata, n_top_genes=2000, method='product', plot=True):
    info_log.print('---------> Sorting and selecting top highly variable genes.')

    expr = np.array(extract_data_matrix_from_adata(adata, use_rep=None, torch_tensor=False))
    
    variation = expr.var(axis=0, keepdims=False)
    mean = expr.mean(axis=0, keepdims=False)
    product = mean*variation
    
    adata.var['variations'] = variation
    adata.var['means'] = mean
    adata.var['var_mean_products'] = product
    
    adata.uns['hvg'] = {'n_top_genes': n_top_genes,
                        'method': method,
    }

    if method=='product':
        idx_data = product
    elif method=='variation':
        idx_data = variation
    elif method=='mean':
        idx_data = mean

    gene_idx = idx_data.argsort()[::-1][:n_top_genes]
    
    adata.var['highly_variable']=False
    adata.var['highly_variable'][gene_idx] = True

    if plot:
        plot_variable_gene(variation, mean, gene_idx)
        

def plot_variable_gene(variation, mean, index):
    
    mean_s = mean[index]
    variation_s = variation[index]

    plt.scatter(mean, variation)

    plt.scatter(mean_s, variation_s, color='red')

    plt.title('Variable genes plot')
    plt.xlabel('mean')
    plt.ylabel('variation')

    plt.show()


    










