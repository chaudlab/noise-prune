import sys, os
import numpy as np
import numpy.linalg as nla
import linear_net_fns as lnf

prev_gen_fn_dir = os.path.abspath('..') + '/2020_02_general_fns/'
sys.path.append(prev_gen_fn_dir)

import lin_alg_fns as laf

def symmetrize_matrix(W):
    '''Meant to be used on an triangular matrix (though of course it'll
    symmetrize anything). '''

    return W + W.T - np.diag(W.diagonal())

def get_edge_list_from_adj_matrix(W, is_symmetric=False, get_wts=False):
    '''Test this again, though glanced at it and looks sane.
    Though fix the non-symmetric case in that it returns self-connections.'''
    
    if is_symmetric:
        edge_list_idx = np.nonzero(np.triu(W,1))
    else:
        edge_list_idx = np.nonzero(W)
    
    # In both cases was returning np.transpose(edge_list_idx) before   
    if get_wts:
        edge_wts = W[edge_list_idx]
        return edge_list_idx, edge_wts
    else:
        return edge_list_idx
    
def make_random_triu_matrix_given_density(n_vertices, edges, entry_type='uniform_positive', 
    params={}, symmetrize=False):
    '''Generate a random adjacency matrix for a symmetric weighted graph. 
    If edges is a fraction then treat it as a desired density. Otherwise as a count.
    If symmetrize is False then just return an upper triangular matrix.'''

    G = np.zeros((n_vertices, n_vertices))

    if edges<1:
        # Convert density into number of edges. 
        n_edges = int(edges * n_vertices * (n_vertices-1) / 2.)
    else:
        # Assumes that the number of edges = number of nonzero
        # matrix entries, so divide by 2 because we'll symmetrize
        n_edges = edges // 2

    all_row_idx, all_col_idx = np.triu_indices(n_vertices, k=1)
    sel_idx_idx = np.random.choice(len(all_row_idx), size=n_edges, replace=False)
    sel_row_idx = all_row_idx[sel_idx_idx]
    sel_col_idx = all_col_idx[sel_idx_idx] 

    if entry_type == 'uniform_positive':
        edge_weights = np.random.rand(len(sel_row_idx))
    elif entry_type == 'normal':
        edge_weights = params['mean'] + params['sigma'] * np.random.randn(len(sel_row_idx))
    
    G[sel_row_idx, sel_col_idx] = edge_weights

    if symmetrize:
        return symmetrize_matrix(G)
    else:
        return G
    
def make_random_asymm_matrix_given_density(n_vertices, edges, entry_type='uniform_positive', 
    params={}):
    '''run the above function twice and add the triangular matrices G1 + G2.T'''
    G1 = make_random_triu_matrix_given_density(n_vertices, edges, entry_type=entry_type, 
    params=params)
    G2 = make_random_triu_matrix_given_density(n_vertices, edges, entry_type=entry_type, 
    params=params)
    return G1 + G2.T

def make_imb_asymm_clustered_graph(n_per_block, inner_block_edges, inter_block_edges):
    '''This is similar to undirected_graph_fns' make_imb_clustered_graph function,
    but uses asymmetric matrices for each cluster and asymmetric long range connections'''
    
    # I think could use np.sum(n_per_block) instead
    n_vertices = 0
    for each_block in n_per_block:
        n_vertices = n_vertices + each_block

    # Local within cluster connections    
    W_without_lr = np.zeros((n_vertices, n_vertices))
    graph_params = {'mean' : 1., 'sigma' : 1.}
    i1 = 0
    for each_block in n_per_block:
        curr_mat = make_random_asymm_matrix_given_density(each_block, edges=inner_block_edges, 
            entry_type='normal', params=graph_params)
        # Maybe i1 = i0 + each_block?
        i0 = i1
        i1 = i1 + each_block
        W_without_lr[i0:i1, i0:i1] = curr_mat
    
    # Long range connections
    lr_conns = make_random_asymm_matrix_given_density(n_vertices, inter_block_edges, entry_type='uniform_positive')
    W_with_lr = W_without_lr + lr_conns

    # Might be able to move this to the calling function and use the extremal ew function (since this is making
    # a choice of the longest timescales for the network it returns)
    tmp_ew, tmp_ev = nla.eig(W_with_lr)
    W = W_with_lr/(1.1*np.max(np.real(tmp_ew)))
    return W 
     