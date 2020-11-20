'''February 18th 2020
Add reworked some more, to have get_edge_list_from_adj_matrix return a tuple of arrays rather 
than a 2-D array. The tuple of arrays is the natural Numpy addressing syntax (I think), and I 
appreciate why now so switch to using it.
August 17th 2019
Reworked these a bit so test them again.
Feburary 18th 2019
Functions to set up and manipulate graphs.
'''

import numpy as np
import numpy.linalg as nla
# import sys, time, os, datetime

def get_edge_list_from_adj_matrix(W, is_symmetric=True, get_wts=False):
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

def assign_entries(N, nz_indices, nz_values):
    '''Create an N x N matrix with the indices in nz_indices corresponding to
    nz_values, and 0 otherwise. Sort of an inverse to the above function, though a 
    bit more general and also will be used to assign values besides weights to edges (e.g.,
    sampling probabilities.)
    '''

    A = np.zeros((N,N))
    A[nz_indices] = nz_values
    return A

def symmetrize_matrix(W):
    '''Meant to be used on an triangular matrix (though of course it'll
    symmetrize anything). '''

    return W + W.T - np.diag(W.diagonal())    

# Retest the indexing on these. I think G[(sel_row_idx, sel_col_idx)] is the same
# as G[sel_row_idx, sel_col_idx] but check.

def make_random_matrix_given_density(n_vertices, edges, entry_type='uniform_positive', 
    params={}, symmetrize=True):
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

#can add more parameters here if needed later
def make_clustered_graph(n_blocks, n_per_block): 
    n_vertices = n_blocks * n_per_block
    W_without_lr = np.zeros((n_vertices, n_vertices))

    graph_params = {'mean' : 1., 'sigma' : 1.}

    for i_block in range(n_blocks):
        curr_mat = make_random_matrix_given_density(n_per_block, edges=0.8, entry_type='normal',
                                                  params=graph_params)
        i0 = i_block * n_per_block
        i1 = i0 + n_per_block
        W_without_lr[i0:i1, i0:i1] = curr_mat

    # Now make some random connections
    lr_conns = make_random_matrix_given_density(n_vertices, 0.01, entry_type='uniform_positive')
    W_with_lr = W_without_lr + lr_conns
    tmp_ew, tmp_ev = nla.eig(W_with_lr)
    W = W_with_lr/(1.1*np.max(np.real(tmp_ew)))
    return W

#making a more generic clustered graph function and using above for reference. n_per_block is a vector whose entries are the number of nodes in a given block. ex n_p_b = [50,100] will give a 150x150 matrix with blocks of size 50 and 100 respectively.
def make_imb_clustered_graph(n_per_block, inner_block_edges, inter_block_edges):
    n_vertices = 0
    for each_block in n_per_block:
        n_vertices = n_vertices + each_block
    W_without_lr = np.zeros((n_vertices, n_vertices))
    graph_params = {'mean' : 1., 'sigma' : 1.}
    i1 = 0
    for each_block in n_per_block:
        curr_mat = make_random_matrix_given_density(each_block, edges=inner_block_edges, entry_type='normal',
                                                  params=graph_params)
        i0 = i1
        i1 = i1 + each_block
        W_without_lr[i0:i1, i0:i1] = curr_mat
    # Now make some random connections
    lr_conns = make_random_matrix_given_density(n_vertices, inter_block_edges, entry_type='uniform_positive')
    W_with_lr = W_without_lr + lr_conns
    tmp_ew, tmp_ev = nla.eig(W_with_lr)
    W = W_with_lr/(1.1*np.max(np.real(tmp_ew)))
    return W  
    

# Not sure I'm using these here, and removed make_random_triu_matrix_given_density, so 
# commenting out for now.    
# def make_random_asymmetric_matrix_given_density(n_vertices, edges, entry_type='uniform_positive', params={}):
#     U1 = make_random_triu_matrix_given_density(n_vertices, edges, entry_type='uniform_positive', params={})
#     U2 = make_random_triu_matrix_given_density(n_vertices, edges, entry_type='uniform_positive', params={})
#     return U1 + U2.T
    
# def make_latent_variable_matrix(n_vertices, f):
#     '''Make a matrix where the i,j-th entry is given by
#     f(i,j).'''
#     G = np.zeros((n_vertices, n_vertices))
#     for i in range(n_vertices):
#         for j in range(n_vertices):
#             G[i,j] = f(i,j)
#     return G

def get_laplacian(W):
    '''Returns Laplacian of weighted graph, given by L = D-W,
    where W is the weighted adjacency matrix, and D is the diagonal matrix
    of weighted edges'''

    # Axis doesn't matter because symmetric
    wts = np.sum(W, axis=0)
    D = np.diag(wts)
    return D - W

def get_signed_laplacian(W):
    '''Attempt to generalize Laplacian to negative weights. Construct the
    diagonal of a Laplacian-like operator by summing absolute weights.'''
   
    wts = np.sum(np.abs(W), axis=0)
    D = np.diag(wts)
    return D - W


# Test before using because switched convention on get_edge_list_from_adj_matrix
# and added a transpose to make up.
# def get_signed_incidence_matrix(W):
#     '''Might need to be careful about head and tail if/when I switch to
#     directed graphs. Using the convention that W[i,j] describes the edge 
#     from node j to node i.  For symmetric matrix doesn't matter (and don't use this 
#     for non_symmetric without thinking about it.'''
    
#     n_vertices = len(W)
#     edge_list = np.transpose(get_edge_list_from_adj_matrix(W, is_symmetric=True))
#     n_edges = len(edge_list)

#     B = np.zeros((n_edges, n_vertices))
#     for i, edge in enumerate(edge_list):
#         B[i, edge[0]] = 1
#         B[i, edge[1]] = -1
#     return B
