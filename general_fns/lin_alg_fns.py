'''August 17th, 2019
A few convenient general linear algebra functions.'''

import numpy as np
import numpy.linalg as la
from collections import defaultdict
import scipy.linalg as sla
import undirected_graph_fns as gf

def sort_eig(A, ret_ev=False, hermitian=False):
    '''Returns sorted eigenvalues and eigenvectors of A. 
    Modifying slightly to take a hermitian option and will call the appropriate
    eval functions if so.'''

    if ret_ev:
        if hermitian:
            ew, ev = la.eigh(A)
        else:
            ew, ev = la.eig(A)
        idx = np.argsort(ew)
        return ew[idx], ev[:,idx]
    else:
        if hermitian:
            return np.sort(la.eigvalsh(A))
        else:
            return np.sort(la.eigvals(A))

def get_norm_dp(v1,v2):
    '''Get dot product between normalized vectors v1 and v2.'''

    n1 = v1/la.norm(v1)
    n2 = v2/la.norm(v2)
    return np.dot(n1,n2)

def get_pinv(inp_mat):
    '''Compute the pseudoinverse of inp_mat.'''

    u, s, vh = la.svd(inp_mat)

    s_inv = np.array([0. if np.abs(x)<1e-8 else 1./x for x in s])
    s_inv_mat = np.diag(s_inv)
    out_mat = np.dot(np.dot(u,s_inv_mat),vh)
    return out_mat

def get_matrix_sqrt(inp_mat):
    '''Get the square root of a positive semi-definite matrix.
    Note that we don't check for positive-semi-definiteness but
    result won't make sense if not.'''

    u, s, vh = la.svd(inp_mat)
    sqrt_s_mat = np.diag(np.sqrt(s))
    sqrt_mat = np.dot(np.dot(u, sqrt_s_mat), vh)
    return sqrt_mat

def compare_quadratic_forms(LG, LH, n_tests, normed=False):
    '''Compare the two quadratic forms LG and LH by looking at 
    their values on random vectors.'''
    
    n_nodes = len(LG)
    comp_vals = np.zeros((n_tests, 2))

    for i in range(n_tests):
        x = np.random.randn(n_nodes)
        if normed:
            x = x/la.norm(x)

        comp_vals[i,0] = np.dot(x.T, np.dot(LG,x))
        comp_vals[i,1] = np.dot(x.T, np.dot(LH,x))

    return comp_vals

# Some helper functions to shift general matrices to make them into the right form
# for this set of sparsification functions. Note that the sparsification functions
# themselves assume that the matrices are diagonally dominant.

def get_off_diag_row_sums(W, abs_val=True):
    '''Get the sum of each row of W ignoring diagonal.'''

    if abs_val:
        row_sums = np.sum(np.abs(W - np.diag(np.diag(W))), axis=1)
    else:
        row_sums = np.sum(W - np.diag(np.diag(W)), axis=1)
    return row_sums

def shift_to_make_diagonally_dominant(W, shift_type='constant', diag_type='pos'): 
    '''Convert W into a diagonally-dominant matrix, either by adding a 
    constant shift to every diagonal, or adding a diagonal matrix. 
    Note that this function partially takes into account an existing diagonal, 
    but the later functions may not treat the pre-existing diagonal correctly so 
    be careful when using it with non-zero diagonals.'''

    diag_els = np.diag(W) 
    
    row_sums = get_off_diag_row_sums(W)
    if diag_type=='pos':
        if shift_type == 'constant':
            shift_amount = np.max(row_sums - diag_els)
            shift_W = W + shift_amount * np.eye(len(W))
        elif shift_type == 'individual':
            shift_W = W + np.diag(row_sums - diag_els)
        else:
            print('Unknown shift_type')
    if diag_type=='neg':
        if shift_type == 'constant':
            shift_amount = np.max(row_sums + diag_els)
            shift_W = W - shift_amount * np.eye(len(W))
        elif shift_type == 'individual':
            shift_W = W - np.diag(row_sums + diag_els)
        else:
            print('Unknown shift_type')
    return shift_W

def shift_matrix_to_set_extremal_ew(A, desired_ew, ew_type='largest'):
    if ew_type == 'largest':
        rel_ew = np.max(np.real(la.eigvals(A)))
    elif ew_type == 'smallest':
        rel_ew = np.min(np.real(la.eigvals(A)))
    else:
        print('Unknown ew type')
        return np.nan
    shift_amount = -rel_ew + desired_ew
    return A + shift_amount * np.eye(len(A))

    	