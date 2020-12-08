'''October 21st 2020
Combined set of symmetric and non-symmetric sparsification functions.
'''

import sys, os
import numpy as np
import numpy.linalg as nla

import linear_net_fns as lnf
import lin_alg_fns as laf

def compare_spectrum(S_new, ew_old, ev_old, ret_ew=False):
    '''Takes in a sparsification and the spectrum of the original
    matrix and returns various measures of performance eps_ew, eps_ev 
    and S_ev_angle. 
    Use only with symmetric matrices (otherwise check the measures).'''

    spect_comp = {}
    ew_new = laf.sort_eig(S_new)
    
    # Measure how much the eigenvalues have moved
    spect_comp['eps_ew'] = np.real(ew_new / ew_old - 1)

    # Compute terms of the form x^T*S_new*x, which are what is guaranteed to be preserved.
    # Rather than looking at random x look at the eigenvectors of the old matrix.
    rq_list_new = np.array([ev_old[:,i] @ (S_new @ ev_old[:,i]) for i in range(len(S_new))])
    # So this will reflect something about how much the eigenvectors have moved
    spect_comp['eps_ev'] = rq_list_new / ew_old - 1
    # Also look at the normalized dot product between v and S_new*v, where v is an ev.
    # This should be 1 if v is close to an eigenvector of the new matrix
    S_ev_norm_new = nla.norm(S_new @ ev_old, axis=0)
    spect_comp['S_ev_angle']= rq_list_new / S_ev_norm_new
    if ret_ew:
        spect_comp['ew'] = ew_new
    return spect_comp

def normalize_probs(p, params):
    '''Normalize the probabilities in p, set a minimum threshold below
    which things are 0, make sure nothing is above 1, etc.
    Note that normalization is done first before anything is clipped, so that
    after clipping normalization may not hold exactly.
    '''

    samp_probs = p.copy()
    if 'norm_type' in params:
        if params['norm_type'] == 'sum':
            samp_probs = params['norm_val'] * p/np.sum(p)
    if 'zero_thresh' in params:
        samp_probs[samp_probs<params['zero_thresh']] = 0
    if 'max_val' in params:
        # Typically use with max_val of 1.
        samp_probs[samp_probs>params['max_val']] = params['max_val']
    return samp_probs

def get_diff_matrix(C, sign_matrix):
    '''Returns the matrix of differences of C in the form required by
    noise-prune.
    diff_matrix[i,j] = C[i,i] + C[j,j] - 2*sign_matrix[i,j]*C[i,j].
    Uses numpy broadcasting instead of loops, so should be much faster.
    '''

    diags = np.diag(C) 
    i_matrix = np.expand_dims(diags, axis=1)
    diag_contribution = i_matrix + i_matrix.T
    diff_matrix = diag_contribution - 2*sign_matrix*C
    return diff_matrix

def get_sparsify_probs(A, normalization={}, return_diffs=False, 
    matrix_type='general'):
    
    '''Return noise-prune sampling probabilities. 

    This works for symmetric matrices too. Just need to zero out probs on one
    side of the diagonal before passing to the sparsification fn.
    '''
    
    assert np.all(np.diag(A)<=0), 'Diagonal of A is positive!'

    # First, get the covariance matrix
    C = lnf.pred_cov_from_white_noise(A, matrix_type=matrix_type)

    # Next, get the difference matrix (C_ii + C_jj - 2*sign(W_ij)*C_ij). Note that
    # since we're passing signs from A we'll get some stuff along the diagonal. Also
    # note that the diff function now has a minus sign rather than plus sign.
    diff_matrix = get_diff_matrix(C, np.sign(A))

    # Put these together to get the unnormalized sampling probabilities which are
    # |w_ij|(C_ii + C_jj - 2*sign(W_ij)*C_ij). Note that because we have A we'll
    # get some diagonal stuff that needs to be removed because it'll mess up
    # normalization.

    samp_probs_unn = np.abs(A) * diff_matrix
    np.fill_diagonal(samp_probs_unn, 0)
    
    # Normalize. Could also let the calling function normalize
    samp_probs = normalize_probs(samp_probs_unn, normalization)

    # Note that samp_probs are always a matrix now
    if return_diffs:
        return samp_probs, diff_matrix
    else:
        return samp_probs

def get_sparsify_probs_control(A, strategy='weights', normalization={}):
    '''Various control probabilities to compare with noise prune. '''
    
    if strategy == 'weights':
        samp_probs_unn = np.abs(A)
    elif strategy == 'uniform':
        samp_probs_unn = (np.abs(A)>0).astype(bool)
    else:
        print('Unknown means to generate weights')
        return np.nan

    # Since A might have a diagonal (because of leak), samp_probs_unn might 
    # also have diagonal components. Remove these. 
    np.fill_diagonal(samp_probs_unn, 0)

    samp_probs = normalize_probs(samp_probs_unn, normalization)
    
    return samp_probs

def set_diagonal(sparse_matrix, orig_matrix, diagonal_type='zero'):
    '''Set the diagonal entries after pruning.
    Note that this probably overwrites the original sparse matrix'''

    if diagonal_type == 'original':
        # Just put back the original diagonal
        sparse_matrix_with_diag = sparse_matrix
        np.fill_diagonal(sparse_matrix_with_diag, np.diag(orig_matrix))
    elif diagonal_type == 'zero':
        sparse_matrix_with_diag = sparse_matrix
    elif diagonal_type == 'row_sum':
        assert np.all(np.diag(orig_matrix)<=0), 'Diagonal of A is positive!'
        new_diag = np.diag(orig_matrix) + (laf.get_off_diag_row_sums(orig_matrix, abs_val=True) - 
            laf.get_off_diag_row_sums(sparse_matrix, abs_val=True))
        sparse_matrix_with_diag = sparse_matrix
        np.fill_diagonal(sparse_matrix_with_diag, new_diag)
    else:
        print('Unknown diagonal type')
    return sparse_matrix_with_diag

def sparsify_given_probs(A, samp_probs, symmetric=False, diagonal_type='zero', rescale = True):
    '''Sparsify the matrix A using the sampling probabilities given in
    samp_probs. Removed the with replacement option, so that each edge is
    simply sampled independently with its own probability.

    Note that if symmetric=True then the sampling probs should be 0 in one triangle
    of the matrix, since it just symmetrizes the pruned matrix.
    '''

    if np.any(np.diag(samp_probs)):
        print('Warning there are non-zero sampling probs along diagonal')
    if symmetric==True:
        if (nla.norm(np.tril(samp_probs))>0) and (nla.norm(np.triu(samp_probs))>0):
            print('Probabilities should be zero on one side of diagonal for symmetric')
    
    N = len(A)
    edges_to_keep = (np.random.rand(N, N) < samp_probs)
    sparse_matrix = np.zeros_like(A)
    if rescale:
        print('Maximum rescale ', np.max(1./samp_probs[edges_to_keep]))
    else:
        sparse_matrix[edges_to_keep] = A[edges_to_keep]

    if symmetric==True:
        sparse_matrix_symm = sparse_matrix + sparse_matrix.T
        assert np.sum(np.abs(np.diag(sparse_matrix_symm)))<1e-8, 'nonzero diagonal'
        sparse_matrix_with_diag = set_diagonal(sparse_matrix_symm, A, diagonal_type=diagonal_type)
    else:
        sparse_matrix_with_diag = set_diagonal(sparse_matrix, A, diagonal_type=diagonal_type)

    return sparse_matrix_with_diag




