'''
August 7th 2020
Functions to simulate noise-driven linear systems and to predict their 
covariance. Updated version of file in 2020_02_general_fns/
Renamed because I'm also importing functions from 2020_02_general_fns and 
if we add that path in there might be confusion if files have the same name.
'''

import numpy as np
import numpy.linalg as nla

def sim_linear_net(A, params):
    '''Simulate linear network with specified input and white noise. 
    Note that A includes the leak term. '''

    n_nodes = A.shape[1]
    x = np.zeros((params['n_steps'], n_nodes))
    if 'ics' in params:
        x[0] = params['ics']
    # Set up the noise
    sim_sigma = np.sqrt(params['dt']) * params['noise_std']
    dt_b = params['dt'] * params['input']

    for i in range(params['n_steps']-1):
        x[i+1] = x[i] + params['dt'] * (A @ x[i]) + dt_b[i] + sim_sigma * np.random.randn(n_nodes)
    return x

def predict_ou_cov(A, B):
    '''Predict the covariance matrix of the OU process of the form
    dx/dt = Ax + Bxi(t),
    where xi is white noise. Should test again.'''

    N = len(A)
    ew, ev = nla.eig(A)
    ev_inv = nla.inv(ev)
    
    bbt = np.dot(B, B.conjugate().T)
    
    Q_tilde = np.dot(ev_inv, np.dot(bbt, ev_inv.conjugate().T))

    # M = np.zeros_like(A)
    M = np.zeros((A.shape[0], A.shape[1]), dtype=complex) #this is tilde_C

    for i in range(N):
        for j in range(N):
            M[i,j] = -Q_tilde[i,j]/(ew[i] + np.conj(ew[j]))

    pred_cov = np.dot(ev, np.dot(M, ev.conjugate().T))

    if np.max(np.abs(np.imag(pred_cov)))>1e-8:
        print('May have noticeable imaginary part. Check!')
    return np.real(pred_cov)


def pred_cov_from_white_noise(A, matrix_type='general'):
    '''Predict the covariance matrix of the noise-driven linear 
    dynamical system (i.e., OU process) given by
    dx/dt = A*x(t) + xi(t),
    where xi(t) is white noise. This version is specialized to white noise, 
    so that we can be a bit faster.  
    
    The matrix_type flag is 'general' by default but I'm guessing this will be 
    slow since it needs to diagonalize the matrix. Can be set to symmetric or 
    normal and should be faster since it can just compute the inverse in that case.
    Also note that this doesn't include an overall sigma^2 scaling factor on the
    variance.

    I've briefly tested the symmetric and general cases though not the normal case
    (but normal agrees with symmetric for a symmetric matrix). Could stand to do 
    more testing throughout because the empirical covariance takes a while to 
    converge. Could maybe do exact OU simulation and use a larger time step.
    '''

    if matrix_type == 'symmetric':
        C = -nla.inv(A)/2.
    elif matrix_type == 'normal':
        C = -nla.inv(A + A.T)
    elif matrix_type == 'general':
        C = predict_ou_cov(A, np.eye(len(A)))
    else:
        print('Unknown matrix type')
        C = np.nan
    return C


    