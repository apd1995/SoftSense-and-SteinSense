#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:03:49 2023


This script computes numerical derivatives of denoiser using autograd.
Note that we need the transpose of the jacobian matrix for AMP.


@author: apratimdey
"""

from autograd import jacobian
#from autograd import make_jvp
import autograd.numpy as np
import block_soft_thresholding as bs


def block_soft_thresholding_onsager_nonsingular(X, Z, tau, Sigma_inv):
    X = X.astype(np.float64)
    jacobian_mat_list = [jacobian(bs.block_soft_thresholding_nonsingular_vec)(X[i,:], tau, Sigma_inv) for i in range(X.shape[0])]
    onsager_list = [np.matmul(jacobian_mat, Z.T) for jacobian_mat in jacobian_mat_list]
    return sum(onsager_list).T/Z.shape[0]
        
    
def block_soft_thresholding_onsager_singular(X, Z, tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    X = X.astype(np.float64)
    jacobian_mat_list = [jacobian(bs.block_soft_thresholding_singular_vec)(X[i,:], tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv) for i in range(X.shape[0])]
    onsager_list = [np.matmul(jacobian_mat, Z.T) for jacobian_mat in jacobian_mat_list]
    return sum(onsager_list).T/Z.shape[0]

    
    
    
    
    