#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:07:53 2023

@author: apratimdey
"""

import autograd.numpy as np


def block_soft_thresholding_nonsingular(y, tau, Sigma_inv):
    quad_whitening = np.dot(np.matmul(Sigma_inv, y), y)
    if quad_whitening > 0:
        return y * max(0, 1 - (tau/np.sqrt(quad_whitening)))
    else:
        return y
    

def block_soft_thresholding_diagonal(y, tau, diag_inv):
    quad_whitening = sum(diag_inv * y**2)
    if quad_whitening > 0:
        return y * max(0, 1 - (tau/np.sqrt(quad_whitening)))
    else:
        return y


def block_soft_thresholding_singular(y, tau, Sigma_eigvals, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    # changing coordinates to get uncorrelated components
    y_indep = np.matmul(Sigma_eigvecs.T, y)
    
    y_indep_nonzero =  y_indep[nonzero_indices]
    
    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = block_soft_thresholding_diagonal(y_indep_nonzero, tau, Sigma_nonzero_eigvals_inv)
    
    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    y_indep_zero = y_indep[zero_indices]
    signal_newbasis_zero = y_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = np.concatenate((signal_newbasis_zero, signal_newbasis_indep))
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = np.matmul(Sigma_eigvecs, signal_newbasis)
    
    return signal_originalbasis
