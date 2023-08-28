#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:07:53 2023

@author: apratimdey
"""

import autograd.numpy as np


def block_soft_thresholding_nonsingular(y, tau, Sigma):
    quad_whitening = y.T @ np.linalg.inv(Sigma) @ y
    if quad_whitening > 0:
        return y * max(0, 1 - (tau/np.sqrt(quad_whitening)))
    else:
        return y


def block_soft_thresholding_singular(y, tau, Sigma):
    D, U = np.linalg.eigh(Sigma)
    
    # changing coordinates to get uncorrelated components
    y_indep = U.T @ y
    
    nonzero_indices = (D > 0)
    D_nonzero = np.diag(D[nonzero_indices])
    y_indep_nonzero =  y_indep[nonzero_indices]
    
    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = block_soft_thresholding_nonsingular(y_indep_nonzero, tau, D_nonzero)
    
    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    y_indep_zero = y_indep[zero_indices]
    signal_newbasis_zero = y_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = np.concatenate((signal_newbasis_zero, signal_newbasis_indep))
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = U @ signal_newbasis
    
    return signal_originalbasis


def block_soft_thresholding(y, tau, Sigma):
    if np.linalg.det(Sigma) == 0:
    # if np.isclose(np.linalg.det(Sigma), 0, atol = 1e-10):
        return block_soft_thresholding_singular(y, tau, Sigma)
    else:
        return block_soft_thresholding_nonsingular(y, tau, Sigma)
