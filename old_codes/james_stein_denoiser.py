#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 21:29:23 2023

@author: apratimdey
"""


import numpy as np


def james_stein_denoiser_nonsingular(y, Sigma):
    d = len(y)
    quad_whitening = y.T @ np.linalg.inv(Sigma) @ y
    return y * max(0, 1 - ((d-2)/quad_whitening))


def james_stein_denoiser_singular(y, Sigma):
    D, U = np.linalg.eigh(Sigma)
    
    # changing coordinates to get uncorrelated components
    y_indep = U.T @ y
    
    # eps**(1/(d-k)): for simplicity let's ignore this
    
    nonzero_indices = (D > 0)
    D_nonzero = np.diag(D[nonzero_indices])
    y_indep_nonzero =  y_indep[nonzero_indices]
    
    # apply James Stein on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = james_stein_denoiser_nonsingular(y_indep_nonzero, D_nonzero)
    
    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    y_indep_zero = y_indep[zero_indices]
    signal_newbasis_zero = y_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = np.concatenate((signal_newbasis_zero, signal_newbasis_indep))
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = U @ signal_newbasis
    
    return signal_originalbasis


def james_stein_denoiser(y, Sigma):
    if np.linalg.det(Sigma) == 0:
    # if np.isclose(np.linalg.det(Sigma), 0, atol = 1e-10):
        return james_stein_denoiser_singular(y, Sigma)
    else:
        return james_stein_denoiser_nonsingular(y, Sigma)
