#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:07:53 2023

@author: apratimdey
"""

import autograd.numpy as np


def block_soft_thresholding_nonsingular_vec(y, tau, Sigma_inv):
    quad_whitening = np.dot(np.matmul(Sigma_inv, y), y)
    if quad_whitening > 0:
        return y * max(0, 1 - (tau/np.sqrt(quad_whitening)))
    else:
        return y
    

def block_soft_thresholding_nonsingular(X: np.ndarray, tau: float, Sigma_inv: np.ndarray) -> np.ndarray:
    """
    Applies block soft thresholding rowwise to denoise Y.

    Parameters
    ----------
    Y : np.ndarray
        Noisy signal.
    tau : float
        Threshold for block soft thresholding.
    Sigma_inv : np.ndarray
        Noise precision matrix.

    Returns
    -------
    None.
    """
    quad_whitening = np.sum(X * np.matmul(X, Sigma_inv), axis=1)
    block_soft_thresholding_coeff = np.where(quad_whitening != 0, 1 - tau/np.sqrt(quad_whitening), 1)
    block_soft_thresholding_coeff = np.where(block_soft_thresholding_coeff > 0, block_soft_thresholding_coeff, 0)
    return X * block_soft_thresholding_coeff[:, np.newaxis]


def block_soft_thresholding_diagonal_vec(y, tau, diag_inv):
    quad_whitening = sum(diag_inv * y**2)
    if quad_whitening > 0:
        return y * max(0, 1 - (tau/np.sqrt(quad_whitening)))
    else:
        return y
    

def block_soft_thresholding_diagonal(X, tau, diag_inv):
    quad_whitening = np.sum(X**2 * diag_inv, axis=1)
    block_soft_thresholding_coeff = np.where(quad_whitening != 0, 1 - tau/np.sqrt(quad_whitening), 1)
    block_soft_thresholding_coeff = np.where(block_soft_thresholding_coeff > 0, block_soft_thresholding_coeff, 0)
    return X * block_soft_thresholding_coeff[:, np.newaxis]


def block_soft_thresholding_singular_vec(y, tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    # changing coordinates to get uncorrelated components
    y_indep = np.matmul(Sigma_eigvecs.T, y)
    
    y_indep_nonzero =  y_indep[nonzero_indices]
    
    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = block_soft_thresholding_diagonal_vec(y_indep_nonzero, tau, Sigma_nonzero_eigvals_inv)
    
    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    y_indep_zero = y_indep[zero_indices]
    signal_newbasis_zero = y_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = np.concatenate((signal_newbasis_zero, signal_newbasis_indep))
    # signal_newbasis = np.zeros(len(y), dtype = float)
    # signal_newbasis[nonzero_indices] = signal_newbasis_indep
    # signal_newbasis[zero_indices] = signal_newbasis_zero
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = np.matmul(Sigma_eigvecs, signal_newbasis)
    
    return signal_originalbasis


def block_soft_thresholding_singular(X, tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    # changing coordinates to get uncorrelated components
    X_indep = np.matmul(X, Sigma_eigvecs)
    
    X_indep_nonzero =  X_indep[:, nonzero_indices]
    
    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = block_soft_thresholding_diagonal(X_indep_nonzero, tau, Sigma_nonzero_eigvals_inv)
    
    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    X_indep_zero = X_indep[:, zero_indices]
    signal_newbasis_zero = X_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = np.zeros(X.shape, dtype = float)
    signal_newbasis[:, nonzero_indices] = signal_newbasis_indep
    signal_newbasis[:, zero_indices] = signal_newbasis_zero
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = np.matmul(signal_newbasis, Sigma_eigvecs.T)
    
    return signal_originalbasis
