#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:25:35 2023

This script contains functions to compute a generic AMP iterate
for matrix recovery given undersampled measurements.

The param 'noise_cov_.' always refers to noise covariance for a noisy signal.
For example, noise_cov_current refers to noise covariance of the noisy signal (before applying denoiser)
at current iteration.

@author: apratimdey
"""

import numpy as np
import block_soft_thresholding as bs
import denoiser_derivatives as dd


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)


def update_signal_denoised_nonsingular(signal_noisy_current: float,
                           tau: float,
                           noise_cov_current_inv: float):
    
    return bs.block_soft_thresholding_nonsingular(signal_noisy_current, tau, noise_cov_current_inv)


def update_signal_denoised_singular(signal_noisy_current: float,
                           tau: float,
                           noise_cov_current_eigvecs: float,
                           noise_cov_current_nonzero_indices,
                           noise_cov_current_nonzero_eigvals_inv: float):
    
    return bs.block_soft_thresholding_singular(signal_noisy_current, tau, noise_cov_current_eigvecs, noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv)


# def onsager_matrix_nonsingular(signal_noisy_current: float,
#                    tau: float,
#                     noise_cov_current_inv: float,
#                     num_measurements: float):
    
    
    
#     jacobian_array = np.apply_along_axis(block_soft_thresholding_jacobian_nonsingular, 1, signal_noisy_current, tau, noise_cov_current_inv)
#     return np.sum(jacobian_array, axis = 0)/num_measurements


# def onsager_matrix_singular(signal_noisy_current: float,
#                             tau: float,
#                             noise_cov_current_eigvals: float,
#                             noise_cov_current_eigvecs: float,
#                             noise_cov_current_nonzero_indices,
#                             noise_cov_current_nonzero_eigvals_inv: float,
#                             num_measurements: float):
    
#     jacobian_array = np.apply_along_axis(block_soft_thresholding_jacobian_singular, 1, signal_noisy_current, tau, noise_cov_current_eigvals, noise_cov_current_eigvecs, noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv)
#     return np.sum(jacobian_array, axis = 0)/num_measurements


# def onsager_term_nonsingular(Residual_prev: float,
#                  signal_noisy_current: float,
#                  tau: float,
#                  noise_cov_current_inv: float):

#     onsager_matrix_ = onsager_matrix_nonsingular(signal_noisy_current, tau, noise_cov_current_inv, Residual_prev.shape[0])
#     return np.matmul(Residual_prev, onsager_matrix_)


# def onsager_term_singular(Residual_prev: float,
#                           signal_noisy_current: float,
#                             tau: float,
#                             noise_cov_current_eigvals: float,
#                             noise_cov_current_eigvecs: float,
#                             noise_cov_current_nonzero_indices,
#                             noise_cov_current_nonzero_eigvals_inv: float):

#     onsager_matrix_ = onsager_matrix_singular(signal_noisy_current, tau, noise_cov_current_eigvals, noise_cov_current_eigvecs, noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv, Residual_prev.shape[0])
#     return np.matmul(Residual_prev, onsager_matrix_)


def update_residual_nonsingular(A: float,
                    Y: float,
                    signal_noisy_current: float,
                    signal_denoised_current: float,
                    Residual_prev: float,
                    tau: float,
                    noise_cov_current_inv: float):
    
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = dd.block_soft_thresholding_onsager_nonsingular(signal_noisy_current, Residual_prev, tau, noise_cov_current_inv)
    return naive_residual + onsager_term_


def update_residual_singular(A: float,
                    Y: float,
                    signal_noisy_current: float,
                    signal_denoised_current: float,
                    Residual_prev: float,
                    tau: float,
                    noise_cov_current_eigvecs: float,
                    noise_cov_current_nonzero_indices,
                    noise_cov_current_nonzero_eigvals_inv: float):
    
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = dd.block_soft_thresholding_onsager_singular(signal_noisy_current, Residual_prev, tau, noise_cov_current_eigvecs, noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv)
    return naive_residual + onsager_term_


def amp_iteration_nonsingular(A: float,
                  Y: float,
                  signal_denoised_prev: float,
                  Residual_prev: float,
                  tau: float,
                  noise_cov_current_inv: float):
    
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    #noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised_nonsingular(signal_noisy_current, tau, noise_cov_current_inv)
    Residual_current = update_residual_nonsingular(A, Y, signal_noisy_current, signal_denoised_current, Residual_prev, tau, noise_cov_current_inv)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


def amp_iteration_singular(A: float,
                  Y: float,
                  signal_denoised_prev: float,
                  Residual_prev: float,
                  tau: float,
                noise_cov_current_eigvecs: float,
                noise_cov_current_nonzero_indices,
                noise_cov_current_nonzero_eigvals_inv: float):
    
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    #noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised_singular(signal_noisy_current, tau, noise_cov_current_eigvecs, noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv)
    Residual_current = update_residual_singular(A, Y, signal_noisy_current, signal_denoised_current, Residual_prev, tau, noise_cov_current_eigvecs, noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}






