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
from block_soft_thresholding import block_soft_thresholding
from denoiser_derivatives import block_soft_thresholding_jacobian


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    
    return signal_denoised_prev + A.T@Residual_prev


def update_signal_denoised(signal_noisy_current: float,
                           tau: float,
                           noise_cov_current: float):
    
    return np.apply_along_axis(block_soft_thresholding, 1, signal_noisy_current, tau, noise_cov_current)


def onsager_matrix(signal_noisy_current: float,
                   tau: float,
                    noise_cov_current: float,
                    num_measurements: float):
    
    jacobian_array = np.apply_along_axis(block_soft_thresholding_jacobian, 1, signal_noisy_current, tau, noise_cov_current)
    return np.sum(jacobian_array, axis = 0)/num_measurements


def onsager_term(Residual_prev: float,
                 signal_noisy_current: float,
                 tau: float,
                 noise_cov_current: float):

    onsager_matrix_ = onsager_matrix(signal_noisy_current, tau, noise_cov_current, Residual_prev.shape[0])
    return Residual_prev @ onsager_matrix_


def update_residual(A: float,
                    Y: float,
                    signal_noisy_current: float,
                    signal_denoised_current: float,
                    Residual_prev: float,
                    tau: float,
                    noise_cov_current: float):
    
    naive_residual = Y - A @ signal_denoised_current
    onsager_term_ = onsager_term(Residual_prev, signal_noisy_current, tau, noise_cov_current)
    return naive_residual + onsager_term_


def amp_iteration(A: float,
                  Y: float,
                  signal_denoised_prev: float,
                  Residual_prev: float,
                  tau: float,
                  noise_cov_current: float):
    
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    #noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised(signal_noisy_current, tau, noise_cov_current)
    Residual_current = update_residual(A, Y, signal_noisy_current, signal_denoised_current, Residual_prev, tau, noise_cov_current)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}





















