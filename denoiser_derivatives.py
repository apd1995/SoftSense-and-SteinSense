#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:03:49 2023


This script computes numerical derivatives of denoiser using autograd.
Note that we need the transpose of the jacobian matrix for AMP.


@author: apratimdey
"""

from autograd import jacobian
from block_soft_thresholding import block_soft_thresholding_nonsingular, block_soft_thresholding_singular


def block_soft_thresholding_jacobian_nonsingular(y, tau, Sigma_inv):
    
    jacobian_func = jacobian(block_soft_thresholding_nonsingular)
    return jacobian_func(y, tau = tau, Sigma_inv = Sigma_inv).T


def block_soft_thresholding_jacobian_singular(y, tau, Sigma_eigvals, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    
    jacobian_func = jacobian(block_soft_thresholding_singular)
    return jacobian_func(y, tau = tau, Sigma_eigvals = Sigma_eigvals, Sigma_eigvecs = Sigma_eigvecs, nonzero_indices = nonzero_indices, Sigma_nonzero_eigvals_inv = Sigma_nonzero_eigvals_inv).T
