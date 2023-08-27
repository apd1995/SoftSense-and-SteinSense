#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:03:49 2023


This script computes numerical derivatives of denoiser using autograd.
Note that we need the transpose of the jacobian matrix for AMP.


@author: apratimdey
"""


from autograd import jacobian
from block_soft_thresholding import block_soft_thresholding


def block_soft_thresholding_jacobian(y, tau, Sigma):
    
    jacobian_func = jacobian(block_soft_thresholding)
    return jacobian_func(y, tau = tau, Sigma = Sigma).T
