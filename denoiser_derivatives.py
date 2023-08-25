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

# =============================================================================
# # x = np.array([5,3], dtype=float)

# # def cost(x, pagla, rishi):
# #     return pagla*np.array([[1,2], [3,4]]) @ x + rishi*x

# # #gradient_cost = grad(cost)
# # jacobian_cost = jacobian(cost)

# # #gradient_cost(x)
# # jacobian_cost(x, pagla = 2, rishi = 4)

# =============================================================================