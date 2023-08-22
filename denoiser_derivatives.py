#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:48:32 2023

This script computes numerical derivatives of denoiser using autograd.
Note that we need the transpose of the jacobian matrix for AMP.


@author: apratimdey
"""

from autograd import jacobian
from james_stein_denoiser import james_stein_denoiser


def james_stein_jacobian(y, Sigma):
    
    jacobian_func = jacobian(james_stein_denoiser)
    return jacobian_func(y, Sigma).T

# =============================================================================
# x = np.array([5,3], dtype=float)
# 
# def cost(x, pagla):
#     return pagla*np.array([[1,2], [3,4]]) @ x
# 
# gradient_cost = grad(cost)
# jacobian_cost = jacobian(cost)
# 
# #gradient_cost(x)
# jacobian_cost(x, pagla = 2)
# 
# =============================================================================
