#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:13:18 2023

This script computes the minimax threshold for block soft thresholding. Assumes
noise level is 1. notation follows Don-Joh-Mon (2013).

Tested with R equivalent.

@author: apratimdey
"""

from scipy import integrate
from scipy.stats import chi2
from scipy.optimize import brentq
import numpy as np


def h_deno_integrand(x, tau, signal_ncol):
    return  (np.sqrt(x) - tau) * chi2.pdf(x, df = signal_ncol)


def h_deno_integral(tau, signal_ncol):
    return max(1e-10, integrate.quad(h_deno_integrand, tau**2, np.inf, args=(tau, signal_ncol))[0])


def h(tau, signal_ncol):
    numerator = tau
    denominator = h_deno_integral(tau, signal_ncol)
    return numerator/denominator


def h_centered(tau, sparsity, signal_ncol):
    return h(tau, signal_ncol) + 1 - (1/sparsity)


def minimax_tau_threshold(sparsity, signal_ncol):
    #root = newton(h_centered, 0, args=(sparsity, signal_ncol), maxiter = 1000, full_output=True)
    #root = fsolve(func = h_centered, x0 = -2, args = (sparsity, signal_ncol))
    root = brentq(h_centered, 0, 1e+10, args = (sparsity, signal_ncol))
    return root


def minimax_tau_threshold_approx(sparsity, signal_ncol):
    return (1-sparsity)*np.sqrt(signal_ncol)

