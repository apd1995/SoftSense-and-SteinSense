#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:47:22 2023

@author: apratimdey
"""

import numpy as np
from scipy import integrate
from scipy.stats import chi2, ncx2


def JS_risk_at_zero(signal_ncol, sigma):
    def integrand(x, signal_ncol):
        return  (1 - ((signal_ncol-2)/x))**2 * x * chi2.pdf(x, df = signal_ncol)
    return sigma**2 * integrate.quad(integrand, signal_ncol-2, np.inf, args=signal_ncol)[0]/signal_ncol


def JS_risk_nonzero_integrand(x, mean_norm_sq, signal_ncol, sigma):
    d = signal_ncol
    return np.where(x<sigma**2 * (d-2), x - (d*sigma**2), (d*sigma**2) - (sigma**4*(d-2)**2/x)) * ncx2.pdf(x, d, mean_norm_sq)


def JS_risk_nonzero_mean(mean_norm_sq, signal_ncol, sigma):
    return integrate.quad(JS_risk_nonzero_integrand, 0, np.inf, args=(mean_norm_sq, signal_ncol, sigma))[0]


def JS_risk_nonzero_normal_integrand(x, signal_ncol, sigma):
    return JS_risk_nonzero_mean(x, signal_ncol, sigma)*chi2.pdf(x, signal_ncol)


def JS_risk_nonzero_normal(signal_ncol, sigma):
    return integrate.quad(JS_risk_nonzero_normal_integrand, 0, np.inf, args=(signal_ncol, sigma))[0]/signal_ncol


def JS_risk(sparsity, signal_ncol, sigma):
    return (1-sparsity)*JS_risk_at_zero(signal_ncol, sigma) + sparsity*JS_risk_nonzero_normal(signal_ncol, sigma)


def state_evolution(sparsity, signal_ncol, sigma, delta):
    return JS_risk(sparsity, signal_ncol, sigma)/delta
    