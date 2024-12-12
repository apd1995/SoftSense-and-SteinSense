#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:45:43 2023

@author: apratimdey
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy

def logreg_data(x: float, y: float) -> dict:
    """
    Performs logistic regression with predictor array x and response array y and
    outputs the intercept, slope and LD50.

    Parameters
    ----------
    x : float
        x values for logistic regression.
    y : float
        Binary response for logistic regression.

    Returns
    -------
    dict
        Dictionary containing the intercept, slope and LD50 from logistic regression.

    """
    
    logreg = LogisticRegression(penalty = None)
    
    x = np.array(x)
    y = np.array(y)
    y = [int(y[j]) for j in range(len(y))]
    
    
    if sum(y) == len(y):
        return "All successes. Try higher sparsity than "+str(max(x))
    
    elif sum(y) == 0:
        return "All failures. Try lower sparsity than "+str(min(x))
    
    else:
    
        logreg.fit(x.reshape(-1, 1), y)
        
        intercept = logreg.intercept_.item()
        slope = logreg.coef_.item()
        LD50 = -intercept/slope
        
        return {'intercept': intercept,
                'slope': slope,
                'LD50': LD50}


def two_point_design_logistic(x: float, y: float, p1: float, p2: float ) -> dict:
    
    dat = logreg_data(x, y)
    
    if type(dat) == str:
        return dat
    
    else:
    
        alpha = dat['LD50']
        beta = dat['slope']
        
        q1 = -np.log(1/p1 - 1)
        q2 = -np.log(1/p2 - 1)
        
        return {'dose1': q1/beta + alpha,
                'dose2': q2/beta + alpha}


def white_optimal_two_point_logistic(x: float, y: float) -> dict:
    return two_point_design_logistic(x, y, p1 = 0.824, p2 = 0.176)


def LD50_var_fun(p: float, N: int, beta: float) -> float:
    q = 1 - p
    lamb = np.log(p/q)
    expr1 = 1/(beta**2 * p * q* N)
    expr2 = ((3*p**2 - p*q + 3*q**2)*lamb**2 - 6*lamb*(p-q) + 3)/(beta * lamb * p * q * N)**2
    return expr1 + expr2


def LD50_var_minimizer(N: int, beta: float) -> float:
    res = scipy.optimize.minimize_scalar(LD50_var_fun, bounds = (0, 1), args = (N, beta))
    return res.x


def kalish_optimal_two_point_logistic(x: float, y: float) -> dict:
    
    dat = logreg_data(x, y)
    
    if type(dat) == str:
        return dat
    
    else:
    
        N = len(y)
        dat = logreg_data(x, y)
        beta = dat['slope']
        alpha = dat['LD50']
        p_optim = LD50_var_minimizer(N, beta)
        
        p1 = p_optim
        p2 = 1-p_optim
        
        q1 = -np.log(1/p1 - 1)
        q2 = -np.log(1/p2 - 1)
        
        return {'dose1': q1/beta + alpha,
                'dose2': q2/beta + alpha}

