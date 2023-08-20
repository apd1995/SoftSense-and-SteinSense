#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:49:43 2023

@author: apratimdey
"""

import json
import numpy as np


def get_param_grids_from_json(json_filename):
    
    with open(json_filename) as file:
        file_contents = json.load(file)

    signal_nrow_upper = file_contents['signal_nrow_upper']
    signal_nrow_lower = file_contents['signal_nrow_lower']
    signal_nrow_len = file_contents['signal_nrow_len']
    signal_nrow_grid = np.linspace(signal_nrow_lower, signal_nrow_upper, num = signal_nrow_len)
    signal_ncol_upper = file_contents['signal_ncol_upper']
    signal_ncol_lower = file_contents['signal_ncol_lower']
    signal_ncol_len = file_contents['signal_ncol_len']
    signal_ncol_grid = np.linspace(signal_ncol_lower, signal_ncol_upper, num = signal_ncol_len)
    sparsity_grid_lower = file_contents['sparsity_grid_lower']
    sparsity_grid_upper = file_contents['sparsity_grid_upper']
    sparsity_grid_len = int(file_contents['sparsity_grid_len'])
    sparsity_grid = np.linspace(sparsity_grid_lower, sparsity_grid_upper, num = sparsity_grid_len)
    delta_grid_lower = file_contents['delta_grid_lower']
    delta_grid_upper = file_contents['delta_grid_upper']
    delta_grid_len = int(file_contents['delta_grid_len'])
    delta_grid = np.linspace(delta_grid_lower, delta_grid_upper, num = delta_grid_len)
    nMonte = int(file_contents['nMonte'])
    err_tol = file_contents['err_tol']
    output_json_filename = file_contents['output_json_filename']
    output_data_filename = file_contents['output_data_filename']
    sparsity_tol = file_contents['sparsity_tol']
    denoiser = file_contents['denoiser']
    max_iter = file_contents['max_iter']
    
    return {'signal_nrow_grid': signal_nrow_grid,
            'signal_ncol_grid': signal_ncol_grid,
            'sparsity_grid': sparsity_grid,
            'delta_grid': delta_grid,
            'nMonte': nMonte,
            'err_tol': err_tol,
            'sparsity_tol': sparsity_tol,
            'max_iter': max_iter,
            'output_json_filename': output_json_filename,
            'output_data_filename': output_data_filename,
            'denoiser': denoiser}
