#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:09:06 2023

@author: apratimdey
"""

import multiprocessing as mp
from get_param_grids_from_json import get_param_grids_from_json
from write_param_to_json import write_param_to_json


def exec_write_param_to_json(input_json_filename: str):
    
    param_grids_dict = get_param_grids_from_json(input_json_filename)
    
    signal_nrow_grid = param_grids_dict['signal_nrow_grid']
    signal_ncol_grid = param_grids_dict['signal_ncol_grid']
    sparsity_grid = param_grids_dict['sparsity_grid']
    delta_grid = param_grids_dict['delta_grid']
    nMonte = param_grids_dict['nMonte']
    err_tol = param_grids_dict['err_tol']
    sparsity_tol = param_grids_dict['sparsity_tol']
    output_json_filename = param_grids_dict['output_json_filename']
    output_data_filename = param_grids_dict['output_data_filename']
    denoiser = param_grids_dict['denoiser']
    max_iter = param_grids_dict['max_iter']

    pool = mp.Pool()
    pool.starmap(write_param_to_json, [(sparsity, delta, signal_nrow, signal_ncol, mc, err_tol, sparsity_tol, max_iter, output_json_filename, output_data_filename, denoiser) for mc in range(nMonte) for signal_ncol in signal_ncol_grid for signal_nrow in signal_nrow_grid for delta in delta_grid for sparsity in sparsity_grid])
    
    
if __name__=="__main__":
    exec_write_param_to_json("AMP_inputs.json")