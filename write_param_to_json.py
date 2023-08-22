#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:39:18 2023

@author: apratimdey
"""

import json


def write_param_to_json(sparsity,
                        delta,
                        signal_nrow,
                        signal_ncol,
                        mc,
                        err_tol,
                        sparsity_tol,
                        max_iter,
                        output_json_filename,
                        output_data_filename,
                        denoiser):
    
    output_dict = {'sparsity': round(sparsity, 2),
                   'delta': round(delta, 2),
                   'signal_nrow': signal_nrow,
                   'signal_ncol': signal_ncol,
                   'mc': mc,
                   'err_tol': err_tol,
                   'sparsity_tol': sparsity_tol,
                   'max_iter': max_iter,
                   'output_data_filename': output_data_filename,
                   'denoiser': denoiser}
    
    with open(output_json_filename, 'a') as fp:
        fp.write(json.dumps(output_dict) + '\n')