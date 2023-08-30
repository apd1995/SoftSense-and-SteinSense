#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:48:06 2023

@author: apratimdey
"""

# needs editing

import numpy as np
import pandas as pd
import cvxpy as cvx
import time
import amp_iteration as amp


def seed(sparsity: float,
         delta: float,
         N: int,
         B: float,
         rel_err_tol: float,
         mc: int,
         sparsity_tol: float) -> int:
    return round(1 + round(sparsity * 1000) + round(delta * 1000) + round(N * 1000) + round(B * 1000) + round(rel_err_tol * 100000) + mc * 100000 + round(sparsity_tol * 1000000))


def gen_iid_normal_mtx(num_measurements, signal_nrow, rng):
    """
    Generates a single random num_measurements by signal_nrow matrix with iid signal_nrow(0,1) entries

    Parameters
    ----------
    num_measurements : int
        Number of rows of measurement matrix.
    signal_nrow : int
        Number of rows of signal matrix.

    Returns
    -------
    numpy.ndarray
        num_measurements by signal_nrow matrix.

    """
    
    return rng.normal(0, 1, (num_measurements, signal_nrow))


def rec_stats(X_true: float,
              X_rec: float,
              sparsity: float,
              delta: float,
              k: int,
              n: int,
              mc: int,
              rel_err_tol: float,
              sparsity_tol: float,
              err_explosion_tol: float,
              iter_count: int,
              time_seconds: float,
              denoiser: str,
              output_filename: str):
    
    N, B = X_true.shape
    start_time = time.time()
    rel_err = cvx.norm(X_true-X_rec, "fro").value/cvx.norm(X_true, "fro").value
    avg_err = cvx.norm(X_true - X_rec, "fro").value/np.sqrt(N*B)
    max_row_err = cvx.mixed_norm(X_true - X_rec, 2, np.inf).value/np.sqrt(B)
    norm_2_1_true = cvx.mixed_norm(X_true, 2, 1).value/(N*np.sqrt(B))
    norm_2_1_rec = cvx.mixed_norm(X_rec, 2, 1).value/(N*np.sqrt(B))
    norm_2_2_true = cvx.mixed_norm(X_true, 2, 2).value/np.sqrt(N*B)
    norm_2_2_rec = cvx.mixed_norm(X_rec, 2, 2).value/np.sqrt(N*B)
    norm_2_infty_true = cvx.mixed_norm(X_true, 2, np.inf).value/np.sqrt(B)
    norm_2_infty_rec = cvx.mixed_norm(X_rec, 2, np.inf).value/np.sqrt(B)
    soft_sparsity = np.mean(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > sparsity_tol)
    nonzero_rows_rec = np.sum(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > sparsity_tol)
    zero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)==0)
    zero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)<=sparsity_tol)
    tpr = sum(zero_indices_true * zero_indices_rec)/max(1, sum(zero_indices_true))
    nonzero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)!=0)
    nonzero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)>sparsity_tol)
    tnr = sum(nonzero_indices_true * nonzero_indices_rec)/max(1, sum(nonzero_indices_true))
    end_time = time.time()
    time_seconds = round(time_seconds + end_time - start_time, 2)
    
    out_dict = {'sparsity': sparsity,
                'delta': delta,
                'signal_nrow': N,
                'signal_ncol': B,
                'nonzero_rows_true': k,
                'num_measurements': n,
                'mc': mc,
                'iter_count': iter_count,
                'err_explosion_tol': err_explosion_tol,
                'rel_err_tol': rel_err_tol,
                'rel_err': rel_err,
                'avg_err': avg_err,
                'max_row_err': max_row_err,
                'norm_2_1_true': norm_2_1_true,
                'norm_2_1_rec': norm_2_1_rec,
                'norm_2_2_true': norm_2_2_true,
                'norm_2_2_rec': norm_2_2_rec,
                'norm_2_infty_true': norm_2_infty_true,
                'norm_2_infty_rec': norm_2_infty_rec,
                'sparsity_tol': sparsity_tol,
                'soft_sparsity': soft_sparsity,
                'nonzero_rows_rec': nonzero_rows_rec,
                'tpr': tpr,
                'tnr': tnr,
                'denoiser': denoiser,
                'time_seconds': time_seconds}
    df_colnames = list(out_dict)
    out_df = pd.DataFrame(out_dict, index=[0])
    out_df.to_csv(output_filename, mode = 'a', columns = df_colnames, header = False, index = True, index_label = True)
    return out_dict


def run_amp_instance(sparsity: float,
                     delta: float,
                     signal_nrow: int,
                     signal_ncol: int,
                     mc: int,
                     max_iter:int,
                     rel_err_tol: float,
                     sparsity_tol: float,
                     err_explosion_tol: float,
                     output_filename: str):
    
    denoiser = "james-stein"
    
    rng = np.random.default_rng(seed=seed(sparsity, delta, signal_nrow, signal_ncol, rel_err_tol, mc, sparsity_tol))

    k = int(sparsity * signal_nrow) + 1
    X_true = np.zeros((signal_nrow, signal_ncol), dtype=float)
    nonzero_indices = rng.choice(range(signal_nrow), k, replace=False)
    X_true[nonzero_indices, :] = rng.normal(0, 1, (k, signal_ncol))
    
    num_measurements = int(delta * signal_nrow) + 1
    A = gen_iid_normal_mtx(num_measurements, signal_nrow, rng)/np.sqrt(num_measurements)
    Y = A @ X_true
    
    N = signal_nrow
    B = signal_ncol
    n = num_measurements
    
    iter_count = 0
    start_time = time.time()
    X_denoised_current = np.zeros((N, B), dtype = float)
    Residual_current = Y
    end_time = time.time()
    time_seconds = round(end_time - start_time, 2)
    rec_stats_dict = rec_stats(X_true,
                               X_denoised_current,
                               sparsity,
                               delta,
                               k,
                               n,
                               mc,
                               rel_err_tol,
                               sparsity_tol,
                               err_explosion_tol,
                               iter_count,
                               time_seconds,
                               denoiser,
                               output_filename)
    rel_err = rec_stats_dict['rel_err']
    
    while iter_count<max_iter and rel_err>rel_err_tol and rel_err<err_explosion_tol:
        iter_count = iter_count + 1
        start_time = time.time()
        X_denoised_prev = X_denoised_current
        Residual_prev = Residual_current
        noise_cov_current = Residual_prev.T @ Residual_prev/n
        dict_current = amp.amp_iteration(A, Y, X_denoised_prev, Residual_prev, noise_cov_current)
        X_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        end_time = time.time()
        time_seconds = end_time - start_time
        rec_stats_dict = rec_stats(X_true,
                                   X_denoised_current,
                                   sparsity,
                                   delta,
                                   k,
                                   n,
                                   mc,
                                   rel_err_tol,
                                   sparsity_tol,
                                   err_explosion_tol,
                                   iter_count,
                                   time_seconds,
                                   denoiser,
                                   output_filename)
        rel_err = rec_stats_dict['rel_err']
        
run_amp_instance(0.2, 0.25, 500, 10, 0, 500, 1e-5, 1e-4, 1e+7, "AMP_500_10_try.csv")

# def run_amp_experiment(filepath):
    
#     with open(filepath) as user_file:
#         file_contents = json.load(user_file)
    
#     N_grid_upper = file_contents['signal_nrow_upper']
#     N_grid_lower = file_contents['signal_nrow_lower']
#     N_grid_len = file_contents['signal_nrow_len']
#     N_grid = np.linspace(N_grid_lower, N_grid_upper, num = N_grid_len, dtype = int)
#     B_grid_upper = file_contents['signal_ncol_upper']
#     B_grid_lower = file_contents['signal_ncol_lower']
#     B_grid_len = file_contents['signal_ncol_len']
#     B_grid = np.linspace(B_grid_lower, B_grid_upper, num = B_grid_len, dtype = int)
#     sparsity_grid_lower = file_contents['sparsity_grid_lower']
#     sparsity_grid_upper = file_contents['sparsity_grid_upper']
#     sparsity_grid_len = int(file_contents['sparsity_grid_len'])
#     sparsity_grid = np.linspace(sparsity_grid_lower, sparsity_grid_upper, num = sparsity_grid_len)
#     delta_grid_lower = file_contents['delta_grid_lower']
#     delta_grid_upper = file_contents['delta_grid_upper']
#     delta_grid_len = int(file_contents['delta_grid_len'])
#     delta_grid = np.linspace(delta_grid_lower, delta_grid_upper, num = delta_grid_len)
#     nMonte = int(file_contents['nMonte'])
#     rel_err_tol = file_contents['rel_err_tol']
#     output_filename = file_contents['output_filename']
#     sparsity_tol = file_contents['sparsity_tol']
#     max_iter = file_contents['max_iter']

#     pool = multiprocessing.Pool()
#     pool.starmap(amp_df, [(sparsity_grid[a], delta_grid[b], N_grid[c], B_grid[d], mc, max_iter, rel_err_tol, sparsity_tol, output_filename) for mc in range(1,nMonte+1) for d in range(len(B_grid)) for c in range(len(N_grid)) for b in range(len(delta_grid)) for a in range(len(sparsity_grid))])
    

# if __name__ == '__main__':
#     run_amp_experiment('amp_inputs.json')

