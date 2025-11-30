#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 19:02:59 2025

@author: apratimdey
"""
import numpy as np
import cvxpy as cvx
from pandas import DataFrame, concat
import time
from EMS.manager import do_on_cluster, get_gbq_credentials, read_json
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import logging

logging.basicConfig(level=logging.INFO)
log_gbq = logging.getLogger('pandas_gbq')
log_gbq.setLevel(logging.DEBUG)
log_gbq.addHandler(logging.StreamHandler())
logging.getLogger('jax').setLevel(logging.ERROR)

def seed(block_idx: int,
         num_nonzero_blocks: float,
         num_nonzeros_per_block: float,
         num_measurements: float,
         num_blocks: float,
         block_size: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float) -> int:
    return round(1 + round(block_idx*1000) + round(num_nonzero_blocks * 1000) + round(num_nonzeros_per_block * 1000) + round(num_measurements * 1000) + round(num_blocks * 1000) + round(block_size * 1000) + round(err_tol * 100000) + round(mc * 100000) + round(sparsity_tol * 1000000))


def james_stein_block(y, sigma_sq):
    d = len(y)
    quad_whitening = np.sum(y**2)/sigma_sq
    if quad_whitening > (d-2):
        return y * (1 - ((d-2)/quad_whitening))
    else:
        return np.zeros_like(y, dtype = float)
    

def james_stein_full_vec(X, M, B, sigma_sq) -> np.ndarray:
    """
    Applies james stein rowwise to denoise Y.

    Parameters
    ----------
    Y : np.ndarray
        Noisy signal.
    Sigma_inv : np.ndarray
        Noise precision matrix.

    Returns
    -------
    None.
    """
    X_denoised = np.zeros_like(X, dtype = float)
    for block_idx in range(M):
        X_denoised[(block_idx*B):((block_idx + 1)*B)] = james_stein_block(X[(block_idx*B):((block_idx + 1)*B)], sigma_sq)
    return X_denoised

def soft_thresh(X, thresh, sigma_sq):
    sigma = np.sqrt(sigma_sq)
    return np.where(np.abs(X) > thresh*sigma, X - np.sign(X)*thresh*sigma, 0.)

def combined_denoiser_full_vec(X, M, B, thresh, sigma_sq):
    tmp = james_stein_full_vec(X, M, B, sigma_sq)
    res = soft_thresh(tmp, thresh, sigma_sq)
    return res


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)


def update_signal_denoised_combined(signal_noisy_current: float,
                           num_blocks,
                           block_size,
                           thresh,
                           noise_var_current: float):
    return combined_denoiser_full_vec(signal_noisy_current, num_blocks, block_size, thresh, noise_var_current)


def combined_div(X, M, B, thresh, noise_var, n):
    norm_sq_vals = np.zeros(M, dtype = float)
    for block_idx in range(M):
        norm_sq_vals[block_idx] = np.sum(X[(block_idx*B):((block_idx + 1)*B)]**2)/noise_var
    norm_sq_vals_full = np.repeat(norm_sq_vals, B)
    X_scaled_sq = X**2/noise_var
    coeffs_JS = np.where(norm_sq_vals_full>(B-2),
                         1 - ((B-2)/norm_sq_vals_full) + (2*(B-2)*X_scaled_sq/norm_sq_vals_full**2), 
                         0.)
    X_JS = james_stein_full_vec(X, M, B, noise_var)
    noise_sd = np.sqrt(noise_var)
    coeffs_ST = np.where(np.abs(X_JS) > thresh * noise_sd, 1., 0.)
    return np.sum(coeffs_ST * coeffs_JS)/n


def combined_onsager(X, M, B, thresh, noise_var, Z, n):
    X = np.array(X)
    onsager_term = combined_div(X, M, B, thresh, noise_var, n)*Z
    return onsager_term
    

def update_residual_combined(A,
                    Y,
                    signal_noisy_current,
                    signal_denoised_current,
                    Residual_prev,
                    num_blocks,
                    block_size,
                    thresh,
                    noise_var_current):
    naive_residual = Y - A @ signal_denoised_current
    onsager_term_ = combined_onsager(signal_noisy_current,
                                        num_blocks,
                                        block_size,
                                        thresh,
                                        noise_var_current,
                                        Residual_prev,
                                        len(Y))
    return naive_residual + onsager_term_


def amp_iteration_combined(A: float,
                Y: float,
                signal_denoised_prev: float,
                Residual_prev: float,
                num_blocks: int,
                block_size: int,
                thresh: float,
                noise_var_current: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    signal_denoised_current = update_signal_denoised_combined(signal_noisy_current,
                                                     num_blocks,
                                                     block_size,
                                                     thresh,
                                                     noise_var_current)
    Residual_current = update_residual_combined(A,
                                    Y,
                                    signal_noisy_current,
                                    signal_denoised_current,
                                    Residual_prev,
                                    num_blocks,
                                    block_size,
                                    thresh,
                                    noise_var_current)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


def gen_iid_normal_mtx(num_measurements, num_blocks, block_size, rng):
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
    return rng.normal(0, 1, (num_measurements, num_blocks * block_size))


def recovery_stats(X_true: float,
              X_rec: float,
              sparsity_tol: float,
              A: np.ndarray,
              Y_true: np.ndarray,
              M, B):
    
    X_true_mat = X_true.reshape((M, B))
    X_rec_mat = X_rec.reshape((M, B))

    zero_blocks_true = (np.apply_along_axis(np.linalg.norm, 1, X_true_mat)==0)
    zero_blocks_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec_mat)/np.sqrt(B)<=sparsity_tol)

    nonzero_blocks_true = (np.apply_along_axis(np.linalg.norm, 1, X_true_mat)!=0)
    nonzero_blocks_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec_mat)/np.sqrt(B)>sparsity_tol)
    
    dict_observables = {
                'rel_err': cvx.norm(X_true_mat-X_rec_mat, "fro").value/cvx.norm(X_true_mat, "fro").value,
                'avg_err': cvx.norm(X_true_mat - X_rec_mat, "fro").value/np.sqrt(M*B),
                'max_row_err': cvx.mixed_norm(X_true_mat - X_rec_mat, 2, np.inf).value/np.sqrt(B),
                'norm_2_1_true': cvx.mixed_norm(X_true_mat, 2, 1).value/(M*np.sqrt(B)),
                'norm_2_1_rec': cvx.mixed_norm(X_rec_mat, 2, 1).value/(M*np.sqrt(B)),
                'norm_2_2_true': cvx.mixed_norm(X_true_mat, 2, 2).value/np.sqrt(M*B),
                'norm_2_2_rec': cvx.mixed_norm(X_rec_mat, 2, 2).value/np.sqrt(M*B),
                'norm_2_infty_true': cvx.mixed_norm(X_true_mat, 2, np.inf).value/np.sqrt(B),
                'norm_2_infty_rec': cvx.mixed_norm(X_rec_mat, 2, np.inf).value/np.sqrt(B),
                'soft_sparsity': np.mean(np.apply_along_axis(np.linalg.norm, 1, X_rec_mat)/np.sqrt(B) > sparsity_tol),
                'nonzero_rows_rec': np.sum(np.apply_along_axis(np.linalg.norm, 1, X_rec_mat)/np.sqrt(B) > sparsity_tol),
                'tpr': sum(zero_blocks_true * zero_blocks_rec)/max(1, sum(zero_blocks_true)),
                'tnr': sum(nonzero_blocks_true * nonzero_blocks_rec)/max(1, sum(nonzero_blocks_true))
                }
    
    return dict_observables


def add_row_to_df(dict_to_add, df):
    return concat([df, DataFrame(dict_to_add, index = [0])], ignore_index=True)


def run_amp_instance(**dict_params):
    
    k = dict_params['num_nonzero_blocks']
    l = dict_params['num_nonzeros_per_block']
    n = dict_params['num_measurements']
    M = dict_params['num_blocks']
    B = dict_params['block_size']
    err_tol = dict_params['err_tol']
    mc = dict_params['mc']
    sparsity_tol = dict_params['sparsity_tol']
    max_iter = dict_params['max_iter']
    err_explosion_tol = dict_params['err_explosion_tol']
    
    iter_count = 0
    N = M * B
    
    rng = np.random.default_rng(seed=seed(2*N, k, l, n, M, B, err_tol, mc, sparsity_tol))
    signal_true = np.zeros(N, dtype=float)
    nonzero_block_indices = rng.choice(range(M), k, replace=False)
    for nonzero_block_idx in nonzero_block_indices:
        rng_block = np.random.default_rng(seed=seed(nonzero_block_idx, k, l, n, M, B, err_tol, mc, sparsity_tol))
        block_nonzero_indices = nonzero_block_idx*B + rng.choice(range(B), l, replace= False)
        signal_true[block_nonzero_indices] = rng_block.normal(0, 1, l)
    signal_true = np.array(signal_true)
   
    A = gen_iid_normal_mtx(n, M, B, rng)/np.sqrt(n)
    Y_true = np.matmul(A, signal_true)
    
    block_sparsity = k/M
    per_block_sparsity = l/B
    dict_params['block_sparsity'] = block_sparsity
    dict_params['per_block_sparsity'] = per_block_sparsity
    dict_params['undersampling_ratio'] = n/N
    
    output_df = None
    
    iter_count = 0
    
    signal_denoised_current = np.zeros(N, dtype = float)
    Residual_current = Y_true
    
    dict_observables = recovery_stats(signal_true,
                               signal_denoised_current,
                               sparsity_tol,
                               A,
                               Y_true, M, B)
    rel_err = dict_observables['rel_err']
    min_rel_err = rel_err
    
    while iter_count<max_iter and rel_err>err_tol and rel_err<err_explosion_tol:
        
        tick = time.perf_counter()
        
        iter_count = iter_count + 1

        noise_var_current = np.var(Residual_current)
        thresh_grid = np.linspace(0.01, 2, 200)
        measurement_err_grid = []

        for thresh_val in thresh_grid:
            dict_current = amp_iteration_combined(
                A=A,
                Y=Y_true,
                signal_denoised_prev=signal_denoised_current,
                Residual_prev=Residual_current,
                num_blocks=M,
                block_size=B,
                thresh=thresh_val,
                noise_var_current=noise_var_current
            )
            measurement_err_grid.append(np.mean(dict_current['Residual_current']**2))

        best_thresh = thresh_grid[np.argmin(measurement_err_grid)]
        
        dict_current = amp_iteration_combined(A = A, 
                                     Y = Y_true, 
                                    signal_denoised_prev = signal_denoised_current,
                                    Residual_prev = Residual_current, 
                                    num_blocks = M,
                                    block_size = B,
                                    thresh = best_thresh,
                                    noise_var_current = noise_var_current)
    
        signal_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        
        dict_observables = recovery_stats(signal_true,
                                   signal_denoised_current,
                                   sparsity_tol,
                                   A,
                                   Y_true, M, B)
    
        rel_err = dict_observables['rel_err']
        min_rel_err = min(rel_err, min_rel_err)
        tock = time.perf_counter() - tick
        print(rel_err, round(tock, 2), best_thresh)
        
        if iter_count % 50 == 0:
            dict_observables['min_rel_err'] = min_rel_err
            dict_observables['best_thresh'] = best_thresh
            dict_observables['iter_count'] = iter_count
            dict_observables['time_seconds'] = round(tock, 2)
            combined_dict = {**dict_params, **dict_observables}
            output_df = add_row_to_df(combined_dict, output_df)

    if iter_count % 50 != 0:
        dict_observables['min_rel_err'] = min_rel_err
        dict_observables['best_thresh'] = best_thresh
        dict_observables['iter_count'] = iter_count
        dict_observables['time_seconds'] = round(tock, 2)
        combined_dict = {**dict_params, **dict_observables}
        output_df = add_row_to_df(combined_dict, output_df)

    return output_df


def test_experiment() -> dict:
    exp = {
               'num_nonzero_blocks': 400,
               'num_nonzeros_per_block': 2,
               'num_measurements': 4000,
               'num_blocks': 1000,
               'block_size': 10,
               'max_iter': 1000,
               'err_tol': 1e-5,
               'sparsity_tol': 1e-4,
               'err_explosion_tol': 100,
               'mc': 100,

           }
    return exp

def do_local_experiment():
    exp = test_experiment()
    run_amp_instance(**exp)

def do_sherlock_experiment(json_file: str):
    exp = read_json(json_file)
    nodes = 500
    with SLURMCluster(queue='donoho,stat,hns,owners,normal',
                      cores=1, memory='12GiB', processes=1,
                      walltime='24:00:00') as cluster:
        cluster.scale(jobs=nodes)
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())
        cluster.scale(0)
        

if __name__ == '__main__':
    do_sherlock_experiment('exp_dicts/array_AMP_JS_ST_combined_best_thresh.json')
    # do_local_experiment()