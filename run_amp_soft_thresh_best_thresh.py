#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 09:03:09 2025

@author: apratimdey
"""

import numpy as np
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


def seed(num_nonzeros: float,
         signal_dim: float,
         num_measurements: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float) -> int:
    return round(1 + round(signal_dim * 1000) + round(num_measurements * 1000) + round(num_nonzeros * 1000) + round(mc * 1000) + round(err_tol * 10000) + round(sparsity_tol * 1000))


def soft_thresh(X, thresh, sigma):
    return np.where(np.abs(X) > thresh * sigma, X - np.sign(X) * thresh * sigma, 0.)


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)


def update_signal_denoised(signal_noisy_current: float,
                           noise_var_current: float,
                           tau: float):
    sigma = np.sqrt(noise_var_current)
    return soft_thresh(signal_noisy_current, thresh = tau, sigma = sigma)


def soft_thresh_div(signal_noisy, noise_var, tau):
    sigma = np.sqrt(noise_var)
    return np.sum(np.abs(signal_noisy) > tau * sigma)


def soft_thresh_onsager(X, noise_var, Z, n, tau):
    n = len(Z)
    return Z * soft_thresh_div(X, noise_var, tau) / n
    

def update_residual(A,
                    Y,
                    signal_noisy_current,
                    signal_denoised_current,
                    Residual_prev,
                    noise_var_current,
                    tau):
    naive_residual = Y - A @ signal_denoised_current
    onsager_term_ = soft_thresh_onsager(signal_noisy_current,
                                        noise_var_current,
                                        Residual_prev,
                                        len(Y),
                                        tau)
    return naive_residual + onsager_term_


def amp_iteration(A: float,
                Y: float,
                signal_denoised_prev: float,
                Residual_prev: float,
                noise_var_current: float,
                tau: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    signal_denoised_current = update_signal_denoised(signal_noisy_current,
                                                     noise_var_current,
                                                     tau)
    Residual_current = update_residual(A,
                                    Y,
                                    signal_noisy_current,
                                    signal_denoised_current,
                                    Residual_prev,
                                    noise_var_current,
                                    tau)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


def gen_iid_normal_mtx(num_measurements, signal_dim, rng):
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
    return rng.normal(0, 1, (num_measurements, signal_dim))


def recovery_stats(X_true: float,
              X_rec: float,
              sparsity_tol: float,
              A: np.ndarray):

    zero_indices_true = (X_true==0)
    zero_indices_rec = (np.abs(X_rec)<=sparsity_tol)

    nonzero_indices_true = (X_true!=0)
    nonzero_indices_rec = (np.abs(X_rec)>sparsity_tol)
    
    dict_observables = {
                'rel_err': np.linalg.norm(X_true-X_rec)/np.linalg.norm(X_true),
                'avg_err': np.linalg.norm(X_true - X_rec)/np.sqrt(len(X_true)),
                'soft_sparsity': np.mean(np.abs(X_rec) > sparsity_tol),
                'tpr': sum(zero_indices_true * zero_indices_rec)/max(1, sum(zero_indices_true)),
                'tnr': sum(nonzero_indices_true * nonzero_indices_rec)/max(1, sum(nonzero_indices_true))
                }
    
    return dict_observables


def add_row_to_df(dict_to_add, df):
    return concat([df, DataFrame(dict_to_add, index = [0])], ignore_index=True)


def run_amp_instance(**dict_params):
    
    k = dict_params['num_nonzeros']
    n = dict_params['num_measurements']
    N = dict_params['signal_dim']
    err_tol = dict_params['err_tol']
    mc = dict_params['mc']
    sparsity_tol = dict_params['sparsity_tol']
    max_iter = dict_params['max_iter']
    
    iter_count = 0
    
    rng = np.random.default_rng(seed=seed(k, N, n, err_tol, mc, sparsity_tol))
    signal_true = np.zeros(N, dtype=float)
    nonzero_indices = rng.choice(range(N), k, replace=False)
    signal_true[nonzero_indices] = rng.normal(0, 1, k)
    signal_true = np.array(signal_true)
   
    A = gen_iid_normal_mtx(n, N, rng)/np.sqrt(n)
    Y_true = A @ signal_true
    
    dict_params['undersampling_ratio'] = n/N
    
    output_df = None
    
    signal_denoised_current = np.zeros(N, dtype = float)
    Residual_current = Y_true
    
    dict_observables = recovery_stats(signal_true,
                               signal_denoised_current,
                               sparsity_tol,
                               A)
    rel_err = dict_observables['rel_err']
    min_rel_err = rel_err
    
    while iter_count<max_iter and rel_err>err_tol:
        tick = time.perf_counter()
        
        iter_count = iter_count + 1

        noise_var_current = np.var(Residual_current)
        thresh_grid = np.linspace(0.01, 2, 200)
        measurement_err_grid = []

        for thresh_val in thresh_grid:
            dict_current = amp_iteration(
                A=A,
                Y=Y_true,
                signal_denoised_prev=signal_denoised_current,
                Residual_prev=Residual_current,
                noise_var_current=noise_var_current,
                tau=thresh_val
            )
            measurement_err_grid.append(np.var(dict_current['Residual_current']))

        best_thresh = thresh_grid[np.argmin(measurement_err_grid)]
        
        dict_current = amp_iteration(A = A, 
                                     Y = Y_true, 
                                    signal_denoised_prev = signal_denoised_current,
                                    Residual_prev = Residual_current,
                                    noise_var_current = noise_var_current,
                                    tau = best_thresh)
    
        signal_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        
        dict_observables = recovery_stats(signal_true,
                                   signal_denoised_current,
                                   sparsity_tol,
                                   A)
    
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
               'num_nonzeros': 100,
               'num_measurements': 310,
               'signal_dim': 1000,
               'max_iter': 500,
               'err_tol': 1e-5,
               'sparsity_tol': 1e-4,
               'err_explosion_tol': 100,
               'mc': 0,

           }
    return exp

dict_params = test_experiment()

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
    # do_sherlock_experiment('exp_dicts/array_AMP_ST_best_thresh.json')
    do_local_experiment()