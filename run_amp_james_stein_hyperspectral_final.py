#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:30:20 2023

@author: apratimdey
"""

import numpy as np
from numpy.random import Generator
import cvxpy as cvx
from pandas import DataFrame, concat
import time

from EMS.manager import do_on_cluster, get_gbq_credentials, do_test_experiment, read_json, unroll_experiment
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import dask
import coiled
import logging
import json
import torch

logging.basicConfig(level=logging.INFO)
log_gbq = logging.getLogger('pandas_gbq')
log_gbq.setLevel(logging.DEBUG)
log_gbq.addHandler(logging.StreamHandler())
import jax
import jax.numpy as jnp
logging.getLogger('jax').setLevel(logging.ERROR)


def seed(iter_count: int,
         prob_quantile: float,
         num_measurements: float,
         signal_nrow: float,
         signal_ncol: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float,
         wavelet_level: int,
         band: int,
         subband: int) -> int:
    return round(1 + round(iter_count*1000) + round(prob_quantile * 1000) + round(num_measurements * 1000) + round(signal_nrow * 1000) + round(signal_ncol * 1000) + round(err_tol * 100000) + round(mc * 100000) + round(sparsity_tol * 1000000) + round(wavelet_level * 1000000) + round(band * 1000000) + round(subband * 1000000))


@jax.jit
def james_stein_nonsingular_vec(y,
                                Sigma_inv):
    d = len(y)
    quad_whitening = jnp.dot(y, jnp.dot(Sigma_inv, y))
    return jax.lax.cond(quad_whitening > (d-2),
                    lambda y: y * (1 - ((d-2)/quad_whitening)),   # True branch (lambda function)
                    lambda y: jnp.zeros(d),  # False branch (lambda function)
                    y)  # Operand to pass to selected branch
    

def james_stein_nonsingular(X: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
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
    james_stein_nonsingular_vectorized = jax.vmap(james_stein_nonsingular_vec, in_axes = (0, None))
    return james_stein_nonsingular_vectorized(X, Sigma_inv)


@jax.jit
def james_stein_diagonal_vec(y,
                             diag_inv):
    d = len(y)
    quad_whitening = jnp.sum(diag_inv * y**2)
    return jax.lax.cond(quad_whitening > (d-2),
                    lambda y: y * (1 - ((d-2)/quad_whitening)),   # True branch (lambda function)
                    lambda y: jnp.zeros(d),  # False branch (lambda function)
                    y)  # Operand to pass to selected branch


def james_stein_diagonal(X, diag_inv):
    james_stein_diagonal_vectorized = jax.vmap(james_stein_diagonal_vec, in_axes = (0, None))
    return james_stein_diagonal_vectorized(X, diag_inv)


@jax.jit
def james_stein_singular_vec(y,
                             Sigma_eigvecs,
                             nonzero_indices_int,
                             zero_indices_int,
                             Sigma_nonzero_eigvals_inv):
    
    # changing coordinates to get uncorrelated components
    y_indep = jnp.matmul(Sigma_eigvecs.T, y)
    
    y_indep_nonzero =  y_indep[jnp.array(nonzero_indices_int)]
    
    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = james_stein_diagonal_vec(y_indep_nonzero, Sigma_nonzero_eigvals_inv)
    
    # when D has a 0 entry, it means we have perfect precision
    # zero_indices = ~nonzero_indices
    y_indep_zero = y_indep[jnp.array(zero_indices_int)]
    signal_newbasis_zero = y_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = jnp.concatenate((signal_newbasis_zero, signal_newbasis_indep))
    # signal_newbasis = np.zeros(len(y), dtype = float)
    # signal_newbasis[nonzero_indices] = signal_newbasis_indep
    # signal_newbasis[zero_indices] = signal_newbasis_zero
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = jnp.matmul(Sigma_eigvecs, signal_newbasis)
    
    return signal_originalbasis


def james_stein_singular(X, Sigma_eigvecs,
                         nonzero_indices_int, zero_indices_int, Sigma_nonzero_eigvals_inv):
    james_stein_diagonal_vectorized = jax.vmap(james_stein_diagonal_vec, in_axes = (0, None, None, None, None))
    return james_stein_diagonal_vectorized(X, Sigma_eigvecs,
                             nonzero_indices_int, zero_indices_int, Sigma_nonzero_eigvals_inv)


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)



def update_signal_denoised_nonsingular(signal_noisy_current: float,
                                       noise_cov_current_inv: float):
    return james_stein_nonsingular(signal_noisy_current, noise_cov_current_inv)



def update_signal_denoised_singular(signal_noisy_current: float,
                                    noise_cov_current_eigvecs: float,
                                    noise_cov_current_nonzero_indices_int,
                                    noise_cov_current_zero_indices_int,
                                    noise_cov_current_nonzero_eigvals_inv: float):
    return james_stein_singular(signal_noisy_current, noise_cov_current_eigvecs,
                                            noise_cov_current_nonzero_indices_int,
                                            noise_cov_current_zero_indices_int,
                                            noise_cov_current_nonzero_eigvals_inv)


@jax.jit
def james_stein_onsager_nonsingular(X,
                                    Z,
                                    Sigma_inv):
    X = jnp.array(X)
    dd_jacobian = jax.jacfwd(james_stein_nonsingular_vec, argnums=0)
    # dd_jacobian = jax.jit(jax.jacfwd(james_stein_nonsingular_vec, argnums=0))
    jac_vectorized = jax.vmap(dd_jacobian, in_axes = (0, None))
    sum_jacobians = jac_vectorized(X, Sigma_inv).sum(axis = 0)
    onsager_term = jnp.matmul(Z, sum_jacobians.T)
    return onsager_term / Z.shape[0]
  
    
@jax.jit
def james_stein_onsager_singular(X,
                                 Z,
                                 Sigma_eigvecs,
                                 nonzero_indices_int,
                                 zero_indices_int,
                                 Sigma_nonzero_eigvals_inv):
    X = jnp.array(X)
    # selected_rows = rng.choice(X.shape[0], int(selected_rows_frac*X.shape[0]), replace = False)
    dd_jacobian = jax.jacfwd(james_stein_singular_vec, argnums=0)
    # dd_jacobian = jax.jit(jax.jacfwd(james_stein_singular_vec, argnums=0))
    jac_vectorized = jax.vmap(dd_jacobian, in_axes = (0, None, None, None, None))
    sum_jacobians = jac_vectorized(X, Sigma_eigvecs, nonzero_indices_int, zero_indices_int, Sigma_nonzero_eigvals_inv).sum(axis = 0)
    onsager_term = jnp.matmul(Z, sum_jacobians.T)
    return onsager_term / Z.shape[0]


def update_residual_singular(A: float,
                             Y: float,
                             signal_noisy_current: float,
                             signal_denoised_current: float,
                             Residual_prev: float,
                             noise_cov_current_eigvecs: float,
                             noise_cov_current_nonzero_indices_int,
                             noise_cov_current_zero_indices_int,
                             noise_cov_current_nonzero_eigvals_inv: float):
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = james_stein_onsager_singular(signal_noisy_current,
                                                         Residual_prev,
                                                         noise_cov_current_eigvecs,
                                                         noise_cov_current_nonzero_indices_int,
                                                         noise_cov_current_zero_indices_int,
                                                         noise_cov_current_nonzero_eigvals_inv)
    return naive_residual + onsager_term_


def update_residual_nonsingular(A: float,
                                Y: float,
                                signal_noisy_current: float,
                                signal_denoised_current: float,
                                Residual_prev: float,
                                noise_cov_current_inv: float):
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = james_stein_onsager_nonsingular(signal_noisy_current,
                                                            Residual_prev,
                                                            noise_cov_current_inv)
    return naive_residual + onsager_term_


def amp_iteration_nonsingular(A: float,
                              Y: float,
                              signal_denoised_prev: float,
                              Residual_prev: float,
                              noise_cov_current_inv: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    # noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised_nonsingular(signal_noisy_current, noise_cov_current_inv)
    Residual_current = update_residual_nonsingular(A,
                                                   Y,
                                                   signal_noisy_current,
                                                   signal_denoised_current,
                                                   Residual_prev,
                                                   noise_cov_current_inv)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


def amp_iteration_singular(A: float,
                           Y: float,
                           signal_denoised_prev: float,
                           Residual_prev: float,
                           noise_cov_current_eigvecs: float,
                           noise_cov_current_nonzero_indices_int,
                           noise_cov_current_zero_indices_int,
                           noise_cov_current_nonzero_eigvals_inv: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    # noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised_singular(signal_noisy_current, noise_cov_current_eigvecs,
                                                              noise_cov_current_nonzero_indices_int,
                                                              noise_cov_current_zero_indices_int,
                                                              noise_cov_current_nonzero_eigvals_inv)
    Residual_current = update_residual_singular(A, Y, signal_noisy_current, signal_denoised_current, Residual_prev,
                                                noise_cov_current_eigvecs, noise_cov_current_nonzero_indices_int,
                                                noise_cov_current_zero_indices_int,
                                                noise_cov_current_nonzero_eigvals_inv)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


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


def recovery_stats(X_true: float,
              X_rec: float,
              sparsity_tol: float,
              A: np.ndarray,
              Y_true: np.ndarray):
    
    N, B = X_true.shape
    n = Y_true.shape[0]
    Y_rec = np.matmul(A, X_rec)

    zero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)==0)
    zero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)<=sparsity_tol)

    nonzero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)!=0)
    nonzero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)>sparsity_tol)
    
    dict_observables = {
                'rel_err': cvx.norm(X_true-X_rec, "fro").value/cvx.norm(X_true, "fro").value,
                'rel_err_measurements': cvx.norm(Y_true-Y_rec, "fro").value/cvx.norm(Y_true, "fro").value,
                'avg_err': cvx.norm(X_true - X_rec, "fro").value/np.sqrt(N*B),
                'avg_err_measurements': cvx.norm(Y_true - Y_rec, "fro").value/np.sqrt(n*B),
                'max_row_err': cvx.mixed_norm(X_true - X_rec, 2, np.inf).value/np.sqrt(B),
                'max_row_err_measurements': cvx.mixed_norm(Y_true - Y_rec, 2, np.inf).value/np.sqrt(B),
                'norm_2_1_true': cvx.mixed_norm(X_true, 2, 1).value/(N*np.sqrt(B)),
                'norm_2_1_rec': cvx.mixed_norm(X_rec, 2, 1).value/(N*np.sqrt(B)),
                'norm_2_2_true': cvx.mixed_norm(X_true, 2, 2).value/np.sqrt(N*B),
                'norm_2_2_rec': cvx.mixed_norm(X_rec, 2, 2).value/np.sqrt(N*B),
                'norm_2_infty_true': cvx.mixed_norm(X_true, 2, np.inf).value/np.sqrt(B),
                'norm_2_infty_rec': cvx.mixed_norm(X_rec, 2, np.inf).value/np.sqrt(B),
                'soft_sparsity': np.mean(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > sparsity_tol),
                'nonzero_rows_rec': np.sum(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > sparsity_tol),
                'tpr': sum(zero_indices_true * zero_indices_rec)/max(1, sum(zero_indices_true)),
                'tnr': sum(nonzero_indices_true * nonzero_indices_rec)/max(1, sum(nonzero_indices_true)),
                'norm_2_1_true_measurements': cvx.mixed_norm(Y_true, 2, 1).value/(n*np.sqrt(B)),
                'norm_2_1_rec_measurements': cvx.mixed_norm(Y_rec, 2, 1).value/(n*np.sqrt(B)),
                'norm_2_2_true_measurements': cvx.mixed_norm(Y_true, 2, 2).value/np.sqrt(n*B),
                'norm_2_2_rec_measurements': cvx.mixed_norm(Y_rec, 2, 2).value/np.sqrt(n*B),
                'norm_2_infty_true_measurements': cvx.mixed_norm(Y_true, 2, np.inf).value/np.sqrt(B),
                'norm_2_infty_rec_measurements': cvx.mixed_norm(Y_rec, 2, np.inf).value/np.sqrt(B)
                }
    
    return dict_observables


def add_row_to_df(dict_to_add, df):
    return concat([df, DataFrame(dict_to_add, index = [0])], ignore_index=True)


def get_sparsified_array(original_array, prob_quantile):
  norm_list = np.array([np.sum(original_array[:, i, j]**2) for i in range(original_array.shape[1]) for j in range(original_array.shape[2])])
  quantile_threshold = np.quantile(norm_list, q = 1 - prob_quantile)
  sparse_array = np.zeros_like(original_array, dtype = float)
  for i in range(original_array.shape[1]):
    for j in range(original_array.shape[2]):
      if np.sum(original_array[:, i, j]**2)>quantile_threshold:
        sparse_array[:, i, j] = original_array[:, i, j]
  return sparse_array


def run_amp_instance(**dict_params):
    
    wavelet_level = dict_params['wavelet_level']
    band = int(dict_params['band'])
    subband = int(dict_params['subband'])
    prob_quantile = dict_params['prob_quantile']
    n = dict_params['num_measurements']
    B = dict_params['signal_ncol']
    N = dict_params['signal_nrow']
    err_tol = dict_params['err_tol']
    mc = dict_params['mc']
    sparsity_tol = dict_params['sparsity_tol']
    max_iter = dict_params['max_iter']
    err_explosion_tol = dict_params['err_explosion_tol']
    
    iter_count = 0
    
    rng = np.random.default_rng(seed=seed(iter_count, prob_quantile, n, N, B, err_tol, mc, sparsity_tol, wavelet_level, band, subband))
    
    with open('hyperspectral_data/wavelet_decomp/indian_pines_wavelet_decomp.json', 'r') as file:
        wavelet_decomp_dict = json.load(file)
    key_val = f'band_{band}_subband_{subband}'
    wavelet_dat = np.array(wavelet_decomp_dict[key_val])
    wavelet_dat = wavelet_dat[rng.choice(range(wavelet_dat.shape[0]), min(B, wavelet_dat.shape[0]), replace=False), :, :]
    
    sparsified_array = get_sparsified_array(wavelet_dat, prob_quantile)
    wavelet_dat = None
    sparsified_array_flat = torch.flatten(torch.tensor(sparsified_array), 1)
    sparsified_array = None
    signal_true = (np.transpose(sparsified_array_flat)).numpy()
    sparsified_array_flat = None
       
    A = gen_iid_normal_mtx(n, N, rng)/np.sqrt(n)
    Y_true = np.matmul(A, signal_true)
    
    dict_params['signal_nrow'] = N
    dict_params['undersampling_ratio'] = n/N
    dict_params['actual_nonzero_rows'] = np.sum(np.sum(signal_true**2, axis = 1) != 0.)
    dict_params['actual_block_sparsity'] = np.mean(np.sum(signal_true**2, axis = 1) != 0.)
    
    output_df = None
    
    signal_denoised_current = np.zeros((N, B), dtype = float)
    Residual_current = Y_true
    
    dict_observables = recovery_stats(signal_true,
                               signal_denoised_current,
                               sparsity_tol,
                               A,
                               Y_true)
    rel_err = dict_observables['rel_err']
    # rec_stats_dict['iter_count'] = iter_count
    min_rel_err = rel_err
    
    while iter_count<max_iter and rel_err>err_tol and rel_err<err_explosion_tol:
        tick = time.perf_counter()
        
        iter_count = iter_count + 1

        noise_cov_current = np.cov(Residual_current.T)
        
        D, U = np.linalg.eigh(noise_cov_current)
        D = np.round(D, 10)
        
        if np.all(D > 0):
            noise_cov_current_inv = np.matmul(U * 1.0/D, U.T)
            dict_current = amp_iteration_nonsingular(A, Y_true, 
                                                     signal_denoised_current,
                                                     Residual_current, noise_cov_current_inv)
        else:
            nonzero_indices = (D > 0)
            nonzero_indices_int = np.where(nonzero_indices)[0]
            zero_indices_int = np.where(~nonzero_indices)[0]
            D_nonzero_inv = 1/D[nonzero_indices_int]
            dict_current = amp_iteration_singular(A, Y_true, 
                                                  signal_denoised_current, 
                                                  Residual_current, 
                                                  U, nonzero_indices_int, zero_indices_int, D_nonzero_inv)
        
        signal_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        dict_observables = recovery_stats(signal_true,
                                   signal_denoised_current,
                                   sparsity_tol,
                                   A,
                                   Y_true)
        rel_err = dict_observables['rel_err']
        min_rel_err = min(rel_err, min_rel_err)
        tock = time.perf_counter() - tick
        if iter_count % 50 == 0:
            dict_observables['avg_trace_resid_cov'] = np.mean(D)
            dict_observables['min_rel_err'] = min_rel_err
            dict_observables['iter_count'] = iter_count
            dict_observables['time_seconds'] = round(tock, 2)
            combined_dict = {**dict_params, **dict_observables}
            output_df = add_row_to_df(combined_dict, output_df)

    if iter_count % 50 != 0:
        dict_observables['avg_trace_resid_cov'] = np.mean(D)
        dict_observables['min_rel_err'] = min_rel_err
        dict_observables['iter_count'] = iter_count
        dict_observables['time_seconds'] = round(tock, 2)
        combined_dict = {**dict_params, **dict_observables}
        output_df = add_row_to_df(combined_dict, output_df)

    return output_df


def test_experiment() -> dict:
    exp = {'table_name':'amp-test-hyperspectral',
           'params': [{
               'prob_quantile': [0.1],
               'num_measurements': [100],
               'signal_nrow': [5329],
               'signal_ncol': [10],
               'max_iter': [1],
               'wavelet_level': [3],
               'band': [0],
               'subband': [0],
               'err_tol': [1e-5],
               'sparsity_tol': [1e-4],
               'err_explosion_tol': [100],
               'mc': [0],
               'selected_rows_frac': [1.0]
                }]
           }
    return exp

def do_sherlock_experiment(json_file: str):
    exp = read_json(json_file)
    nodes = 400
    with SLURMCluster(queue='donoho,stat,hns,owners,normal',
                      cores=1, memory='4GiB', processes=1,
                      walltime='24:00:00') as cluster:
        cluster.scale(jobs=nodes)
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())
        cluster.scale(0)


def do_coiled_experiment(json_file: str):
    exp = read_json(json_file)
    # logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    software_environment = 'adonoho/amp_matrix_recovery'
    # coiled.delete_software_environment(name=software_environment)
    logging.info('Creating environment.')
    coiled.create_software_environment(
        name=software_environment,
        conda="environment-coiled.yml",
        pip=[
            "git+https://GIT_TOKEN@github.com/adonoho/EMS.git"
        ]
    )
    with coiled.Cluster(software=software_environment,
                        n_workers=960, worker_vm_types=['n1-standard-1'],
                        use_best_zone=True, spot_policy='spot') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def do_local_experiment():
    exp = test_experiment()
    logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=None)
            # do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def read_and_do_local_experiment(json_file: str):
    exp = read_json(json_file)
    with LocalCluster(dashboard_address='localhost:8787', n_workers=4) as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=None)
            # do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def do_test_exp():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_test_experiment(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def do_test():
    exp = test_experiment()
    print(exp)
    pass
    df = run_amp_instance(dict_params = exp)
    df.to_csv("temp.csv")


def count_params(json_file: str):
    exp = read_json(json_file)
    params = unroll_experiment(exp)
    logging.info(f'Count of instances: {len(params)}.')


if __name__ == '__main__':
    do_sherlock_experiment('exp_dicts/hyperspectral_data/indian_pines/AMP_matrix_recovery_JS_hyperspectral_indian_pines_sherlock.json')
    # do_sherlock_experiment('exp_dicts/hyperspectral_data/indian_pines/JS_band_1_subband_0.json')



