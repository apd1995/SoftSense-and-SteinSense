#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 02:10:26 2023

@author: apratimdey
"""

import numpy as np
from numpy.random import Generator
import cvxpy as cvx
from pandas import DataFrame, concat
import time
# import amp_iteration as amp
# from minimax_tau_threshold import minimax_tau_threshold

from EMS.manager import do_on_cluster, get_gbq_credentials, do_test_experiment, read_json, unroll_experiment
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import dask
import coiled
import logging
import json

logging.basicConfig(level=logging.INFO)
log_gbq = logging.getLogger('pandas_gbq')
log_gbq.setLevel(logging.DEBUG)
log_gbq.addHandler(logging.StreamHandler())
import jax
import jax.numpy as jnp
logging.getLogger('jax').setLevel(logging.ERROR)


def seed(iter_count: int,
         nonzero_rows: float,
         num_measurements: float,
         signal_nrow: float,
         signal_ncol: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float) -> int:
    return round(1 + round(iter_count*1000) + round(nonzero_rows * 1000) + round(num_measurements * 1000) + round(signal_nrow * 1000) + round(signal_ncol * 1000) + round(err_tol * 100000) + round(mc * 100000) + round(sparsity_tol * 1000000))


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
    d = X.shape[1]
    quad_whitening = np.sum(X * np.matmul(X, Sigma_inv), axis=1)
    james_stein_coeff = np.where(quad_whitening > (d-2), 1 - ((d-2)/quad_whitening), 0.0)
    return X * james_stein_coeff[:, np.newaxis]


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
    d = X.shape[1]
    quad_whitening = np.sum(X**2 * diag_inv, axis=1)
    james_stein_coeff = np.where(quad_whitening > (d-2), 1 - ((d-2)/quad_whitening), 0.0)
    return X * james_stein_coeff[:, np.newaxis]


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
    # changing coordinates to get uncorrelated components
    X_indep = np.matmul(X, Sigma_eigvecs)
    
    X_indep_nonzero =  X_indep[:, nonzero_indices_int]
    
    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = james_stein_diagonal(X_indep_nonzero, Sigma_nonzero_eigvals_inv)
    
    # when D has a 0 entry, it means we have perfect precision
    # zero_indices = ~nonzero_indices
    X_indep_zero = X_indep[:, zero_indices_int]
    signal_newbasis_zero = X_indep_zero
    
    # combine the two to get signal_newbasis
    signal_newbasis = np.zeros(X.shape, dtype = float)
    signal_newbasis[:, nonzero_indices_int] = signal_newbasis_indep
    signal_newbasis[:, zero_indices_int] = signal_newbasis_zero
    
    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = np.matmul(signal_newbasis, Sigma_eigvecs.T)
    
    return signal_originalbasis


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


# def warm_up():
#     y = np.ones(5, dtype = float)
#     Sigma_inv = np.eye(5, dtype = float)
#     res1 = james_stein_nonsingular_vec(y, Sigma_inv)
#     res2 = james_stein_diagonal_vec(y, np.ones(5, dtype = float))
#     res3 = james_stein_singular_vec(y, Sigma_inv, np.array([0,1,2]), np.array([3,4]), np.ones(3, dtype = float))
#     res4 = james_stein_onsager_nonsingular(np.ones((1000,5), dtype = float), Sigma_inv, Sigma_inv, np.array([0]), 1.0)
#     res5 = james_stein_onsager_singular(np.ones((1000,5), dtype = float), Sigma_inv, Sigma_inv, np.array([0,1,2]), np.array([3,4]), np.ones(3, dtype = float), np.array([0]), 1.0)
#     return True
    

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


# def warm_up_2():
#     res1 = amp_iteration_nonsingular(np.eye(1000), np.eye(1000), 2*np.eye(1000), np.eye(1000), np.eye(1000), np.array([0]), 1.0)
#     res2 = amp_iteration_singular(np.eye(3), np.eye(3), 2*np.eye(3), np.eye(3), np.eye(3), np.arange(2), np.array([2]), np.ones(2, dtype = float), np.array([0]), 1.0)
#     return True


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


def run_amp_instance(**dict_params):
    
    k = dict_params['nonzero_rows']
    n = dict_params['num_measurements']
    N = dict_params['signal_nrow']
    B = dict_params['signal_ncol']
    err_tol = dict_params['err_tol']
    mc = dict_params['mc']
    sparsity_tol = dict_params['sparsity_tol']
    max_iter = dict_params['max_iter']
    err_explosion_tol = dict_params['err_explosion_tol']
    
    iter_count = 0
    
    rng = np.random.default_rng(seed=seed(iter_count, k, n, N, B, err_tol, mc, sparsity_tol))
    signal_true = np.zeros((N, B), dtype=float)
    nonzero_indices = rng.choice(range(N), k, replace=False)
    # signal_true[nonzero_indices, :] = rng.normal(0, 1, (k, B))
    # signal_true[nonzero_indices, :] = rng.poisson(2, (k, B))
    signal_true[nonzero_indices, :] = rng.poisson(2, (k, B))
    signal_true = np.array(signal_true)
   
    A = gen_iid_normal_mtx(n, N, rng)/np.sqrt(n)
    Y_true = np.matmul(A, signal_true)
    
    sparsity = k/N
    dict_params['sparsity'] = sparsity
    dict_params['undersampling_ratio'] = n/N
    
    output_df = None
    
    iter_count = 0
    
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

        # signal_denoised_prev = signal_denoised_current
        # signal_denoised_current = None
        # Residual_prev = Residual_current
        # Residual_current = None
        # noise_cov_current = np.matmul(Residual_prev.T, Residual_prev)/n
        noise_cov_current = np.cov(Residual_current.T)
        # noise_cov_current_diag = np.diag(np.var(Residual_prev, axis = 0))
        
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

    #return DataFrame(data = {**dict_params, **dict_observables}).set_index('iter_count')
    return output_df


def test_experiment() -> dict:
    exp = {'table_name':'amp-test',
           'params': [{
               'nonzero_rows': [30],
               'num_measurements': [320],
               'signal_nrow': [1000],
               'signal_ncol': [5],
               'max_iter': [1],
               'err_tol': [1e-5],
               'sparsity_tol': [1e-4],
               'err_explosion_tol': [100],
               'mc': [0],
               'selected_rows_frac': [1.0]
                }]
           }
    # exp = dict(table_name='amp-integer-grids',
    #             base_index=0,
    #             db_url='sqlite:///data/EMS.db3',
    #             multi_res=[{
    #                 'nonzero_rows': list(range(1, 100)),
    #                 'num_measurements': list(range(1, 100)),
    #                 'signal_nrow': [100],
    #                 'signal_ncol': [1, 2, 3, 4, 5],
    #                 'mc': list(range(100)),
    #                 'err_tol': [1e-5],
    #                 'sparsity_tol': [1e-4]
    #             }])
    # exp = dict(table_name='comr-N50-larger-grids',
    #             base_index=0,
    #             db_url='sqlite:///data/EMS.db3',
    #             multi_res=[{
    #                 'nonzero_rows': list(range(1, 50)),
    #                 'num_measurements': list(range(1, 50)),
    #                 'signal_nrow': [50],
    #                 'signal_ncol': [1, 2, 3, 4, 5],
    #                 'mc': list(range(100)),
    #                 'err_tol': [1e-5],
    #                 'sparsity_tol': [1e-4]
    #             }])
    return exp

def do_sherlock_experiment(json_file: str):
    exp = read_json(json_file)
    nodes = 1000
    with SLURMCluster(queue='normal,owners,donoho,hns,stat',
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
            # do_on_cluster(exp, run_amp_instance, client, credentials=None)
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def read_and_do_local_experiment(json_file: str):
    exp = read_json(json_file)
    with LocalCluster(dashboard_address='localhost:8787', n_workers=32) as cluster:
        with Client(cluster) as client:
            # do_on_cluster(exp, run_amp_instance, client, credentials=None)
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


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
    # do_local_experiment()
    # read_and_do_local_experiment('exp_dicts/AMP_matrix_recovery_JS_poisson_jit.json')
    # count_params('updated_undersampling_int_grids.json')
    # do_coiled_experiment('exp_dicts/AMP_matrix_recovery_JS_poisson_jit.json')
    do_sherlock_experiment('exp_dicts/AMP_matrix_recovery_JS_binary_jit_sherlock.json')
    # do_test_exp()
    # do_test()
    # run_block_bp_experiment('block_bp_inputs.json')



