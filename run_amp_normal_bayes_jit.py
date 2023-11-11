#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:58:34 2023

@author: apratimdey
"""

import numpy as np
from numpy.random import Generator
import cvxpy as cvx
from pandas import DataFrame, concat
import time

from EMS.manager import do_on_cluster, get_gbq_credentials, do_test_experiment, read_json, unroll_experiment
from dask.distributed import Client, LocalCluster
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
from jax.scipy.stats import multivariate_normal
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
def normal_bayes_vec(y: np.ndarray,
                     signal_mean_vec: np.ndarray,
                     signal_cov: np.ndarray,
                     noise_cov: np.ndarray,
                     sparsity: float) -> np.ndarray:
    """
    Performs bayes denoising given noisy y = X + Z where X and Z are gaussian vectors with noise_cov non-singular.

    Parameters
    ----------
    y : np.ndarray
        noisy signal.
    signal_mean_vec : np.ndarray
        mean of signal X.
    signal_cov : np.ndarray
        cov of signal X.
    noise_cov : np.ndarray
        cov of noise Z.
    sparsity : float
        X is assumed to be gaussian w.p. sparsity, and else entirely zero.

    Returns
    -------
    np.ndarray
        Returns E(X | X + Z = y).

    """
    nonzero_bayes = signal_mean_vec + jnp.matmul(signal_cov, jnp.matmul(jnp.linalg.inv(signal_cov + noise_cov), y - signal_mean_vec))
    num = sparsity*multivariate_normal.pdf(y, mean = signal_mean_vec, cov = signal_cov + noise_cov)
    deno = num + (1-sparsity)*multivariate_normal.pdf(y, mean = jnp.zeros_like(signal_mean_vec), cov = noise_cov)
    conditional_nonzero_prob = jax.lax.cond(deno > 0,
                                            lambda _: num/deno,
                                            lambda _: 1.,
                                            None)
    return conditional_nonzero_prob*nonzero_bayes

@jax.jit
def normal_bayes(X: np.ndarray,
                 signal_mean_vec: np.ndarray,
                 signal_cov: np.ndarray,
                 noise_cov: np.ndarray,
                 sparsity: float) -> np.ndarray:
    """
    Applies bayes denoiser to each row of signal matrix when noise_cov is non-singular.

    Parameters
    ----------
    X : np.ndarray
        Noisy signal along whose rows denoiser is to be applied.
    signal_mean_vec : np.ndarray
        mean of signal rows.
    signal_cov : np.ndarray
        cov of signal rows.
    noise_cov : np.ndarray
        cov of noise. Needs to be nonsingular.
    sparsity : float
        sparsity level.

    Returns
    -------
    np.ndarray
        Each row of X denoised by Bayes denoiser.

    """
    normal_bayes_vectorized = jax.vmap(normal_bayes_vec, in_axes = (0, None, None, None, None))
    return normal_bayes_vectorized(X, signal_mean_vec, signal_cov, noise_cov, sparsity)


@jax.jit
def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + jnp.matmul(A.T, Residual_prev)


@jax.jit
def update_signal_denoised(signal_noisy_current: np.ndarray,
                           signal_mean_vec: np.ndarray,
                           signal_cov: np.ndarray,
                           noise_cov_current: np.ndarray,
                           sparsity: float):
    """
    Applies denoiser to noisy signal.

    Parameters
    ----------
    signal_noisy_current : np.ndarray
        Noisy signal at current iteration.
    signal_mean_vec : np.ndarray
        Mean of signal row.
    signal_cov : np.ndarray
        Cov of signal row.
    noise_cov_current : np.ndarray
        Current noise cov estimate.
    sparsity : float
        fraction of nonzero rows in signal.

    Returns
    -------
    np.ndarray
        Returns denoised signal matrix.

    """
    return normal_bayes(X = signal_noisy_current,
                        signal_mean_vec = signal_mean_vec,
                        signal_cov = signal_cov,
                        noise_cov = noise_cov_current,
                        sparsity = sparsity)

@jax.jit
def normal_bayes_onsager(X: np.ndarray,
                        Z: np.ndarray,
                        signal_mean_vec: np.ndarray,
                        signal_cov: np.ndarray,
                        noise_cov: np.ndarray,
                        sparsity: float) -> np.ndarray:
    """
    Computes Onsager term.

    Parameters
    ----------
    X : np.ndarray
        noisy signal matrix.
    Z : np.ndarray
        residual matrix.
    signal_mean_vec : np.ndarray
        mean of signal rows.
    signal_cov : np.ndarray
        cov of signal row.
    noise_cov : np.ndarray
        cov of noise.
    sparsity : float
        fraction of nonzero rows in signal matrix.

    Returns
    -------
    np.ndarray
        Onsager term.

    """
    X = jnp.array(X)
    dd_jacobian = jax.jacfwd(normal_bayes_vec, argnums=0)
    # dd_jacobian = jax.jit(jax.jacfwd(james_stein_nonsingular_vec, argnums=0))
    jac_vectorized = jax.vmap(dd_jacobian, in_axes = (0, None, None, None, None))
    sum_jacobians = jac_vectorized(X, signal_mean_vec, signal_cov, noise_cov, sparsity).sum(axis = 0)
    onsager_term = jnp.matmul(Z, sum_jacobians.T)
    return onsager_term / Z.shape[0]


@jax.jit
def update_residual(A: np.ndarray,
                    Y: np.ndarray,
                    signal_noisy_current: np.ndarray,
                    signal_denoised_current: np.ndarray,
                    Residual_prev: np.ndarray,
                    signal_mean_vec: np.ndarray,
                    signal_cov: np.ndarray,
                    noise_cov_current: np.ndarray,
                    sparsity: float) -> np.ndarray:
    naive_residual = Y - jnp.matmul(A, signal_denoised_current)
    onsager_term_ = normal_bayes_onsager(signal_noisy_current,
                                         Residual_prev,
                                         signal_mean_vec,
                                         signal_cov,
                                         noise_cov_current,
                                         sparsity)
    return naive_residual + onsager_term_


@jax.jit
def amp_iteration(A: np.ndarray,
                Y: np.ndarray,
                signal_denoised_prev: np.ndarray,
                Residual_prev: np.ndarray,
                signal_mean_vec: np.ndarray,
                signal_cov: np.ndarray,
                noise_cov_current: np.ndarray,
                sparsity: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    signal_denoised_current = update_signal_denoised(signal_noisy_current,
                                                     signal_mean_vec,
                                                     signal_cov,
                                                     noise_cov_current,
                                                     sparsity)
    Residual_current = update_residual(A,
                                       Y,
                                       signal_noisy_current,
                                       signal_denoised_current,
                                       Residual_prev,
                                       signal_mean_vec,
                                       signal_cov,
                                       noise_cov_current,
                                       sparsity)
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
    signal_true[nonzero_indices, :] = rng.normal(0, 1, (k, B))
    signal_mean_vec = np.zeros(B, dtype = float)
    signal_cov = np.eye(B, dtype = float)
   
    A = gen_iid_normal_mtx(n, N, rng)/np.sqrt(n)
    Y_true = np.matmul(A, signal_true)
    
    sparsity = k/N
    dict_params['sparsity'] = sparsity
    dict_params['undersampling_ratio'] = n/N
    
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
        
        dict_current = amp_iteration(A,
                                     Y_true, 
                                     signal_denoised_current,
                                     Residual_current,
                                     signal_mean_vec,
                                     signal_cov,
                                     noise_cov_current,
                                     sparsity)
        
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
            dict_observables['avg_trace_resid_cov'] = np.trace(noise_cov_current)/B
            dict_observables['min_rel_err'] = min_rel_err
            dict_observables['iter_count'] = iter_count
            dict_observables['time_seconds'] = round(tock, 2)
            combined_dict = {**dict_params, **dict_observables}
            output_df = add_row_to_df(combined_dict, output_df)

    if iter_count % 50 != 0:
        dict_observables['avg_trace_resid_cov'] = np.trace(noise_cov_current)/B
        dict_observables['min_rel_err'] = min_rel_err
        dict_observables['iter_count'] = iter_count
        dict_observables['time_seconds'] = round(tock, 2)
        combined_dict = {**dict_params, **dict_observables}
        output_df = add_row_to_df(combined_dict, output_df)

    return output_df


def test_experiment() -> dict:
    exp = {'nonzero_rows': 200,
           'num_measurements': 900,
           'signal_nrow': 1000,
           'signal_ncol': 5,
           'max_iter': 500,
           'err_tol': 1e-5,
           'sparsity_tol': 1e-4,
           'err_explosion_tol': 100,
           'mc': 9}
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
# dict_params = test_experiment()

def do_coiled_experiment(json_file: str):
    exp = read_json(json_file)
    # logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    software_environment = 'adonoho/amp_matrix_recovery'
    logging.info('Creating environment.')
    coiled.create_software_environment(
        name=software_environment,
        conda="environment-coiled.yml",
        pip=[
            "git+https://GIT_TOKEN@github.com/adonoho/EMS.git"
        ]
    )
    with coiled.Cluster(software=software_environment,
                        n_workers=320, worker_vm_types=['n1-standard-1'],
                        use_best_zone=True, spot_policy='spot') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def do_local_experiment():
    exp = test_experiment()
    logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def read_and_do_local_experiment(json_file: str):
    exp = read_json(json_file)
    with LocalCluster(dashboard_address='localhost:8787', n_workers=32) as cluster:
    # with LocalCluster(dashboard_address='localhost:8787') as cluster:
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
    read_and_do_local_experiment('exp_dicts/AMP_matrix_recovery_normal_bayes_jit.json')
    # count_params('updated_undersampling_int_grids.json')
    # do_coiled_experiment('exp_dicts/AMP_matrix_recovery_normal_bayes_approx_jacobian_2.json')
    # do_test_exp()
    # do_test()
    # run_block_bp_experiment('block_bp_inputs.json')
