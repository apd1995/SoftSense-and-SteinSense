#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:58:34 2023

@author: apratimdey
"""

import numpy as np
from numpy.random import Generator
import cvxpy as cvx
from pandas import DataFrame
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
import autograd.numpy as anp
from autograd import jacobian


def seed(iter_count: int,
         nonzero_rows: float,
         num_measurements: float,
         signal_nrow: float,
         signal_ncol: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float) -> int:
    return round(1 + round(iter_count*1000) + round(nonzero_rows * 1000) + round(num_measurements * 1000) + round(signal_nrow * 1000) + round(signal_ncol * 1000) + round(err_tol * 100000) + round(mc * 100000) + round(sparsity_tol * 1000000))


def multivariate_normal_pdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Returns the multivariate normal pdf at y, provided some checks are satisfied.
    If condition number is large, returns 0.
    If condition number is ok but somehow the quadratic form in gaussian exponent is positive, returns 0.

    Parameters
    ----------
    y : np.ndarray
        Array at which normal pdf is to be calculated.
    mean : np.ndarray
        Mean of normal distribution.
    cov : np.ndarray
        Covariance of normal distribution.
    cov_condition : float
        Condition number of covariance.

    Returns
    -------
    float
        DESCRIPTION.

    """
    # cov_eigvals = np.linalg.eigh(cov)[0]
    # cov_eigvals_sorted = np.sort(np.abs(cov_eigvals))
    # cov_condition = cov_eigvals_sorted[-1]/cov_eigvals_sorted[0]
    if anp.linalg.det(cov) <= 0 or anp.linalg.det(2*anp.pi*cov) <= 0 or anp.sqrt(anp.linalg.det(2*anp.pi*cov)) == 0:
        return 0
    else:
        exponent = -anp.matmul(y - mean, anp.matmul(anp.linalg.inv(cov), y - mean))/2
        if exponent > 0:
            return 0
        else:
            return anp.exp(exponent)/anp.sqrt(anp.linalg.det(2*anp.pi*cov))


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
    nonzero_bayes = signal_mean_vec + anp.matmul(signal_cov, anp.matmul(anp.linalg.inv(signal_cov + noise_cov), y - signal_mean_vec))
    num = sparsity*multivariate_normal_pdf(y, mean = signal_mean_vec, cov = signal_cov + noise_cov)
    deno = num + (1-sparsity)*multivariate_normal_pdf(y, mean = anp.zeros_like(signal_mean_vec), cov = noise_cov)
    if deno > 0:
        conditional_nonzero_prob = num/deno
        return conditional_nonzero_prob * nonzero_bayes
    else:
        return nonzero_bayes


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
    return np.apply_along_axis(normal_bayes_vec, 1, X, signal_mean_vec = signal_mean_vec, signal_cov = signal_cov, noise_cov = noise_cov, sparsity = sparsity)


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)


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


def normal_bayes_onsager(X: np.ndarray,
                        Z: np.ndarray,
                        signal_mean_vec: np.ndarray,
                        signal_cov: np.ndarray,
                        noise_cov: np.ndarray,
                        sparsity: float,
                        rng: Generator,
                        selected_rows_frac: float) -> np.ndarray:
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
    X = X.astype(anp.float64)
    selected_rows = rng.choice(X.shape[0], int(selected_rows_frac*X.shape[0]), replace = False)
    jacobian_mat_list = [jacobian(normal_bayes_vec)(X[i,:], signal_mean_vec, signal_cov, noise_cov, sparsity) for i in selected_rows]
    onsager_list = [anp.matmul(jacobian_mat, Z.T) for jacobian_mat in jacobian_mat_list]
    return sum(onsager_list).T/(Z.shape[0]*selected_rows_frac)


def update_residual(A: np.ndarray,
                    Y: np.ndarray,
                    signal_noisy_current: np.ndarray,
                    signal_denoised_current: np.ndarray,
                    Residual_prev: np.ndarray,
                    signal_mean_vec: np.ndarray,
                    signal_cov: np.ndarray,
                    noise_cov_current: np.ndarray,
                    sparsity: float,
                    rng: Generator,
                    selected_rows_frac: float) -> np.ndarray:
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = normal_bayes_onsager(signal_noisy_current,
                                         Residual_prev,
                                         signal_mean_vec,
                                         signal_cov,
                                         noise_cov_current,
                                         sparsity,
                                         rng,
                                         selected_rows_frac)
    return naive_residual + onsager_term_


def amp_iteration(A: np.ndarray,
                Y: np.ndarray,
                signal_denoised_prev: np.ndarray,
                Residual_prev: np.ndarray,
                signal_mean_vec: np.ndarray,
                signal_cov: np.ndarray,
                noise_cov_current: np.ndarray,
                sparsity: float,
                rng: Generator,
                selected_rows_frac):
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
                                       sparsity,
                                       rng,
                                       selected_rows_frac)
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
              sparsity_tol: float):
    
    N, B = X_true.shape

    zero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)==0)
    zero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)<=10*sparsity_tol)

    nonzero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)!=0)
    nonzero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)>10*sparsity_tol)
    
    dict_observables = {
                'rel_err': cvx.norm(X_true-X_rec, "fro").value/cvx.norm(X_true, "fro").value,
                'avg_err': cvx.norm(X_true - X_rec, "fro").value/np.sqrt(N*B),
                'max_row_err': cvx.mixed_norm(X_true - X_rec, 2, np.inf).value/np.sqrt(B),
                'norm_2_1_true': cvx.mixed_norm(X_true, 2, 1).value/(N*np.sqrt(B)),
                'norm_2_1_rec': cvx.mixed_norm(X_rec, 2, 1).value/(N*np.sqrt(B)),
                'norm_2_2_true': cvx.mixed_norm(X_true, 2, 2).value/np.sqrt(N*B),
                'norm_2_2_rec': cvx.mixed_norm(X_rec, 2, 2).value/np.sqrt(N*B),
                'norm_2_infty_true': cvx.mixed_norm(X_true, 2, np.inf).value/np.sqrt(B),
                'norm_2_infty_rec': cvx.mixed_norm(X_rec, 2, np.inf).value/np.sqrt(B),
                'soft_sparsity': np.mean(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > 10*sparsity_tol),
                'nonzero_rows_rec': np.sum(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > 10*sparsity_tol),
                'tpr': sum(zero_indices_true * zero_indices_rec)/max(1, sum(zero_indices_true)),
                'tnr': sum(nonzero_indices_true * nonzero_indices_rec)/max(1, sum(nonzero_indices_true))
                }
    
    return dict_observables


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
    selected_rows_frac = dict_params['selected_rows_frac']
        
    iter_count = 0
    
    rng = np.random.default_rng(seed=seed(iter_count, k, n, N, B, err_tol, mc, sparsity_tol))

    signal_true = np.zeros((N, B), dtype=float)
    nonzero_indices = rng.choice(range(N), k, replace=False)
    signal_true[nonzero_indices, :] = rng.normal(0, 1, (k, B))
    signal_mean_vec = np.zeros(B, dtype = float)
    signal_cov = np.eye(B, dtype = float)
   
    A = gen_iid_normal_mtx(n, N, rng)/np.sqrt(n)
    Y = np.matmul(A, signal_true)
    
    sparsity = k/N
    dict_params['sparsity'] = sparsity
    dict_params['undersampling_ratio'] = n/N
    
    tick = time.perf_counter()
    
    signal_denoised_current = np.zeros((N, B), dtype = float)
    Residual_current = Y
    
    dict_observables = recovery_stats(signal_true,
                               signal_denoised_current,
                               sparsity_tol)
    rel_err = dict_observables['rel_err']
    # avg_err = dict_observables['avg_err']
    min_rel_err = rel_err
    noise_cov_current = np.matmul(Residual_current.T, Residual_current)/n
    
    while iter_count<max_iter and rel_err>100*err_tol and rel_err<err_explosion_tol:
        
        iter_count = iter_count + 1
        
        rng = np.random.default_rng(seed=seed(iter_count, k, n, N, B, err_tol, mc, sparsity_tol))
        
        signal_denoised_prev = signal_denoised_current
        Residual_prev = Residual_current
        noise_cov_current = np.matmul(Residual_prev.T, Residual_prev)/n
        dict_current = amp_iteration(A,
                                     Y, 
                                     signal_denoised_prev,
                                     Residual_prev,
                                     signal_mean_vec,
                                     signal_cov,
                                     noise_cov_current,
                                     sparsity,
                                     rng,
                                     selected_rows_frac)

        signal_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        
        dict_observables = recovery_stats(signal_true,
                                   signal_denoised_current,
                                   sparsity_tol)
        rel_err = dict_observables['rel_err']
        min_rel_err = min(rel_err, min_rel_err)

    tock = time.perf_counter() - tick
    dict_observables['min_rel_err'] = min_rel_err
    dict_observables['iter_count'] = iter_count
    dict_observables['time_seconds'] = round(tock, 2)

    return DataFrame(data = {**dict_params, **dict_observables}, index = [0])


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
    with LocalCluster(dashboard_address='localhost:8787', n_workers=32, threads_per_worker=1) as cluster:
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
    # read_and_do_local_experiment('exp_dicts/AMP_matrix_recovery_normal_bayes_approx_jacobian_2.json')
    # count_params('updated_undersampling_int_grids.json')
    do_coiled_experiment('exp_dicts/AMP_matrix_recovery_normal_bayes_approx_jacobian_2.json')
    # do_test_exp()
    # do_test()
    # run_block_bp_experiment('block_bp_inputs.json')
