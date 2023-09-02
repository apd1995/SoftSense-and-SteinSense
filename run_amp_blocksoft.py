#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:30:15 2023

@author: apratimdey
"""

import numpy as np
from numpy.random import Generator
import cvxpy as cvx
from pandas import DataFrame
import time
# import amp_iteration as amp
# from minimax_tau_threshold import minimax_tau_threshold

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


# ===== minimax_tau_threshold.py =====
from scipy import integrate
from scipy.stats import chi2
from scipy.optimize import brentq


def h_deno_integrand(x, tau, signal_ncol):
    return  (np.sqrt(x) - tau) * chi2.pdf(x, df = signal_ncol)


def h_deno_integral(tau, signal_ncol):
    return max(1e-10, integrate.quad(h_deno_integrand, tau**2, np.inf, args=(tau, signal_ncol))[0])


def h(tau, signal_ncol):
    numerator = tau
    denominator = h_deno_integral(tau, signal_ncol)
    return numerator/denominator


def h_centered(tau, sparsity, signal_ncol):
    return h(tau, signal_ncol) + 1 - (1/sparsity)


def minimax_tau_threshold(sparsity, signal_ncol):
    #root = newton(h_centered, 0, args=(sparsity, signal_ncol), maxiter = 1000, full_output=True)
    #root = fsolve(func = h_centered, x0 = -2, args = (sparsity, signal_ncol))
    root = brentq(h_centered, 0, 1e+10, args = (sparsity, signal_ncol))
    return root


# ===== amp_iteration.py =====
import autograd.numpy as anp


def block_soft_thresholding_nonsingular_vec(y, tau, Sigma_inv):
    quad_whitening = anp.dot(anp.matmul(Sigma_inv, y), y)
    if quad_whitening > 0:
        return y * max(0, 1 - (tau / anp.sqrt(quad_whitening)))
    else:
        return y


def block_soft_thresholding_nonsingular(X: anp.ndarray, tau: float, Sigma_inv: anp.ndarray) -> anp.ndarray:
    """
    Applies block soft thresholding rowwise to denoise Y.

    Parameters
    ----------
    Y : anp.ndarray
        Noisy signal.
    tau : float
        Threshold for block soft thresholding.
    Sigma_inv : anp.ndarray
        Noise precision matrix.

    Returns
    -------
    None.
    """
    quad_whitening = anp.sum(X * anp.matmul(X, Sigma_inv), axis=1)
    block_soft_thresholding_coeff = anp.where(quad_whitening != 0, 1 - tau / anp.sqrt(quad_whitening), 1)
    block_soft_thresholding_coeff = anp.where(block_soft_thresholding_coeff > 0, block_soft_thresholding_coeff, 0)
    return X * block_soft_thresholding_coeff[:, anp.newaxis]


def block_soft_thresholding_diagonal_vec(y, tau, diag_inv):
    quad_whitening = sum(diag_inv * y ** 2)
    if quad_whitening > 0:
        return y * max(0, 1 - (tau / anp.sqrt(quad_whitening)))
    else:
        return y


def block_soft_thresholding_diagonal(X, tau, diag_inv):
    quad_whitening = anp.sum(X ** 2 * diag_inv, axis=1)
    block_soft_thresholding_coeff = anp.where(quad_whitening != 0, 1 - tau / anp.sqrt(quad_whitening), 1)
    block_soft_thresholding_coeff = anp.where(block_soft_thresholding_coeff > 0, block_soft_thresholding_coeff, 0)
    return X * block_soft_thresholding_coeff[:, anp.newaxis]


def block_soft_thresholding_singular_vec(y, tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    # changing coordinates to get uncorrelated components
    y_indep = anp.matmul(Sigma_eigvecs.T, y)

    y_indep_nonzero = y_indep[nonzero_indices]

    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = block_soft_thresholding_diagonal_vec(y_indep_nonzero, tau, Sigma_nonzero_eigvals_inv)

    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    y_indep_zero = y_indep[zero_indices]
    signal_newbasis_zero = y_indep_zero

    # combine the two to get signal_newbasis
    signal_newbasis = anp.concatenate((signal_newbasis_zero, signal_newbasis_indep))
    # signal_newbasis = anp.zeros(len(y), dtype = float)
    # signal_newbasis[nonzero_indices] = signal_newbasis_indep
    # signal_newbasis[zero_indices] = signal_newbasis_zero

    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = anp.matmul(Sigma_eigvecs, signal_newbasis)

    return signal_originalbasis


def block_soft_thresholding_singular(X, tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    # changing coordinates to get uncorrelated components
    X_indep = anp.matmul(X, Sigma_eigvecs)

    X_indep_nonzero = X_indep[:, nonzero_indices]

    # apply denoiser on these coordinates to estimate the signal in the new basis on indep coordinates
    signal_newbasis_indep = block_soft_thresholding_diagonal(X_indep_nonzero, tau, Sigma_nonzero_eigvals_inv)

    # when D has a 0 entry, it means we have perfect precision
    zero_indices = ~nonzero_indices
    X_indep_zero = X_indep[:, zero_indices]
    signal_newbasis_zero = X_indep_zero

    # combine the two to get signal_newbasis
    signal_newbasis = anp.zeros(X.shape, dtype=float)
    signal_newbasis[:, nonzero_indices] = signal_newbasis_indep
    signal_newbasis[:, zero_indices] = signal_newbasis_zero

    # we have identified U.T @ signal, now we need to get signal i.e revert to original coordinates
    signal_originalbasis = anp.matmul(signal_newbasis, Sigma_eigvecs.T)

    return signal_originalbasis


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)


def update_signal_denoised_nonsingular(signal_noisy_current: float,
                                       tau: float,
                                       noise_cov_current_inv: float):
    return block_soft_thresholding_nonsingular(signal_noisy_current, tau, noise_cov_current_inv)


def update_signal_denoised_singular(signal_noisy_current: float,
                                    tau: float,
                                    noise_cov_current_eigvecs: float,
                                    noise_cov_current_nonzero_indices,
                                    noise_cov_current_nonzero_eigvals_inv: float):
    return block_soft_thresholding_singular(signal_noisy_current, tau, noise_cov_current_eigvecs,
                                            noise_cov_current_nonzero_indices, noise_cov_current_nonzero_eigvals_inv)


from autograd import jacobian

def block_soft_thresholding_onsager_nonsingular(X, Z, tau, Sigma_inv):
    X = X.astype(np.float64)
    jacobian_mat_list = [jacobian(block_soft_thresholding_nonsingular_vec)(X[i, :], tau, Sigma_inv) for i in
                         range(X.shape[0])]
    onsager_list = [anp.matmul(jacobian_mat, Z.T) for jacobian_mat in jacobian_mat_list]
    return sum(onsager_list).T / Z.shape[0]


def block_soft_thresholding_onsager_singular(X, Z, tau, Sigma_eigvecs, nonzero_indices, Sigma_nonzero_eigvals_inv):
    X = X.astype(np.float64)
    jacobian_mat_list = [jacobian(block_soft_thresholding_singular_vec)(X[i, :], tau, Sigma_eigvecs, nonzero_indices,
                                                                        Sigma_nonzero_eigvals_inv) for i in
                         range(X.shape[0])]
    onsager_list = [anp.matmul(jacobian_mat, Z.T) for jacobian_mat in jacobian_mat_list]
    return sum(onsager_list).T / Z.shape[0]


def update_residual_singular(A: float,
                             Y: float,
                             signal_noisy_current: float,
                             signal_denoised_current: float,
                             Residual_prev: float,
                             tau: float,
                             noise_cov_current_eigvecs: float,
                             noise_cov_current_nonzero_indices,
                             noise_cov_current_nonzero_eigvals_inv: float):
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = block_soft_thresholding_onsager_singular(signal_noisy_current, Residual_prev, tau,
                                                             noise_cov_current_eigvecs,
                                                             noise_cov_current_nonzero_indices,
                                                             noise_cov_current_nonzero_eigvals_inv)
    return naive_residual + onsager_term_


def update_residual_nonsingular(A: float,
                                Y: float,
                                signal_noisy_current: float,
                                signal_denoised_current: float,
                                Residual_prev: float,
                                tau: float,
                                noise_cov_current_inv: float):
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = block_soft_thresholding_onsager_nonsingular(signal_noisy_current, Residual_prev, tau,
                                                                noise_cov_current_inv)
    return naive_residual + onsager_term_


def amp_iteration_nonsingular(A: float,
                              Y: float,
                              signal_denoised_prev: float,
                              Residual_prev: float,
                              tau: float,
                              noise_cov_current_inv: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    # noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised_nonsingular(signal_noisy_current, tau, noise_cov_current_inv)
    Residual_current = update_residual_nonsingular(A, Y, signal_noisy_current, signal_denoised_current, Residual_prev,
                                                   tau, noise_cov_current_inv)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


def amp_iteration_singular(A: float,
                           Y: float,
                           signal_denoised_prev: float,
                           Residual_prev: float,
                           tau: float,
                           noise_cov_current_eigvecs: float,
                           noise_cov_current_nonzero_indices,
                           noise_cov_current_nonzero_eigvals_inv: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    # noise_cov_current = Residual_prev.T @ Residual_prev/A.shape[0]
    signal_denoised_current = update_signal_denoised_singular(signal_noisy_current, tau, noise_cov_current_eigvecs,
                                                              noise_cov_current_nonzero_indices,
                                                              noise_cov_current_nonzero_eigvals_inv)
    Residual_current = update_residual_singular(A, Y, signal_noisy_current, signal_denoised_current, Residual_prev, tau,
                                                noise_cov_current_eigvecs, noise_cov_current_nonzero_indices,
                                                noise_cov_current_nonzero_eigvals_inv)
    return {'signal_denoised_current': signal_denoised_current,
            'Residual_current': Residual_current}


def seed(nonzero_rows: float, num_measurements: float, signal_nrow: float, signal_ncol: float, err_tol: float, mc: float, sparsity_tol: float) -> int:
    return round(1 + round(nonzero_rows * 1000) + round(num_measurements * 1000) + round(signal_nrow * 1000) + round(signal_ncol * 1000) + round(err_tol * 100000) + mc * 100000 + round(sparsity_tol * 1000000))


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
    zero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)<=sparsity_tol)

    nonzero_indices_true = (np.apply_along_axis(np.linalg.norm, 1, X_true)!=0)
    nonzero_indices_rec = (np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B)>sparsity_tol)
    
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
                'soft_sparsity': np.mean(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > sparsity_tol),
                'nonzero_rows_rec': np.sum(np.apply_along_axis(np.linalg.norm, 1, X_rec)/np.sqrt(B) > sparsity_tol),
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
    
    rng = np.random.default_rng(seed=seed(k, n, N, B, err_tol, mc, sparsity_tol))

    signal_true = np.zeros((N, B), dtype=float)
    nonzero_indices = rng.choice(range(N), k, replace=False)
    signal_true[nonzero_indices, :] = rng.normal(0, 1, (k, B))
   
    A = gen_iid_normal_mtx(n, N, rng)/np.sqrt(n)
    Y = np.matmul(A, signal_true)
    
    sparsity = k/N
    dict_params['sparsity'] = sparsity
    dict_params['undersampling_ratio'] = n/N
    
    tau = minimax_tau_threshold(sparsity, B)
    
    iter_count = 0
    
    tick = time.perf_counter()
    
    signal_denoised_current = np.zeros((N, B), dtype = float)
    Residual_current = Y
    
    dict_observables = recovery_stats(signal_true,
                               signal_denoised_current,
                               sparsity_tol)
    rel_err = dict_observables['rel_err']
    # rec_stats_dict['iter_count'] = iter_count
    min_rel_err = rel_err
    # rec_stats_dict['min_rel_err'] = min_rel_err
    # dict_observables = rec_stats_dict
    # for key in list(dict_observables):
    #     dict_observables[key] = [dict_observables[key]]
    
    while iter_count<max_iter and rel_err>err_tol and rel_err<err_explosion_tol:
        
        iter_count = iter_count + 1
        
        signal_denoised_prev = signal_denoised_current
        Residual_prev = Residual_current
        noise_cov_current = np.matmul(Residual_prev.T, Residual_prev)/n
        
        D, U = np.linalg.eigh(noise_cov_current)
        D = np.round(D, 10)
        
        if np.all(D > 0):
            noise_cov_current_inv = np.linalg.inv(noise_cov_current)
            dict_current = amp_iteration_nonsingular(A, Y, signal_denoised_prev, Residual_prev, tau, noise_cov_current_inv)
        else:
            nonzero_indices = (D > 0)
            D_nonzero_inv = 1/D[nonzero_indices]
            dict_current = amp_iteration_singular(A, Y, signal_denoised_prev, Residual_prev, tau, U, nonzero_indices, D_nonzero_inv)
        
        signal_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        
        dict_observables = recovery_stats(signal_true,
                                   signal_denoised_current,
                                   sparsity_tol)
        rel_err = dict_observables['rel_err']
        # rec_stats_dict['iter_count'] = iter_count
        min_rel_err = min(rel_err, min_rel_err)
        # rec_stats_dict['running_min_rel_err'] = running_min_rel_err
        # for key in list(dict_observables):
        #     dict_observables[key] = dict_observables[key] + [rec_stats_dict[key]]
    
    tock = time.perf_counter() - tick
    dict_observables['min_rel_err'] = min_rel_err
    dict_observables['iter_count'] = iter_count
    dict_observables['time_seconds'] = round(tock, 2)

    #return DataFrame(data = {**dict_params, **dict_observables}).set_index('iter_count')
    return DataFrame(data = {**dict_params, **dict_observables}, index = [0])


def test_experiment() -> dict:
    exp = {'nonzero_rows': 30,
           'num_measurements': 50,
           'signal_nrow': 100,
           'signal_ncol': 5,
           'max_iter': 50,
           'err_tol': 1e-5,
           'sparsity_tol': 1e-4,
           'err_explosion_tol': 1e+5,
           'mc': 0}
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


def do_coiled_experiment(json_file: str):
    exp = read_json(json_file)
    logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    software_environment = 'adonoho/amp_matrix_recovery'
    logging.info('Creating environment.')
    coiled.create_software_environment(
        name=software_environment,
        conda="environment-coiled.yml",
        pip=[
            "git+https://GIT_TOKEN@github.com/adonoho/EMS.git"
        ]
    )
    with coiled.Cluster(software=software_environment, n_workers=80, worker_vm_types=['n1-standard-1'],
                        use_best_zone=True, compute_purchase_option="spot_with_fallback") as cluster:
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
    with LocalCluster(dashboard_address='localhost:8787', n_workers=12, threads_per_worker=1) as cluster:
    # with LocalCluster(dashboard_address='localhost:8787') as cluster:
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
    # do_local_experiment()
    # read_and_do_local_experiment('exp_dicts/AMP_matrix_recovery_blocksoft_09.json')
    # count_params('updated_undersampling_int_grids.json')
    do_coiled_experiment('exp_dicts/AMP_matrix_recovery_blocksoft_09.json')
    # do_test_exp()
    # do_test()
    # run_block_bp_experiment('block_bp_inputs.json')
