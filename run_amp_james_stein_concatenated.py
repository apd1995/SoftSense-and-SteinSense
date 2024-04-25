#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:46:42 2024

@author: apratimdey
"""

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
logging.getLogger('jax').setLevel(logging.ERROR)


def seed(block_idx: int,
         num_nonzero_blocks: float,
         num_measurements: float,
         num_blocks: float,
         block_size: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float) -> int:
    return round(1 + round(block_idx*1000) + round(num_nonzero_blocks * 1000) + round(num_measurements * 1000) + round(num_blocks * 1000) + round(block_size * 1000) + round(err_tol * 100000) + round(mc * 100000) + round(sparsity_tol * 1000000))


def james_stein_block(y, sigma_sq):
    d = len(y)
    quad_whitening = np.sum(y**2)/sigma_sq
    if quad_whitening > (d-2):
        return y * (1 - ((d-2)/quad_whitening))
    else:
        return np.zeros_like(y, dtype = float)
    # return jax.lax.cond(quad_whitening > (d-2),
    #                 lambda y: y * (1 - ((d-2)/quad_whitening)),   # True branch (lambda function)
    #                 lambda y: jnp.zeros(d),  # False branch (lambda function)
    #                 y)  # Operand to pass to selected branch
    

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


def update_signal_noisy(A: float,
                        signal_denoised_prev: float,
                        Residual_prev: float):
    return signal_denoised_prev + np.matmul(A.T, Residual_prev)


def update_signal_denoised(signal_noisy_current: float,
                           num_blocks,
                           block_size,
                           noise_var_current: float):
    return james_stein_full_vec(signal_noisy_current, num_blocks, block_size, noise_var_current)


def james_stein_div(X, M, B, noise_var, n):
    norm_sq_vals = np.zeros(M, dtype = float)
    for block_idx in range(M):
        norm_sq_vals[block_idx] = np.sum(X[(block_idx*B):((block_idx + 1)*B)]**2)/noise_var
    coeffs = np.where(norm_sq_vals>(B-2), (B - ((B-2)**2/norm_sq_vals)), 0.)
    return np.sum(coeffs)/n


def james_stein_onsager(X, M, B, noise_var, Z, n):
    X = np.array(X)
    onsager_term = james_stein_div(X, M, B, noise_var, n)*Z
    return onsager_term
    

def update_residual(A: float,
                    Y: float,
                    signal_noisy_current: float,
                    signal_denoised_current: float,
                    Residual_prev: float,
                    num_blocks,
                    block_size,
                    noise_var_current: float):
    naive_residual = Y - np.matmul(A, signal_denoised_current)
    onsager_term_ = james_stein_onsager(signal_noisy_current,
                                        num_blocks,
                                        block_size,
                                        noise_var_current,
                                        Residual_prev,
                                        len(Y))
    return naive_residual + onsager_term_


def amp_iteration(A: float,
                Y: float,
                signal_denoised_prev: float,
                Residual_prev: float,
                num_blocks: int,
                block_size: int,
                noise_var_current: float):
    signal_noisy_current = update_signal_noisy(A, signal_denoised_prev, Residual_prev)
    signal_denoised_current = update_signal_denoised(signal_noisy_current,
                                                     num_blocks,
                                                     block_size,
                                                     noise_var_current)
    Residual_current = update_residual(A,
                                    Y,
                                    signal_noisy_current,
                                    signal_denoised_current,
                                    Residual_prev,
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
    
    k = dict_params['num_nonzero_blocks']
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
    
    rng = np.random.default_rng(seed=seed(2*N, k, n, M, B, err_tol, mc, sparsity_tol))
    signal_true = np.zeros(N, dtype=float)
    nonzero_block_indices = rng.choice(range(M), k, replace=False)
    for nonzero_block_idx in nonzero_block_indices:
        rng_block = np.random.default_rng(seed=seed(nonzero_block_idx, k, n, M, B, err_tol, mc, sparsity_tol))
        signal_true[(nonzero_block_idx*B):((nonzero_block_idx+1)*B)] = rng_block.normal(0, 1, B)
    signal_true = np.array(signal_true)
   
    A = gen_iid_normal_mtx(n, M, B, rng)/np.sqrt(n)
    Y_true = np.matmul(A, signal_true)
    
    block_sparsity = k/M
    dict_params['block_sparsity'] = block_sparsity
    dict_params['undersampling_ratio'] = n/N
    
    output_df = None
    
    iter_count = 0
    
    signal_denoised_current = np.zeros(N, dtype = float)
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

        noise_var_current = np.var(Residual_current)

        dict_current = amp_iteration(A, Y_true, 
                                    signal_denoised_current,
                                    Residual_current, 
                                    M,
                                    B,
                                    noise_var_current)
        
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
            dict_observables['min_rel_err'] = min_rel_err
            dict_observables['iter_count'] = iter_count
            dict_observables['time_seconds'] = round(tock, 2)
            combined_dict = {**dict_params, **dict_observables}
            output_df = add_row_to_df(combined_dict, output_df)

    if iter_count % 50 != 0:
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
    do_sherlock_experiment('exp_dicts/AMP_matrix_recovery_JS_normal_concatenated_daveUPenn.json')
    # do_test_exp()
    # do_test()
    # run_block_bp_experiment('block_bp_inputs.json')



