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
import amp_iteration as amp
from minimax_tau_threshold import minimax_tau_threshold

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
        
        if np.linalg.det(noise_cov_current) > 0:
            noise_cov_current_inv = np.linalg.inv(noise_cov_current)
            dict_current = amp.amp_iteration_nonsingular(A, Y, signal_denoised_prev, Residual_prev, tau, noise_cov_current_inv)
        else:
            D, U = np.linalg.eigh(noise_cov_current)
            nonzero_indices = (D>0)
            D_nonzero_inv = 1/D[nonzero_indices]
            dict_current = amp.amp_iteration_singular(A, Y, signal_denoised_prev, Residual_prev, tau, D, U, nonzero_indices, D_nonzero_inv)
        
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


def do_coiled_experiment():
    exp = test_experiment()
    logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    software_environment = 'adonoho/matrix_recovery'
    # logging.info('Deleting environment.')
    # coiled.delete_software_environment(software_environment)
    logging.info('Creating environment.')
    coiled.create_software_environment(
        name=software_environment,
        conda="environment-coiled.yml",
        pip=[
            "git+https://GIT_TOKEN@github.com/adonoho/EMS.git"
        ]
    )
    with coiled.Cluster(software=software_environment, n_workers=10) as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())
            # do_on_cluster(exp, block_bp_instance_df, client, project_id='coiled-data@hs-deep-lab-donoho.iam.gserviceaccount.com')


def do_local_experiment():
    exp = test_experiment()
    logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


def read_and_do_local_experiment(json_file: str):
    exp = read_json(json_file)
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
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
    read_and_do_local_experiment('/exp_dicts/AMP_matrix_recovery_blocksoft_03_1.json')
    # count_params('updated_undersampling_int_grids.json')
    # do_coiled_experiment()
    # do_test_exp()
    # do_test()
    # run_block_bp_experiment('block_bp_inputs.json')
