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

# calculating the minimax threshold

from scipy import special, integrate, optimize

SQRT2PI = np.sqrt(2.0 * np.pi)

def phi(z):
    """Standard normal density."""
    return np.exp(-0.5 * z*z) / SQRT2PI

def Q(t):
    """Upper tail of N(0,1): Q(t) = P(Z > t)."""
    return 0.5 * special.erfc(t / np.sqrt(2.0))

def chi2_pdf(s, df):
    """PDF of chi^2_df at s."""
    # df > 0, s >= 0
    if s < 0:
        return 0.0
    c = 1.0 / (np.power(2.0, df/2.0) * special.gamma(df/2.0))
    return c * (s**(df/2.0 - 1.0)) * np.exp(-s/2.0)

def A_tau(tau):
    """
    A(tau) = E[(|Z| - tau)_+^2] in closed form, Z ~ N(0,1).
    """
    phi_tau = phi(tau)
    Q_tau = Q(tau)
    return 2.0 * ((1.0 + tau*tau) * Q_tau - tau * phi_tau)

def A_tau_prime(tau):
    """
    A'(tau) = d/dtau A(tau) = 4(tau * Q(tau) - phi(tau)).
    """
    phi_tau = phi(tau)
    Q_tau = Q(tau)
    return 4.0 * (tau * Q_tau - phi_tau)


def build_M_functions_continuous(eps1, eps2, B):
    """
    Build M(tau), M'(tau), and a solver that integrates over S ~ chi^2_{B-1}
    using continuous numerical integration (no discrete sum approximation for S).
    """
    if not (0.0 < eps1 < 1.0 and 0.0 < eps2 < 1.0):
        raise ValueError("eps1, eps2 must be in (0,1)")
    if B <= 2:
        raise ValueError("B must be > 2")

    df = B - 1.0
    a = B - 2.0

    # ---- conditional expectations over Z, given S = s ----

    def Ez_sq_term(s, tau):
        """
        For fixed s and tau, compute:
            E_z [ (eta(Z, s) - tau)_+^2 ],
        where Z ~ N(0,1) and eta(z,s) = z * (1 - a/(z^2 + s))_+ for z >= 0,
        integrated over z ≥ 0 and doubled for symmetry.
        """

        def integrand_pos(z):
            z2 = z*z
            shrink = 1.0 - a/(z2 + s)
            if shrink <= 0.0:
                return 0.0
            u = z * shrink
            w = u - tau
            if w <= 0.0:
                return 0.0
            return 2.0 * (w*w) * phi(z)  # factor 2 for ±z

        val, _ = integrate.quad(
            integrand_pos,
            0.0, np.inf,
            epsabs=1e-10, epsrel=1e-8, limit=200
        )
        return val

    def Ez_term_deriv(s, tau):
        """
        For fixed s and tau, compute derivative wrt tau:
            d/dtau E_z [ (eta(Z, s) - tau)_+^2 ]
          = -2 * E_z [ (eta(Z, s) - tau)_+ ].

        Implemented as:
            -4 ∫_0^∞ (eta(z,s) - tau)_+ phi(z) dz   (symmetry).
        """

        def integrand_pos(z):
            z2 = z*z
            shrink = 1.0 - a/(z2 + s)
            if shrink <= 0.0:
                return 0.0
            u = z * shrink
            w = u - tau
            if w <= 0.0:
                return 0.0
            return -4.0 * w * phi(z)  # *2 for ±z and *(-2) from derivative

        val, _ = integrate.quad(
            integrand_pos,
            0.0, np.inf,
            epsabs=1e-10, epsrel=1e-8, limit=200
        )
        return val

    # ---- integrate over S with chi-square density ----

    def C_tau(tau):
        """
        C(tau) = E[(eta(Z,S) - tau)_+^2] =
                 ∫_0^∞ Ez_sq_term(s, tau) * f_S(s) ds.
        """

        def integrand_s(s):
            return Ez_sq_term(s, tau) * chi2_pdf(s, df)

        val, _ = integrate.quad(
            integrand_s,
            0.0, np.inf,
            epsabs=1e-9, epsrel=1e-7, limit=200
        )
        return val

    def C_tau_prime(tau):
        """
        C'(tau) = ∫_0^∞ Ez_term_deriv(s, tau) * f_S(s) ds.
        """

        def integrand_s(s):
            return Ez_term_deriv(s, tau) * chi2_pdf(s, df)

        val, _ = integrate.quad(
            integrand_s,
            0.0, np.inf,
            epsabs=1e-9, epsrel=1e-7, limit=200
        )
        return val

    # ---- M(tau), M'(tau) ----

    def M(tau):
        tau = float(tau)
        return (
            eps1 * eps2 * (1.0 + tau*tau)
            + eps1 * (1.0 - eps2) * A_tau(tau)
            + (1.0 - eps1) * C_tau(tau)
        )

    def M_prime(tau):
        tau = float(tau)
        return (
            2.0 * eps1 * eps2 * tau
            + eps1 * (1.0 - eps2) * A_tau_prime(tau)
            + (1.0 - eps1) * C_tau_prime(tau)
        )

    def find_opt_tau(initial=1.0, tau_max=20.0):
        """
        Find tau>0 such that M'(tau)=0 using bracketing + Brent's method.
        """
        left = 1e-6
        f_left = M_prime(left)

        right = max(initial, left * 10.0)
        f_right = M_prime(right)
        while np.sign(f_left) == np.sign(f_right) and right < tau_max:
            right *= 2.0
            f_right = M_prime(right)

        if np.sign(f_left) == np.sign(f_right):
            raise RuntimeError(
                "Failed to bracket a root of M'(tau) on (0, tau_max). "
                "Try increasing tau_max or changing 'initial'."
            )

        tau_star = optimize.brentq(
            M_prime, left, right, xtol=1e-8, rtol=1e-8, maxiter=200
        )
        return tau_star

    return M, M_prime, find_opt_tau
# end threshold finding

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
    
    _, _, find_opt_tau = build_M_functions_continuous(block_sparsity, per_block_sparsity, B)

    # Solve for tau*
    thresh = find_opt_tau(initial=2.0)
    
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

        dict_current = amp_iteration_combined(A = A, 
                                     Y = Y_true, 
                                    signal_denoised_prev = signal_denoised_current,
                                    Residual_prev = Residual_current, 
                                    num_blocks = M,
                                    block_size = B,
                                    thresh = thresh,
                                    noise_var_current = noise_var_current)
        
        signal_denoised_current = dict_current['signal_denoised_current']
        Residual_current = dict_current['Residual_current']
        
        dict_observables = recovery_stats(signal_true,
                                   signal_denoised_current,
                                   sparsity_tol,
                                   A,
                                   Y_true, M, B)
        
        rel_err = dict_observables['rel_err']
        print(rel_err)
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
    return exp

def do_sherlock_experiment(json_file: str):
    exp = read_json(json_file)
    nodes = 100
    with SLURMCluster(queue='donoho,stat,hns,owners,normal',
                      cores=1, memory='8GiB', processes=1,
                      walltime='24:00:00') as cluster:
        cluster.scale(jobs=nodes)
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())
        cluster.scale(0)
        

if __name__ == '__main__':
    do_sherlock_experiment('exp_dicts/array_AMP_JS_ST_combined.json')
