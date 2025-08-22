#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:40:24 2025

@author: apratimdey
"""

from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, jacfwd, vmap, device_get
from functools import partial
from EMS.manager_new import read_json, do_on_cluster, get_gbq_credentials
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from pandas import DataFrame
import logging
logging.basicConfig(level=logging.INFO)
log_gbq = logging.getLogger('pandas_gbq')
log_gbq.setLevel(logging.DEBUG)
log_gbq.addHandler(logging.StreamHandler())
logging.getLogger('jax').setLevel(logging.ERROR)
import time
import dask.config
dask.config.set({
    "distributed.nanny.timeouts.startup": "300s"   # 5 minutes
})


CHUNK = 100

def seed(gaussian_mean: float,
         nonzero_rows: float,
         num_measurements: float,
         signal_nrow: float,
         signal_ncol: float,
         err_tol: float,
         mc: float,
         sparsity_tol: float) -> int:
    return round(1 + round(gaussian_mean * 100) +
                 round(nonzero_rows * 1000) + 
                 round(num_measurements * 1000) + round(signal_nrow * 1000) + 
                 round(signal_ncol * 1000) + round(err_tol * 100000) + round(mc * 100000) + 
                 round(sparsity_tol * 1000000))


# ─── 1) JAMES–STEIN DENOISERS ───────────────────────────────────────────

def js_nonsingular_vec(y, Sigma_inv):
    d    = y.shape[0]
    quad = y @ (Sigma_inv @ y)                               # ( )
    coeff = jnp.where(quad > (d - 2), 1 - (d - 2) / quad, 0.) # scalar
    return y * coeff                                         # (d,)


def js_singular_vec(y, eigvecs, inv_full):
    y0   = eigvecs.T @ y                 # (B,)
    quad = jnp.sum((y0**2) * inv_full)   # scalar
    k    = jnp.count_nonzero(inv_full)   # dynamic, but used only in math
    coeff = jnp.where(quad > (k - 2),
                      1 - (k - 2) / quad,
                      0.0)
    y0_hat = coeff * y0 * (inv_full > 0) + y0 * (inv_full == 0)
    return eigvecs @ y0_hat              # (B,)


# Vectorized versions (compiled once)
v_js_nonsingular = jit(vmap(js_nonsingular_vec, in_axes=(0, None)))
v_js_singular    = jit(vmap(js_singular_vec,
                            in_axes=(0, None, None)))

# Precompute Jacobians of the row-wise functions, batched over rows
jac_js_nonsingular = jit(
    vmap(jacfwd(js_nonsingular_vec, argnums=0), in_axes=(0, None))
)
jac_js_singular = jit(
    vmap(jacfwd(js_singular_vec, argnums=0), in_axes=(0, None, None))
)

@jit
def js_onsager_nonsingular(
    X:        jnp.ndarray,     # shape (N, B)
    Z:        jnp.ndarray,     # shape (n, B)
    Sigma_inv:jnp.ndarray      # shape (B, B)
) -> jnp.ndarray:
    """
    Returns (Z @ sum_i J_i^T) / n, where J_i is the Jacobian
    of js_nonsingular_vec at X[i].
    """
    J = jac_js_nonsingular(X, Sigma_inv)
    sumJ    = jnp.sum(J, axis=0)
    return (Z @ sumJ.T) / Z.shape[0]

@jit
def js_onsager_singular(X, Z, U, inv_full):
    J = jac_js_singular(X, U, inv_full)
    sumJ    = jnp.sum(J, axis=0)
    return (Z @ sumJ.T) / Z.shape[0]

# ─── 2) FUSED AMP LOOP ─────────────────────────────────────────────────

@jit
def recovery_stats_jax(X_true, X_rec, A, Y_true, sparsity_tol):
    """
    Returns the same dictionary as your original `recovery_stats`
    but computed with JAX primitives only (no CVXPY, no NumPy loops).
    All inputs must already live on-device.
    """
    N, B = X_true.shape
    n    = Y_true.shape[0]

    Y_rec = A @ X_rec                               # (n, B)

    # ---------- helpers ----------
    def frob(M):   return jnp.linalg.norm(M)        # ‖·‖_F
    def row_l2(M): return jnp.linalg.norm(M, axis=1)

    # sparsity masks
    nz_true = row_l2(X_true) > 0
    nz_rec  = row_l2(X_rec)  / jnp.sqrt(B) > sparsity_tol

    # mixed norms
    norm_2_1   = lambda M: row_l2(M).sum()
    norm_2_2   = lambda M: row_l2(M).mean()
    norm_2_inf = lambda M: row_l2(M).max()

    stats = {
        'rel_err'                       : frob(X_true - X_rec) / frob(X_true),
        'rel_err_measurements'          : frob(Y_true - Y_rec) / frob(Y_true),
        'avg_err'                       : frob(X_true - X_rec) / jnp.sqrt(N*B),
        'avg_err_measurements'          : frob(Y_true - Y_rec) / jnp.sqrt(n*B),
        'max_row_err'                   : row_l2(X_true - X_rec).max() / jnp.sqrt(B),
        'max_row_err_measurements'      : row_l2(Y_true - Y_rec).max() / jnp.sqrt(B),

        'norm_2_1_true'                 : norm_2_1(X_true) / (N*jnp.sqrt(B)),
        'norm_2_1_rec'                  : norm_2_1(X_rec)  / (N*jnp.sqrt(B)),
        'norm_2_2_true'                 : norm_2_2(X_true) / jnp.sqrt(N*B),
        'norm_2_2_rec'                  : norm_2_2(X_rec)  / jnp.sqrt(N*B),
        'norm_2_inf_true'               : norm_2_inf(X_true) / jnp.sqrt(B),
        'norm_2_inf_rec'                : norm_2_inf(X_rec)  / jnp.sqrt(B),

        'soft_sparsity'                 : (row_l2(X_rec)/jnp.sqrt(B) > sparsity_tol).mean(),
        'nonzero_rows_rec'              : (row_l2(X_rec)/jnp.sqrt(B) > sparsity_tol).sum(),

        'tpr'                           : ( (~nz_true) & (~nz_rec) ).sum() /
                                          jnp.maximum(1, (~nz_true).sum()),
        'tnr'                           : (  nz_true  &  nz_rec  ).sum() /
                                          jnp.maximum(1, nz_true.sum()),

        # measurement-space norms
        'norm_2_1_true_measurements'    : norm_2_1(Y_true) / (n*jnp.sqrt(B)),
        'norm_2_1_rec_measurements'     : norm_2_1(Y_rec)  / (n*jnp.sqrt(B)),
        'norm_2_2_true_measurements'    : norm_2_2(Y_true) / jnp.sqrt(n*B),
        'norm_2_2_rec_measurements'     : norm_2_2(Y_rec)  / jnp.sqrt(n*B),
        'norm_2_inf_true_measurements'  : norm_2_inf(Y_true) / jnp.sqrt(B),
        'norm_2_inf_rec_measurements'   : norm_2_inf(Y_rec)  / jnp.sqrt(B),
    }
    return stats

@partial(jit, static_argnames=("steps",))
def amp_chunk(A, Y, X0, R0, err_tol, err_explosion_tol, *, X_true, steps):
    n, N = A.shape

    def step(carry, _):
        X, R, stop_flag = carry
        
        # Compute relative error
        rel = jnp.linalg.norm(X - X_true) / (jnp.linalg.norm(X_true) + 1e-12)
        new_stop = (rel < err_tol) | (rel > err_explosion_tol)
        
        def do_compute(_):

            X_noisy = X + A.T @ R
            Cov = jnp.cov(R.T)
            D, U = jnp.linalg.eigh(Cov)
            nonsingular_branch = jnp.all(D > 0)
    
            def do_nonsingular():
                Sigma_inv = (U * (1 / D)[None, :]) @ U.T
                X_denoised = v_js_nonsingular(X_noisy, Sigma_inv)
                Rn = Y - A @ X_denoised + js_onsager_nonsingular(X_noisy, R, Sigma_inv)
                return X_denoised, Rn, jnp.array(0)
    
            def do_singular():
                inv_full = jnp.where(D > 0, 1 / D, 0)
                X_denoised = v_js_singular(X_noisy, U, inv_full)
                Rn = Y - A @ X_denoised + js_onsager_singular(X_noisy, R, U, inv_full)
                return X_denoised, Rn, jnp.array(1)

            return lax.cond(nonsingular_branch, do_nonsingular, do_singular)
        
        X_new, R_new, branch_used = lax.cond(new_stop, lambda _: (X, R, jnp.array(-1)), do_compute, operand=None)

        stop_flag = stop_flag | new_stop

        return (X_new, R_new, stop_flag), (rel, branch_used)

    (Xf, Rf, stop_final), (rels, branches) = lax.scan(
        step, (X0, R0, False), None, length=steps
    )
    
    hit = (rels < err_tol) | (rels > err_explosion_tol)
    idx = jnp.argmax(hit)
    rel_at_stop = jnp.where(hit.any(), rels[idx], rels[-1])

    return Xf, Rf, rel_at_stop, stop_final, steps, branches


def run_amp_instance(**dict_params):
    
    mu = dict_params['gaussian_mean']
    k = dict_params['nonzero_rows']
    n = dict_params['num_measurements']
    N = dict_params['signal_nrow']
    B = dict_params['signal_ncol']
    err_tol = dict_params['err_tol']
    mc = dict_params['mc']
    sparsity_tol = dict_params['sparsity_tol']
    max_iter = dict_params['max_iter']
    err_explosion_tol = dict_params['err_explosion_tol']
    
    start_time = time.perf_counter()
    
    seed_val = seed(mu, k, n, N, B, err_tol, mc, sparsity_tol)
    
    rng = np.random.default_rng(seed=seed_val)
    nz_idx = rng.choice(N, k, replace=False)
    nonzero_vals = rng.normal(mu, 1, (k, B))
    
    X_true = jnp.zeros((N, B)).at[nz_idx].set(nonzero_vals).astype(jnp.float64)

    A = rng.normal(0, 1, (n, N)) / jnp.sqrt(n)
    Y = A @ X_true

    X, R = jnp.zeros_like(X_true), Y
    df   = None
    it = 0
    min_rel = 1
    
    rows = []
    
    while it < max_iter:
        # ---- heavy compute on GPU ---------------
        steps = min(CHUNK, max_iter - it)
        t0 = time.perf_counter()
        X, R, rel, stop, used, branch_trace = amp_chunk(A, Y, X, R, err_tol, err_explosion_tol, X_true=X_true, steps=steps)
        X.block_until_ready()
        elapsed = time.perf_counter() - t0
        
        branches = device_get(branch_trace)

        it += used
        min_rel = min(min_rel, rel.item())
        # -----------------------------------------

        # ---- full statistics  (GPU → host) ------
        stats_gpu = recovery_stats_jax(X_true, X, A, Y, sparsity_tol)
        stats_host  = device_get(stats_gpu)
        observables = {k: v.item() for k, v in stats_host.items()}    # Device → Python
        time_since_start = time.perf_counter() - start_time
        observables.update({
            'iter_count'          : int(it),
            'min_rel_err'         : min_rel,
            'chunk_time' : elapsed,
            'time_since_start': time_since_start,
            'skipped': int((branches == -1).sum()),
            'singular': int((branches == 1).sum()),
            'nonsingular': int((branches == 0).sum())
        })
        rows.append({**dict_params, **observables})
        # -----------------------------------------

        if bool(stop) or it >= max_iter:
            break
    
    df = DataFrame(rows)

    return df
    

# ─── 4) CLUSTER LAUNCHER ────────────────────────────────────────────────

def do_sherlock_experiment(json_file: str):
    exp = read_json(json_file)
    with SLURMCluster(queue='donoho,gpu,stat,hns,owners,normal',
                      cores=1, memory='10GiB', processes=1,
                      walltime='24:00:00', job_extra_directives=['--gres=gpu:1',
                                                                 '--constraint="GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5"'],
                      job_script_prologue=[
        'module load cuda/12.6.1',
        'export JAX_ENABLE_X64=true'
    ],
                      death_timeout='60s') as cluster:
        cluster.adapt(minimum = 50, maximum = 200, target_duration = "10h", interval = "30s")
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())
        cluster.scale(0)
        
        
def read_and_do_local_experiment(json_file: str):
    exp = read_json(json_file)
    with LocalCluster(dashboard_address='localhost:8787', n_workers=2) as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, run_amp_instance, client, credentials=None)
            # do_on_cluster(exp, run_amp_instance, client, credentials=get_gbq_credentials())


if __name__ == '__main__':
    do_sherlock_experiment('exp_dicts/AMP_matrix_recovery_JS_gaussian_nonzero_jaxcuda.json')
    # read_and_do_local_experiment('exp_dicts/AMP_matrix_recovery_JS_gaussian_nonzero_jaxcuda.json')
    # d = run_amp_instance(**{'gaussian_mean': 0,
    #                 'nonzero_rows': 1000,
    #                 'signal_nrow': 5000,
    #                 'signal_ncol': 50,
    #                 'num_measurements': 1147,
    #                 'err_tol': 0.0001,
    #                 'sparsity_tol': 0.0001,
    #                 'mc': 0,
    #                 'err_explosion_tol': 100,
    #                 'max_iter': 5000})
    # print(d[['rel_err', 'singular', 'nonsingular']])


