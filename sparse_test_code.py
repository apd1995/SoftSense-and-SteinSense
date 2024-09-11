import timeit
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Original NumPy version
def block_soft_thresholding_nonsingular_numpy(X: np.ndarray, tau: float, Sigma_inv: np.ndarray) -> np.ndarray:
    quad_whitening = np.sum(X * np.matmul(X, Sigma_inv), axis=1)
    block_soft_thresholding_coeff = np.where(quad_whitening > tau**2, 1 - (tau / np.sqrt(quad_whitening)), 0.0)
    return X * block_soft_thresholding_coeff[:, np.newaxis]

# JAX-optimized dense version
@jax.jit
def block_soft_thresholding_nonsingular_jax(X: jnp.ndarray, tau: float, Sigma_inv: jnp.ndarray) -> jnp.ndarray:
    quad_whitening = jnp.sum(X * jnp.matmul(X, Sigma_inv), axis=1)
    block_soft_thresholding_coeff = jnp.where(quad_whitening > tau**2, 1 - (tau / jnp.sqrt(quad_whitening)), 0.0)
    return X * block_soft_thresholding_coeff[:, jnp.newaxis]

# JAX sparse version without JIT, no dense conversion
def block_soft_thresholding_nonsingular_jax_sparse(X: jnp.ndarray, tau: float, Sigma_inv: jnp.ndarray) -> jnp.ndarray:
    quad_whitening = jnp.sum(X * jnp.matmul(X, Sigma_inv), axis=1)
    block_soft_thresholding_coeff_dense = jnp.where(quad_whitening > tau**2, 1 - (tau / jnp.sqrt(quad_whitening)), 0.0)
    
    # Create sparse block_soft_thresholding_coeff
    block_soft_thresholding_coeff_sparse = BCOO.fromdense(block_soft_thresholding_coeff_dense)
    
    # Multiply element-wise with the sparse matrix - stay in sparse form
    X_sparse = BCOO.fromdense(X)
    
    # Sparse element-wise multiplication (row-wise)
    return X_sparse * block_soft_thresholding_coeff_sparse[:, None]

# Generate signal + noise data
def generate_signal_plus_noise(n_rows: int, n_cols: int, rng) -> np.ndarray:
    # Generate signal with probability 0.1 for i.i.d normal, and 0.9 for entirely zero rows
    signal = rng.normal(0, 1, (n_rows, n_cols)) * (rng.uniform(0, 1, n_rows) < 0.1)[:, np.newaxis]
    
    # Generate noise as i.i.d. normal
    noise = rng.normal(0, 1, (n_rows, n_cols))
    
    # Signal + noise
    X = signal + noise
    return X

# Set dimensions and parameters
n_rows, n_cols = 10000, 1000  # Large matrix (10000 rows, 1000 columns)
tau = 1.5
rng = np.random.default_rng(seed = 100)

# Generate signal + noise data for X
X = generate_signal_plus_noise(n_rows, n_cols, rng)

# Sigma_inv is just the identity matrix
Sigma_inv = np.eye(n_cols)

# Convert data to JAX arrays
X_jax = jnp.array(X)
Sigma_inv_jax = jnp.array(Sigma_inv)

# Define functions for timeit to execute
def time_numpy():
    return block_soft_thresholding_nonsingular_numpy(X, tau, Sigma_inv)

def time_jax():
    return block_soft_thresholding_nonsingular_jax(X_jax, tau, Sigma_inv_jax).block_until_ready()

def time_jax_sparse():
    return block_soft_thresholding_nonsingular_jax_sparse(X_jax, tau, Sigma_inv_jax)

# Warm-up for JAX (initial compilation)
block_soft_thresholding_nonsingular_jax(X_jax, tau, Sigma_inv_jax).block_until_ready()

# Timing NumPy version using timeit
numpy_time = timeit.timeit('time_numpy()', globals=globals(), number=10)
print(f"NumPy version average time over 10 runs: {numpy_time / 10:.4f} seconds")

# Timing JAX dense version using timeit
jax_time = timeit.timeit('time_jax()', globals=globals(), number=10)
print(f"JAX dense version average time over 10 runs: {jax_time / 10:.4f} seconds")

# Timing JAX sparse version using timeit (no JIT)
jax_sparse_time = timeit.timeit('time_jax_sparse()', globals=globals(), number=10)
print(f"JAX sparse version (no JIT) average time over 10 runs: {jax_sparse_time / 10:.4f} seconds")
