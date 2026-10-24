"""JAX backend for multivariate Gaussian recursive moments.

The recursive state packs lower-triangular Cholesky factors in canonical
row-major order for fit, prediction, and predictive resampling. 

Full covariance matrices are constructed only at prediction.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import solve_triangular

from .Sxx_kern import compute_alpha


# --- Utilities ---
@jit
def structured_cholesky_update_d2(L11, L21, L22, z1, z2, a, ρ=0.5):
    """Explicit d=2 update for speed.

    ``z1`` and ``z2`` have no x-location axis, so adding one here
    makes the same helper broadcast correctly in fit, prediction, and PR.
    """
    z1 = z1[..., None]
    z2 = z2[..., None]

    c = 1.0 - a * ρ**2
    t = a * (1.0 - a) * ρ**2
    s1 = c + t * z1**2
    s2 = s1 + t * z2**2

    A1 = jnp.sqrt(s1)
    A2 = jnp.sqrt(c * s2 / s1)
    B1 = t * z1 / A1

    return (
        A1 * L11,
        A1 * L21 + B1 * L22 * z2,
        A2 * L22,
    )


@jit
def structured_cholesky_update_d3(
    L11, L21, L22, L31, L32, L33, z1, z2, z3, a, ρ=0.5
):
    """Explicit d=3 update for speed."""
    z1 = z1[..., None]
    z2 = z2[..., None]
    z3 = z3[..., None]

    c = 1.0 - a * ρ**2
    t = a * (1.0 - a) * ρ**2
    s1 = c + t * z1**2
    s2 = s1 + t * z2**2
    s3 = s2 + t * z3**2

    A1 = jnp.sqrt(s1)
    A2 = jnp.sqrt(c * s2 / s1)
    A3 = jnp.sqrt(c * s3 / s2)
    B1 = t * z1 / A1
    B2 = A2 * t * z2 / s2

    L11_next = A1 * L11
    L21_next = A1 * L21 + B1 * L22 * z2
    L22_next = A2 * L22
    suffix31 = L32 * z2 + L33 * z3
    L31_next = A1 * L31 + B1 * suffix31
    L32_next = A2 * L32 + B2 * L33 * z3
    L33_next = A3 * L33
    return (
        L11_next,
        L21_next,
        L22_next,
        L31_next,
        L32_next,
        L33_next,
    )


def _packed_positions(d):
    """Canonical row-major positions in a packed lower triangle."""
    return [(row, col) for row in range(d) for col in range(row + 1)]


def _pack_moments(moments):
    """Convert dense moments to the packed array."""
    d = moments["μ"].shape[-1]
    positions = _packed_positions(d)
    means = tuple(moments["μ"][..., col] for col in range(d))
    factor = tuple(moments["L"][..., row, col] for row, col in positions)
    return means, factor


def _unpack_moments(means, factor):
    """Convert packed array to original shape"""
    d = len(means)
    μ = jnp.stack(means, axis=-1)
    L = jnp.zeros(μ.shape[:-1] + (d, d), dtype=μ.dtype)
    for value, (row, col) in zip(factor, _packed_positions(d)):
        L = L.at[..., row, col].set(value)
    return {"μ": μ, "L": L}


def _initial_packed_moments(n_locations, d, dtype, initial_factor=None):
    """Create zero means and packed initial Cholesky factor."""
    if initial_factor is None:
        initial_factor = jnp.eye(d, dtype=dtype)
    means = tuple(jnp.zeros(n_locations, dtype=dtype) for _ in range(d))
    factor = tuple(
        jnp.broadcast_to(initial_factor[row, col], (n_locations,))
        for row, col in _packed_positions(d)
    )
    return means, factor


@jit
def initial_correlation_cholesky(y, initial_correlation=0.):
    """Return the Cholesky factor of the initial correlation matrix."""
    _, d = y.shape
    correlation = (
        (1.0 - initial_correlation) * jnp.eye(d, dtype=y.dtype)
        + initial_correlation * jnp.ones((d, d), dtype=y.dtype)
    )
    return jnp.linalg.cholesky(correlation)


@jit
def initial_covariance_cholesky(
    y, ρ=0.5, initial_correlation=0.
):
    """Return an initial covariance factor with unit marginal variances.
    """
    correlation_factor = initial_correlation_cholesky(
        y, initial_correlation=initial_correlation
    )
    return correlation_factor 


@jit
def packed_matvec(factor, z):
    """Compute ``L z`` directly from canonical packed entries."""
    positions = _packed_positions_from_factor(factor)
    position_index = {
        position: index for index, position in enumerate(positions)
    }
    d = positions[-1][0] + 1
    w = []
    for row in range(d):
        value = factor[position_index[(row, 0)]] * z[..., 0][..., None]
        for col in range(1, row + 1):
            value = (
                value
                + factor[position_index[(row, col)]]
                * z[..., col][..., None]
            )
        w.append(value)
    return tuple(w)


def _packed_positions_from_factor(factor):
    """Infer static positions from a packed factor's pytree length."""
    positions = []
    row = 0
    while len(positions) < len(factor):
        positions.extend((row, col) for col in range(row + 1))
        row += 1
    return positions


@jit
def packed_triangular_solve(factor, residual):
    """Forward solve a single packed lower-triangular system."""
    positions = _packed_positions_from_factor(factor)
    position_index = {
        position: index for index, position in enumerate(positions)
    }
    d = positions[-1][0] + 1
    solution = []
    for row in range(d):
        value = residual[..., row]
        for col in range(row):
            value = value - factor[position_index[(row, col)]] * solution[col]
        solution.append(value / factor[position_index[(row, row)]])
    return jnp.stack(solution, axis=-1)


@jit
def structured_cholesky_update_packed(factor, z, a, ρ=0.5):
    """Apply ``Sigma+ = L (c I + t z z.T) L.T`` to the packed factor in O(d^2).

    Uses structured Cholesky formulas derived from Gaussian elimination.
    Leading dimensions may include predictive-resampling paths.
    """
    positions = _packed_positions_from_factor(factor)
    d = positions[-1][0] + 1

    if d == 1:
        z1 = z[..., 0][..., None]
        c = 1.0 - a * ρ**2
        t = a * (1.0 - a) * ρ**2
        return (factor[0] * jnp.sqrt(c + t * z1**2),)
    if d == 2:
        return structured_cholesky_update_d2(
            factor[0], factor[1], factor[2], z[..., 0], z[..., 1], a, ρ
        )
    if d == 3:
        return structured_cholesky_update_d3(
            factor[0],
            factor[1],
            factor[2],
            factor[3],
            factor[4],
            factor[5],
            z[..., 0],
            z[..., 1],
            z[..., 2],
            a,
            ρ,
        )

    position_index = {
        position: index for index, position in enumerate(positions)
    }
    c = 1.0 - a * ρ**2
    t = a * (1.0 - a) * ρ**2
    s_previous = c
    A = []
    B_coefficient = []
    for col in range(d):
        z_col = z[..., col][..., None]
        s = s_previous + t * z_col**2
        A_col = jnp.sqrt(c * s / s_previous)
        A.append(A_col)
        # Use B_j / A_j = t z_j / s_j to avoid an additional square root.
        B_coefficient.append(A_col * t * z_col / s)
        s_previous = s

    factor_next = [None] * len(positions)
    for row in range(d):
        suffix = jnp.zeros_like(factor[0])
        for col in range(row, -1, -1):
            index = position_index[(row, col)]
            factor_next[index] = (
                A[col] * factor[index] + B_coefficient[col] * suffix
            )
            suffix = suffix + factor[index] * z[..., col][..., None]
    return tuple(factor_next)


@jit
def update_packed_moments(means, factor, z, a, ρ=0.5):
    """Shared packed moment update for fit, prediction, and PR."""
    w = packed_matvec(factor, z)
    aρ = a * ρ
    means_next = tuple(
        means[col] + aρ * w[col] for col in range(len(means))
    )
    factor_next = structured_cholesky_update_packed(factor, z, a, ρ)
    return means_next, factor_next


@jit
def covariance_from_cholesky(L):
    """Construct covariance matrices from one or more Cholesky factors."""
    return jnp.einsum("...ik,...jk->...ij", L, L)


@jit
def log_pdf(
    y,
    moments,
    initial_correlation=0.,
):
    """Log density of a multivariate Gaussian represented by ``(mu, L)``."""
    residual = jnp.asarray(y) - moments["μ"]
    z = solve_triangular(
        moments["L"], residual[..., None], lower=True
    ).squeeze(-1)
    d = residual.shape[-1]
    log_det = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(moments["L"], axis1=-2, axis2=-1)), axis=-1
    )
    return -0.5 * (
        d * jnp.log(2.0 * jnp.pi) + log_det + jnp.sum(z**2, axis=-1)
    )


@jit
def compute_quantile(
    moments,
    α=0.5,
    initial_correlation=0.,
):
    """Return coordinate-wise marginal Gaussian quantiles."""
    marginal_sd = jnp.sqrt(jnp.sum(moments["L"] ** 2, axis=-1))
    return moments["μ"] + marginal_sd * jax.scipy.special.ndtri(α)


# --- Fit model ---
@partial(jax.jit, static_argnames=["block_size"])
def fit_moments(y, log_Sn, ρ=0.5, initial_correlation=0., block_size=None):
    """Fit fully observed outcomes, with same blocking structure for optimization.
    """
    n, d = y.shape
    L_init = initial_covariance_cholesky(y, ρ, initial_correlation)
    carry_init = _initial_packed_moments(n, d, y.dtype, L_init)
    positions = _packed_positions(d)
    diagonal_indices = tuple(
        index for index, (row, col) in enumerate(positions) if row == col
    )

    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)
    half_d_log_2pi = 0.5 * d * jnp.log(2.0 * jnp.pi)

    def scan_body(carry, xs):
        i, y_i, logit_α_i, log_Sn_col = xs
        means, factor = carry

        mean_i = jnp.stack(tuple(value[i] for value in means))
        factor_i = tuple(value[i] for value in factor)
        residual = y_i - mean_i
        z_i = packed_triangular_solve(factor_i, residual)
        a = jax.nn.sigmoid(logit_α_i + log_Sn_col)
        log_preq_i = (
            -0.5 * jnp.sum(z_i**2)
            - half_d_log_2pi
            - sum(
                jnp.log(factor_i[index])
                for index in diagonal_indices
            )
        )
        carry_next = update_packed_moments(means, factor, z_i, a, ρ)

        return carry_next, (z_i, log_preq_i)

    if block_size is None:
        xs = (jnp.arange(n), y, logit_α, log_Sn.T)
        _, (z, log_preqs) = jax.lax.scan(scan_body, carry_init, xs)
    else:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        carry = carry_init
        blocks = []
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            size = stop - start
            # State indices are local; learning rates keep their global indices.
            xs = (
                jnp.arange(size), y[start:stop], logit_α[start:stop],
                log_Sn[start:, start:stop].T,
            )
            carry, output = jax.lax.scan(scan_body, carry, xs)
            blocks.append(output)
            carry = jax.tree.map(lambda value: value[size:], carry)
        z, log_preqs = jax.tree.map(
            lambda *values: jnp.concatenate(values), *blocks
        )
    model = {"z": z, "L_init": L_init}
    return model, jnp.mean(log_preqs)


# --- Predict at test points ---
@partial(jax.jit, static_argnames=["unroll"])
def scan_moment_updates(moments_init, z_seq, log_S, logit_α, ρ=0.5, unroll=1):
    """Apply stored residuals to packed moments, then return mean vectors and Cholesky matrices."""
    carry_init = _pack_moments(moments_init)

    def scan_body(carry, xs):
        z_i, logit_α_i, log_S_col = xs
        a = jax.nn.sigmoid(logit_α_i + log_S_col)
        means, factor = carry
        carry_next = update_packed_moments(means, factor, z_i, a, ρ)
        return carry_next, None

    xs = (z_seq, logit_α, log_S.T)
    (means_final, factor_final), _ = jax.lax.scan(
        scan_body, carry_init, xs, unroll=unroll
    )
    return _unpack_moments(means_final, factor_final)


@jit
def test_moments(model, log_S_test, ρ=0.5):
    n_test, n = log_S_test.shape
    d = model["z"].shape[-1]
    dtype = model["z"].dtype
    moments_init = {
        "μ": jnp.zeros((n_test, d), dtype=dtype),
        "L": jnp.broadcast_to(model["L_init"], (n_test, d, d)),
    }

    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)
    return scan_moment_updates(
        moments_init, model["z"], log_S_test, logit_α, ρ, unroll=1
    )


# --- Permutation averaging ---
@partial(jax.jit, static_argnames=["n_perm"])
def fit_moments_perm(
    y,
    log_Sn,
    ρ=0.5,
    n_perm=10,
    seed=121,
    initial_correlation=0.,
):
    n = y.shape[0]
    key = random.PRNGKey(seed)
    keys = random.split(key, n_perm)
    ind_perm = jax.vmap(lambda k: random.permutation(k, n))(keys)

    def single_perm_fit(ind):
        y_p = y[ind]
        log_Sn_p = log_Sn[ind, :][:, ind]
        return fit_moments(y_p, log_Sn_p, ρ, initial_correlation)

    model_perm, log_preq_perm = jax.vmap(single_perm_fit)(ind_perm)
    model_perm = {**model_perm, "ind_perm": ind_perm}
    return model_perm, jnp.mean(log_preq_perm)


@jit
def test_moments_perm(
    model_perm,
    log_S_test,
    ρ=0.5,
):
    def single_perm_test(model):
        ind_p = model["ind_perm"]
        return test_moments(model, log_S_test[:, ind_p], ρ)

    moments_perm = jax.vmap(single_perm_test)(model_perm)
    μ_p = moments_perm["μ"]
    L_p = moments_perm["L"]

    # With one permutation, return its moments directly; no averaging is needed.
    if μ_p.shape[0] == 1:
        return {"μ": μ_p[0], "L": L_p[0]}

    # Average across permutations on the raw moment scale, then recover the
    # covariance and its Cholesky factor after all recursive updates.
    μ_final = jnp.mean(μ_p, axis=0)
    covariance_p = covariance_from_cholesky(L_p)
    centered_μ = μ_p - μ_final[None, ...]
    covariance_final = jnp.mean(
        covariance_p
        + jnp.einsum("...i,...j->...ij", centered_μ, centered_μ),
        axis=0,
    )
    covariance_final = 0.5 * (
        covariance_final + jnp.swapaxes(covariance_final, -1, -2)
    )
    L_final = jnp.linalg.cholesky(covariance_final)
    return {"μ": μ_final, "L": L_final}


# --- Predictive resampling ---
@partial(jax.jit, static_argnames=["B"])
def pr_moments(
    moments_init,
    log_S_test,
    n,
    ρ=0.5,
    B=10,
    seed=121,
    initial_correlation=0.,
):
    """Run B resampling paths using a canonical packed Cholesky state.

    The lower triangle is packed in row-major order::

        (L00, L10, L11, L20, L21, L22, ...)
    """
    T = log_S_test.shape[1]
    d = moments_init["μ"].shape[-1]
    n_test = moments_init["μ"].shape[-2]

    master_key = random.PRNGKey(seed)
    keys = random.split(master_key, B)

    i_ = jnp.arange(1, T + 1) + n
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    if d == 1:
        # Store univariate means and scales as (B, n_test) arrays.
        z_pr = jax.vmap(
            lambda key: random.normal(key, shape=(T, 1))
        )(keys)
        carry_init = (
            jnp.broadcast_to(moments_init["μ"][..., 0], (B, n_test)),
            jnp.broadcast_to(moments_init["L"][..., 0, 0], (B, n_test)),
        )

        def scan_body(carry, xs):
            μ, L = carry
            z_i, logit_α_i, log_S_col = xs
            a = jax.nn.sigmoid(logit_α_i + log_S_col)
            z_scalar = z_i[:, 0, None]
            w = L * z_scalar
            μ_next = μ + a * ρ * w
            variance_factor = 1.0 + ρ**2 * a * (
                (1.0 - a) * z_scalar**2 - 1.0
            )
            return (μ_next, L * jnp.sqrt(variance_factor)), None

        (μ, L), _ = jax.lax.scan(
            scan_body,
            carry_init,
            (jnp.swapaxes(z_pr, 0, 1), logit_α, log_S_test.T),
            unroll=1,
        )
        return {"μ": μ[..., None], "L": L[..., None, None]}

    means_init, factor_init = _pack_moments(moments_init)
    carry_init = means_init + factor_init

    # Chunking keeps the CPU working set cache-friendly.
    if jax.default_backend() == "cpu":
        batch_size = min(B, max(1, 2500 // n_test))
    else:
        batch_size = B

    def single_path(key):
        z_pr = random.normal(key, shape=(T, d))

        def scan_body(carry, xs):
            z_i, logit_α_i, log_S_col = xs
            a = jax.nn.sigmoid(logit_α_i + log_S_col)
            means, factor = update_packed_moments(
                carry[:d], carry[d:], z_i, a, ρ
            )
            return means + factor, None

        carry, _ = jax.lax.scan(
            scan_body,
            carry_init,
            (z_pr, logit_α, log_S_test.T),
            unroll=1,
        )
        return carry

    packed = jax.lax.map(single_path, keys, batch_size=batch_size)
    means = packed[:d]
    factor = packed[d:]
    return _unpack_moments(means, factor)
