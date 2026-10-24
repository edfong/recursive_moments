"""Missing-outcome backend for multivariate Gaussian recursive moments.
Missing updates currently only support d=3, with only the third outcome possibly missing.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import solve_triangular

from .Sxx_kern import compute_alpha
from .mogauss_fun import (
    _initial_packed_moments,
    _unpack_moments,
    covariance_from_cholesky,
    initial_covariance_cholesky,
    update_packed_moments,
)


@jit
def conditional_last_log_pdf(y, moments):
    """Gaussian log p(y_last | y_previous, x), using raw-unit Cholesky factors.

    The last innovation has conditional variance L[-1,-1]**2. 
    """
    z = solve_triangular(moments["L"], (y - moments["μ"])[..., None],
                         lower=True)[..., -1, 0]
    return -.5 * (jnp.log(2 * jnp.pi) + z**2) - jnp.log(moments["L"][..., -1, -1])


@jit
def update_packed_moments_d3_masked(
    means, factor, z, third_missing, a, ρ=0.5
):
    """Shared d=3 update: mask z3 and preserve L[-1, -1] when Y3 is missing.

    Zeroing z3 alone would incorrectly shrink the final conditional scale.
    All other entries use exactly the fully observed update.
    """
    z = z.at[..., 2].set(jnp.where(third_missing, 0.0, z[..., 2]))
    means_next, factor_next = update_packed_moments(means, factor, z, a, ρ)
    last_scale = jnp.where(
        jnp.asarray(third_missing)[..., None], factor[-1], factor_next[-1]
    )
    return means_next, (*factor_next[:-1], last_scale)


@partial(jax.jit, static_argnames=["block_size"])
def fit_moments(
    y, log_Sn, ρ=0.5, initial_correlation=0., block_size=None
):
    """Fit d=3 when only the third outcome may be missing.
    """
    n, d = y.shape
    if d != 3:
        raise ValueError("The missing-outcome backend supports only d=3.")
    L_init = initial_covariance_cholesky(y, ρ, initial_correlation)
    carry_init = _initial_packed_moments(n, d, y.dtype, L_init)

    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)
    log_2pi = jnp.log(2.0 * jnp.pi)

    def scan_body(carry, xs):
        i, y_i, logit_α_i, log_Sn_col = xs
        means, factor = carry
        mean_i = jnp.stack(tuple(value[i] for value in means))
        factor_i = tuple(value[i] for value in factor)
        a = jax.nn.sigmoid(logit_α_i + log_Sn_col)
        third_missing = jnp.isnan(y_i[2])

        z1 = (y_i[0] - mean_i[0]) / factor_i[0]
        z2 = (y_i[1] - mean_i[1] - factor_i[1] * z1) / factor_i[2]
        # Remove NaN before arithmetic, and mask the innovation before division.
        y3 = jnp.where(third_missing, 0.0, y_i[2])
        residual3 = y3 - mean_i[2] - factor_i[3] * z1 - factor_i[4] * z2
        z3 = jnp.where(third_missing, 0.0, residual3) / factor_i[5]
        z_i = jnp.stack((z1, z2, z3))
        log_preq_i = (
            -0.5 * (z1**2 + z2**2) - log_2pi
            - jnp.log(factor_i[0]) - jnp.log(factor_i[2])
            + jnp.where(third_missing, 0.0,
                        -0.5 * (z3**2 + log_2pi) - jnp.log(factor_i[5]))
        )
        carry_next = update_packed_moments_d3_masked(
            means, factor, z_i, third_missing, a, ρ
        )
        model_i = {"z": z_i, "third_missing": third_missing}
        return carry_next, (model_i, log_preq_i)


    #For optimization, block_size drops completed training locations after each block.
    if block_size is None:
        xs = (jnp.arange(n), y, logit_α, log_Sn.T)
        _, (model, log_preqs) = jax.lax.scan(scan_body, carry_init, xs)
    else:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        carry = carry_init
        blocks = []
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            size = stop - start
            # Local state indices restart at zero; 
            xs = (
                jnp.arange(size), y[start:stop], logit_α[start:stop],
                log_Sn[start:, start:stop].T,
            )
            carry, output = jax.lax.scan(scan_body, carry, xs)
            blocks.append(output)
            carry = jax.tree.map(lambda value: value[size:], carry)
        model, log_preqs = jax.tree.map(
            lambda *values: jnp.concatenate(values), *blocks
        )
    model = {**model, "L_init": L_init}
    return model, jnp.mean(log_preqs)


@jit
def test_moments(model, log_S_test, ρ=0.5):
    """Replay full and third-missing d=3 updates at prediction locations."""
    n_test, n = log_S_test.shape
    d = model["z"].shape[-1]
    if d != 3:
        raise ValueError("The missing-outcome backend supports only d=3.")
    dtype = model["z"].dtype
    carry_init = _initial_packed_moments(n_test, d, dtype, model["L_init"])

    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    def scan_body(carry, xs):
        z_i, third_missing_i, logit_α_i, log_S_col = xs
        a = jax.nn.sigmoid(logit_α_i + log_S_col)
        means, factor = carry

        carry_next = update_packed_moments_d3_masked(
            means, factor, z_i, third_missing_i, a, ρ
        )
        return carry_next, None

    xs = (
        model["z"],
        model["third_missing"],
        logit_α,
        log_S_test.T,
    )
    (means_final, factor_final), _ = jax.lax.scan(
        scan_body, carry_init, xs, unroll=1
    )
    return _unpack_moments(means_final, factor_final)


@partial(jax.jit, static_argnames=["n_perm"])
def fit_moments_perm(
    y,
    log_Sn,
    ρ=0.5,
    n_perm=10,
    seed=121,
    initial_correlation=0.,
):
    keys = random.split(random.PRNGKey(seed), n_perm)
    ind_perm = jax.vmap(lambda key: random.permutation(key, y.shape[0]))(keys)

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

    if μ_p.shape[0] == 1:
        return {"μ": μ_p[0], "L": L_p[0]}

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
    return {"μ": μ_final, "L": jnp.linalg.cholesky(covariance_final)}
