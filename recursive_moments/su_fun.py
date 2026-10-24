import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from .Sxx_kern import compute_alpha


# ---------------------------------------------------------------------
# Heavy-tailed SU update functions
# ---------------------------------------------------------------------
# Chosen ordinary kurtosis λ >= 3.
# Use λ > 3 for the Johnson SU heavy-tailed version.
λ = 40.0


# ---------------------------------------------------------------------
# Symmetric Johnson SU helper functions
# ---------------------------------------------------------------------

@jit
def su_params(λ=λ):
    """
    Parameters for the symmetric standardized Johnson SU transform.
    g_λ(z) = sinh(z / δ_λ) / D_λ
    """
    omega = jnp.sqrt(jnp.sqrt(2.0 * (λ - 1.0)) - 1.0)
    delta = 1.0 / jnp.sqrt(jnp.log(omega))
    D = jnp.sqrt(0.5 * (omega - 1.0) * (omega + 1.0))
    return omega, delta, D


@jit
def su_g(z, λ=λ):
    """
    Forward standardized transform: ε = g_λ(z)
    """
    _, delta, D = su_params(λ)
    return jnp.sinh(z / delta) / D


@jit
def su_g_inv(eps, λ=λ):
    """
    Inverse standardized transform:
        z = g_λ^{-1}(ε)
    """
    _, delta, D = su_params(λ)
    return delta * jnp.arcsinh(D * eps)



@jit
def su_R1_R2(z_i, ρ=0.5, λ=λ):
    """
    Compute R^{(1)}_λ(z_i) and R^{(2)}_λ(z_i) for the symmetric SU case.
    """
    omega, delta, D = su_params(λ)

    ρ2 = ρ**2
    one_minus_ρ2 = 1.0 - ρ2

    # omega_tilde = exp((1 - ρ^2) / δ^2) = omega^(1 - ρ^2)
    omega_tilde = omega ** one_minus_ρ2
    sqrt_omega_tilde = omega ** (0.5 * one_minus_ρ2)

    u = ρ * z_i / delta
    sinh_u = jnp.sinh(u)
    sinh_u2 = sinh_u**2

    R1 = sqrt_omega_tilde * sinh_u / D

    # D(omega_tilde, t_tilde)^2, with t_tilde^2 = sinh_u^2.
    D_tilde2 = (
        0.5
        * (omega_tilde - 1.0)
        * (2.0 * sinh_u2 * omega_tilde + omega_tilde + 1.0)
    )

    R2 = (D_tilde2 + omega_tilde * sinh_u2) / (D**2)

    return R1, R2


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

@jit
def log_pdf(y, moments, λ=λ):
    """
    Log PDF under
        Y = μ + σ g_λ(Z),    Z ~ N(0,1),
    """
    μ = moments["μ"]
    σ2 = moments["σ2"]
    σ = jnp.sqrt(σ2)

    omega, delta, D = su_params(λ)

    eps = (y - μ) / σ
    z = su_g_inv(eps, λ)

    # log p_Y(y) = log φ(z) + log |dz/dy|
    #
    # z = δ asinh(D eps)
    # dz/dy = δ D / [σ sqrt(1 + (D eps)^2)]
    log_phi_z = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * z**2
    log_jac = (
        jnp.log(delta)
        + jnp.log(D)
        - jnp.log(σ)
        - 0.5 * jnp.log1p((D * eps) ** 2)
    )

    return log_phi_z + log_jac


@jit
def compute_quantile(moments, α=0.5, λ=λ):
    """
    Conditional quantile under the heavy-tailed SU family.
    """
    zα = jax.scipy.special.ndtri(α)
    return moments["μ"] + jnp.sqrt(moments["σ2"]) * su_g(
        zα,
        λ,
    )


# ---------------------------------------------------------------------
# Fit model
# ---------------------------------------------------------------------

@jit
def fit_moments(y, log_Sn, ρ=0.5, λ=λ):
    n = y.shape[0]

    # Initialize moments
    moments = {
        "μ": jnp.zeros(n),
        "σ2": jnp.ones(n),
    }

    # Precompute alpha_i
    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    def scan_body(moments_carry, xs):
        μ = moments_carry["μ"]
        σ2 = moments_carry["σ2"]

        i, y_i, logit_α_i, log_Sn_col = xs

        # --------------------------------------------------------------
        # Residual under the heavy-tailed transform
        # --------------------------------------------------------------
        σ = jnp.sqrt(σ2)
        σ_i = σ[i]

        eps_i = (y_i - μ[i]) / σ_i
        z_i = su_g_inv(eps_i, λ)

        # --------------------------------------------------------------
        # Prequential log likelihood under SU family
        # --------------------------------------------------------------
        moments_i = {
            "μ": μ[i],
            "σ2": σ2[i],
        }
        log_preq_i = log_pdf(y_i, moments_i, λ)

        # --------------------------------------------------------------
        # Stable alpha computation
        # --------------------------------------------------------------
        αi_xx = jax.nn.sigmoid(logit_α_i + log_Sn_col)

        # --------------------------------------------------------------
        # Heavy-tailed K = 2 moment update
        # --------------------------------------------------------------
        R1_i, R2_i = su_R1_R2(z_i, ρ, λ)

        μ_next = μ + αi_xx * σ * R1_i

        σ2_next = σ2 * (
            1.0
            + αi_xx * (
                R2_i
                - 1.0
                - αi_xx * R1_i**2
            )
        )

        moments_next = {
            "μ": μ_next,
            "σ2": σ2_next,
        }

        model_i = {
            "z": z_i,
        }

        return moments_next, (model_i, log_preq_i)

    xs = (
        jnp.arange(n),
        y,
        logit_α,
        log_Sn.T,
    )

    _, (model, log_preqs) = jax.lax.scan(
        scan_body,
        moments,
        xs,
    )

    return model, jnp.mean(log_preqs)


# ---------------------------------------------------------------------
# Predict at test points
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnames=["unroll"])
def scan_moment_updates(
    moments_init,
    z_seq,
    log_S,
    logit_α,
    ρ=0.5,
    λ=λ,
    unroll=1,
):
    """
    Shared heavy-tailed K = 2 moment-recursion scan used by
    test_moments and pr_moments_single.
    """

    def scan_body(moments_carry, xs):
        μ = moments_carry["μ"]
        σ2 = moments_carry["σ2"]

        z_i, logit_α_i, log_S_col = xs

        αi_xx = jax.nn.sigmoid(logit_α_i + log_S_col)

        σ = jnp.sqrt(σ2)

        R1_i, R2_i = su_R1_R2(z_i, ρ, λ)

        μ_next = μ + αi_xx * σ * R1_i

        σ2_next = σ2 * (
            1.0
            + αi_xx * (
                R2_i
                - 1.0
                - αi_xx * R1_i**2
            )
        )

        moments_next = {
            "μ": μ_next,
            "σ2": σ2_next,
        }

        return moments_next, None

    xs = (
        z_seq,
        logit_α,
        log_S.T,
    )

    moments_final, _ = jax.lax.scan(
        scan_body,
        moments_init,
        xs,
        unroll=unroll,
    )

    return moments_final


@jit
def test_moments(model, log_S_test, ρ=0.5, λ=λ):
    n_test = log_S_test.shape[0]
    n = log_S_test.shape[1]

    moments_init = {
        "μ": jnp.zeros(n_test),
        "σ2": jnp.ones(n_test),
    }

    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    return scan_moment_updates(
        moments_init,
        model["z"],
        log_S_test,
        logit_α,
        ρ,
        λ,
        unroll=1,
    )


# ---------------------------------------------------------------------
# Permutation averaging
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnames=["n_perm"])
def fit_moments_perm(
    y,
    log_Sn,
    ρ=0.5,
    λ=λ,
    n_perm=10,
    seed=121,
):
    n = y.shape[0]
    key = random.PRNGKey(seed)

    keys = random.split(key, n_perm)
    ind_perm = jax.vmap(lambda k: random.permutation(k, n))(keys)

    def single_perm_fit(ind):
        y_p = y[ind]
        log_Sn_p = log_Sn[ind, :][:, ind]
        return fit_moments(
            y_p,
            log_Sn_p,
            ρ,
            λ,
        )

    model_perm, log_preq_perm = jax.vmap(single_perm_fit)(ind_perm)
    model_perm = {
        **model_perm,
        "ind_perm": ind_perm,
    }

    return model_perm, jnp.mean(log_preq_perm)


@jit
def test_moments_perm(
    model_perm,
    log_S_test,
    ρ=0.5,
    λ=λ,
):
    def single_perm_test(model):
        ind_p = model["ind_perm"]
        log_S_test_p = log_S_test[:, ind_p]
        return test_moments(
            model,
            log_S_test_p,
            ρ,
            λ,
        )

    moments_perm = jax.vmap(single_perm_test)(model_perm)

    μ_p = moments_perm["μ"]
    σ2_p = moments_perm["σ2"]

    # Mixture variance across permutations
    μ_final = jnp.mean(μ_p, axis=0)
    σ2_final = jnp.mean(σ2_p + μ_p**2, axis=0) - μ_final**2

    moments_averaged = {
        "μ": μ_final,
        "σ2": σ2_final,
    }

    return moments_averaged


# ---------------------------------------------------------------------
# Predictive resampling
# ---------------------------------------------------------------------

@jit
def pr_moments_single(
    moments_init,
    log_S_test,
    n,
    key,
    ρ=0.5,
    λ=λ,
):
    T = log_S_test.shape[1]

    i_ = jnp.arange(1, T + 1) + n
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    # Proposition: under predictive resampling, residuals Z_i are iid N(0,1).
    z_pr = random.normal(key, shape=(T,))

    moments_final = scan_moment_updates(
        moments_init,
        z_pr,
        log_S_test,
        logit_α,
        ρ,
        λ,
        unroll=16,
    )

    return moments_final


@partial(jax.jit, static_argnames=["B"])
def pr_moments(
    moments_init,
    log_S_test,
    n,
    ρ=0.5,
    λ=λ,
    B=10,
    seed=121,
):
    master_key = random.PRNGKey(seed)
    keys = random.split(master_key, B)

    vmap_fn = jax.vmap(
        pr_moments_single,
        in_axes=(None, None, None, 0, None, None),
    )

    moments_pr = vmap_fn(
        moments_init,
        log_S_test,
        n,
        keys,
        ρ,
        λ,
    )

    return moments_pr
