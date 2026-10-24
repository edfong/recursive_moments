import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from .Sxx_kern import compute_alpha

# --- Utilities ---
@jit
def log_pdf(y,moments):
    return  -0.5*jnp.log(2*jnp.pi*moments['σ2']) - (0.5/moments['σ2'])*(y - moments["μ"])**2

@jit
def compute_quantile(moments, α = 0.5):
    return moments["μ"] + jnp.sqrt(moments['σ2']) * jax.scipy.special.ndtri(α)


# --- Fit model ---
@jit
def fit_moments(y, log_Sn, ρ=0.5):
    n = y.shape[0] 
    
    # Initialize moments as a single dictionary
    moments = {
        'μ': jnp.zeros(n),
        'σ2': jnp.ones(n)
    }
    # Precompute Constants
    ρ2 = ρ**2
    half_log_2pi = 0.5 * jnp.log(2 * jnp.pi)
    
    # Precompute alpha_i
    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)
    
    # Main loop
    def scan_body(moments_carry, xs):
        # Unpack the current moments and intermediates
        μ = moments_carry['μ']
        σ2 = moments_carry['σ2']

        i, y_i, logit_α_i,log_Sn_col  = xs
        
        # Compute residuals
        σ = jnp.sqrt(σ2)
        σ_i = σ[i]
        z_i = (y_i - μ[i]) / σ_i
        z_i2 = z_i**2

        # Update prequential log likelihood with previous estimates
        log_preq_i = -0.5 * z_i2 - half_log_2pi - 0.5 * jnp.log(σ2[i])

        # Updating      
        ## Stable alpha computation
        αi_xx = jax.nn.sigmoid(logit_α_i + log_Sn_col)
        
        ## Core updates
        ρzi = ρ * z_i
        μ_next = μ + ρzi * (αi_xx * σ)
        σ2_next = σ2 * (1.0 + ρ2 * αi_xx * ((1.0 - αi_xx) * z_i2 - 1.0))
    
        # Pack the updated variables back into the moments/model structure
        moments_next = {
            'μ': μ_next,
            'σ2': σ2_next
        }

        model_i = {
            'z': z_i
        }
        
        return moments_next, (model_i, log_preq_i)
    
    # Scan
    xs = (jnp.arange(n),y, logit_α, log_Sn.T)
    _, (model, log_preqs) = jax.lax.scan(scan_body, moments, xs)
    
    return model, jnp.mean(log_preqs)

# --- Predict at test points ---
@partial(jax.jit, static_argnames=["unroll"])
def scan_moment_updates(moments_init, z_seq, log_S, logit_α, ρ=0.5, unroll=1):
    """
    Shared moment-recursion scan used by test_moments and pr_moments_single.
    """
    ρ2 = ρ**2

    def scan_body(moments_carry, xs):
        μ = moments_carry["μ"]
        σ2 = moments_carry["σ2"]

        z_i, logit_α_i, log_S_col = xs

        αi_xx = jax.nn.sigmoid(logit_α_i + log_S_col)

        σ = jnp.sqrt(σ2)
        z_i2 = z_i**2
        ρzi = ρ * z_i

        μ_next = μ + ρzi * (αi_xx * σ)
        σ2_next = σ2 * (
            1.0 + ρ2 * αi_xx * ((1.0 - αi_xx) * z_i2 - 1.0)
        )

        moments_next = {
            "μ": μ_next,
            "σ2": σ2_next,
        }

        return moments_next, None

    xs = (z_seq, logit_α, log_S.T)

    moments_final, _ = jax.lax.scan(
        scan_body,
        moments_init,
        xs,
        unroll=unroll,
    )

    return moments_final

@jit
def test_moments(model, log_S_test, ρ=0.5):
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
        unroll=1,
    )


# --- Permutation averaging --- 
## Fit permutation
@partial(jax.jit, static_argnames=['n_perm'])
def fit_moments_perm(y, log_Sn, ρ=0.5, n_perm=10, seed=121):
    n = y.shape[0]
    key = random.PRNGKey(seed)
    
    # Generate random permutation indices
    keys = random.split(key, n_perm)
    ind_perm = jax.vmap(lambda k: random.permutation(k, n))(keys) 
    
    # Single permutation row
    def single_perm_fit(ind):
        y_p = y[ind]
        log_Sn_p = log_Sn[ind, :][:, ind] 
        return fit_moments(y_p, log_Sn_p, ρ)
    
    # Vmap across permutations
    model_perm, log_preq_perm = jax.vmap(single_perm_fit)(ind_perm)
    model_perm = {**model_perm, 'ind_perm': ind_perm}

    return model_perm, jnp.mean(log_preq_perm)

## Test permutation
@jit
def test_moments_perm(model_perm, log_S_test, ρ=0.5):
    # Simple permutation
    def single_perm_test(model):
        ind_p = model['ind_perm']
        log_S_test_p = log_S_test[:, ind_p]
        return test_moments(model, log_S_test_p, ρ)
    
    # Vmap
    moments_perm = jax.vmap(single_perm_test)(model_perm)
    
    # Extract permutations for averaging
    μ_p = moments_perm['μ']
    σ2_p = moments_perm['σ2']

    # Average across permutations
    μ_final = jnp.mean(μ_p, axis=0)
    σ2_final = jnp.mean(σ2_p + μ_p**2, axis=0) - μ_final**2 
    
    # Package into the moments dictionary
    moments_averaged = {
        'μ': μ_final,
        'σ2': σ2_final
    }

    return moments_averaged


# --- Predictive resampling ---
@jit
def pr_moments_single(moments_init, log_S_test, n, key, ρ=0.5):
    T = log_S_test.shape[1]

    i_ = jnp.arange(1, T + 1) + n
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    z_pr = random.normal(key, shape=(T,))

    return scan_moment_updates(
        moments_init,
        z_pr,
        log_S_test,
        logit_α,
        ρ,
        unroll=16,
    )

## Vmap over seeds
@partial(jax.jit, static_argnames=[ 'B'])
def pr_moments(moments_init, log_S_test,n, ρ=0.5, B=10, seed=121):
    # Generate B keys
    master_key = random.PRNGKey(seed)
    keys = random.split(master_key, B)
    
    # Setup vmap and run
    vmap_fn = jax.vmap(pr_moments_single, in_axes=(None, None,None, 0, None))
    moments_pr = vmap_fn(moments_init, log_S_test,n, keys, ρ)
    
    return moments_pr



