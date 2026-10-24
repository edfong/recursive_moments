import jax
import jax.numpy as jnp
from jax import custom_jvp, jit, vmap, random, lax
from jax.scipy.stats import norm
from jax.scipy.special import ndtri, logsumexp
from functools import partial
from .Sxx_kern import compute_alpha

# --- Custom JVPs for faster gradients ---
@custom_jvp 
def ndtri_(u):
    return ndtri(u)

@ndtri_.defjvp
def ndtri_jvp(primals, tangents):
    u, = primals
    u_dot, = tangents
    primal_out = ndtri_(u)
    tangent_out = (1.0 / norm.pdf(primal_out)) * u_dot
    return primal_out, tangent_out

@custom_jvp
def norm_logcdf(z):
    return norm.logcdf(z)

@norm_logcdf.defjvp
def norm_logcdf_jvp(primals, tangents):
    z, = primals
    z_dot, = tangents
    primal_out = norm_logcdf(z)
    tangent_out = jnp.exp(norm.logpdf(z) - primal_out) * z_dot
    return primal_out, tangent_out

# --- Copula functions ---
@jit
def norm_copula_logdistribution_logdensity(u, v, ρ):   
    # Copula density is not numerically stable for small u, needs clipping 
    eps = 1e-6
    u = jnp.clip(u, eps, 1.0 - eps) 
    v = jnp.clip(v, eps, 1.0 - eps)

    pu = ndtri_(u)
    pv = ndtri_(v)
    z = (pu - ρ * pv) / jnp.sqrt(1.0 - ρ**2)

    logcop_dist = norm_logcdf(z)
    logcop_dist = jnp.clip(logcop_dist, jnp.log(eps), jnp.log(1.0 - eps))
    logcop_dens = -0.5 * jnp.log(1.0 - ρ**2) + (0.5 / (1.0 - ρ**2)) * (-(ρ**2) * (pu**2 + pv**2) + 2.0 * ρ * pu * pv)

    return logcop_dist, logcop_dens

# --- Update functions ---
@jit
def init_marginals(y): # Assumes standardized data
    mean0, std0 = 0.0, 1.0
    logcdf = norm.logcdf(y, loc=mean0, scale=std0)
    logpdf = norm.logpdf(y, loc=mean0, scale=std0) 
    return logcdf, logpdf

@jit
def update_copula_batch(logcdf_conditionals, logpdf_joints, u, v, logalpha, ρ): 
    # Precompute
    logcop_dist, logcop_dens = norm_copula_logdistribution_logdensity(u, v, ρ)
    log1alpha = jnp.log1p(-jnp.exp(logalpha))

    # Update density and CDF
    logcdf_next = jnp.logaddexp(log1alpha + logcdf_conditionals, logalpha + logcop_dist)
    logpdf_next = jnp.logaddexp(log1alpha, logalpha + logcop_dens) + logpdf_joints     

    return logcdf_next, logpdf_next

# --- Fit model ---
@jit
def fit_copula(y, log_Sn, ρ):
    n = y.shape[0]

    # Initialize
    logcdf_conditionals, logpdf_joints = init_marginals(y)
    
    # Precompute logalpha_x matrix outside loop
    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    # Main loop
    def scan_body(carry, xs_i):

        # Unpack carries
        logcdf_cond, logpdf_joint = carry
        i, logit_α_i,log_Sn_col = xs_i

        # Compute alphaxx stably
        log_αi_xx = jax.nn.log_sigmoid(logit_α_i + log_Sn_col)
        
        # Compute CDF values
        u = jnp.exp(logcdf_cond)
        v = jnp.exp(logcdf_cond[i])  
        
        # Prequential loglik
        preq_loglik_step = logpdf_joint[i]
        
        # Main update
        logcdf_next, logpdf_next = update_copula_batch(
            logcdf_cond, logpdf_joint, u, v, log_αi_xx, ρ
        )
        
        return (logcdf_next, logpdf_next), (v, preq_loglik_step)

    # Scan
    xs = (jnp.arange(n), logit_α, log_Sn.T)
    final_state, (vn_history, preq_loglik) = lax.scan(
        scan_body, 
        (logcdf_conditionals, logpdf_joints), 
        xs
    )
    
    model = {'vn': vn_history}

    return model, jnp.mean(preq_loglik)

# --- Predict at test point ---
@jit
def scan_copula_updates(log_pdf_cdf_init, v_seq, log_S, logit_α, ρ=0.5):
    logcdf0 = log_pdf_cdf_init["log_cdf"]
    logpdf0 = log_pdf_cdf_init["log_pdf"]

    def scan_body(carry, xs_i):
        logcdf_cond, logpdf_joint = carry
        v_i, logit_α_i, log_S_col = xs_i

        log_αi_xx = jax.nn.log_sigmoid(logit_α_i + log_S_col)
        u = jnp.exp(logcdf_cond)

        logcdf_next, logpdf_next = update_copula_batch(
            logcdf_cond,
            logpdf_joint,
            u,
            v_i,
            log_αi_xx,
            ρ,
        )

        return (logcdf_next, logpdf_next), None

    xs = (v_seq, logit_α, log_S.T)

    (final_logcdf, final_logpdf), _ = lax.scan(
        scan_body,
        (logcdf0, logpdf0),
        xs,
    )

    return {
        "log_pdf": final_logpdf,
        "log_cdf": final_logcdf,
    }

@jit
def test_copula(y_test, model, log_S_test, ρ):
    vn = model["vn"]
    n = vn.shape[0]

    logcdf0, logpdf0 = init_marginals(y_test)

    log_pdf_cdf_init = {
        "log_cdf": logcdf0,
        "log_pdf": logpdf0,
    }

    i_ = jnp.arange(1, n + 1)
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    return scan_copula_updates(
        log_pdf_cdf_init,
        vn,
        log_S_test,
        logit_α,
        ρ)

# --- Permutation averaging ---
## Fit
@partial(jax.jit, static_argnames=['n_perm'])
def fit_copula_perm(y, log_Sn, ρ, n_perm=10, key=None):
    if key is None:
        key = random.PRNGKey(121)
    n = y.shape[0]
    keys = random.split(key, n_perm)
    ind_perm = jax.vmap(lambda k: random.permutation(k, n))(keys) 

    
    # Single function
    def single_perm_fit(ind):
        y_p = y[ind]
        log_Sn_p = log_Sn[ind, :][:, ind] 
        
        return fit_copula(y_p, log_Sn_p, ρ)
    
    # Vmap
    model_perm, loglik_perm = vmap(single_perm_fit)(ind_perm)

    # Append with indices
    model_perm = {**model_perm, 'ind_perm': ind_perm}

    return model_perm, jnp.mean(loglik_perm) 

## Test
@jit
def test_copula_perm(y_test, model_perm, log_S_test, ρ):
    def single_perm_test(model_perm):
        log_S_test_perm = log_S_test[:, model_perm['ind_perm']]
        return test_copula(y_test, model_perm, log_S_test_perm, ρ)
        
    # Vmap across permutations
    log_pdf_cdf = vmap(single_perm_test)(model_perm)
    n_perm = log_pdf_cdf['log_pdf'].shape[0]
    
    # Average across permutations
    logcdf_avg = logsumexp(log_pdf_cdf['log_cdf'], axis=0) - jnp.log(n_perm)
    logpdf_avg = logsumexp(log_pdf_cdf['log_pdf'], axis=0) - jnp.log(n_perm)
    
    log_pdf_cdf_avg = {
            'log_pdf': logpdf_avg,
            'log_cdf': logcdf_avg
        }

    return log_pdf_cdf_avg

# --- Predictive resampling ---
@jit
def pr_np_single(log_pdf_cdf_init, log_S_test, n, key, ρ=0.5):
    T = log_S_test.shape[1]

    i_ = jnp.arange(1, T + 1) + n
    α = compute_alpha(i_)
    logit_α = jnp.log(α) - jnp.log1p(-α)

    v_pr = random.uniform(key, shape=(T,))

    return scan_copula_updates(
        log_pdf_cdf_init,
        v_pr,
        log_S_test,
        logit_α,
        ρ)

# Vmap over seeds
@partial(jax.jit, static_argnames=['B'])
def pr_np(log_pdf_cdf_init, log_S_test, n, ρ=0.5, B=10, seed=121):
    # Generate B keys
    master_key = random.PRNGKey(seed)
    keys = random.split(master_key, B)
    
    # Vmap
    vmap_fn = jax.vmap(pr_np_single, in_axes=(None, None, None, 0, None))
    log_pdf_cdf_B = vmap_fn(log_pdf_cdf_init, log_S_test, n, keys, ρ)
    
    return log_pdf_cdf_B
