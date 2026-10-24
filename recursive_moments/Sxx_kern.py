import jax
import jax.numpy as jnp

# --- Overall learning rate ---
@jax.jit
def compute_alpha(i): # i should be >= 1
    return  (2.0 - 1.0 / i) / (i + 1.0)**0.9

# --- Continuous ---
@jax.jit
def logSxx_gauss(Xtest, Xn, ρx):
    """
    Assumes standardized covariates
    """
    ρx2 = ρx**2
    one_minus_ρ2 = 1.0 - ρx2
    
    C = jnp.sum(-0.5 * jnp.log(one_minus_ρ2))
    B = -0.5 * ρx2 / one_minus_ρ2
    
    v1 = (Xtest**2) @ B
    v2 = (Xn**2) @ B
    
    W = ρx / one_minus_ρ2
    cross_term = (Xtest * W) @ Xn.T  # Avoids instantiating an explicit Diagonal matrix
    
    return C + v1[:, None] + v2[None, :] + cross_term

@jax.jit
def logSxx_gauss_self(Xn, ρx):
    """
    Assumes standardized covariates; this version is faster for calculating similarity with self.
    """
    ρx2 = ρx**2
    one_minus_ρ2 = 1.0 - ρx2
    
    C = jnp.sum(-0.5 * jnp.log(one_minus_ρ2))
    B_sq = -0.5 * ρx2 / one_minus_ρ2
    
    v = (Xn**2) @ B_sq
    
    W = ρx / one_minus_ρ2
    cross_term = (Xn * W) @ Xn.T
    
    return C + v[:, None] + v[None, :] + cross_term
