"""Simulation data, ground truths, scoring and timing utilities."""

import time

import jax
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import RegularGridInterpolator

# ------------------------------------------------------------------
# Drawing samples
# ------------------------------------------------------------------

# Heterogeneous example
def sample_y_het(X):
    """
    Generate y from heterogeneous example.
    """
    X = X[:,0]
    y = sp.stats.norm.rvs(loc = 5/(1+np.exp(-X)), scale = 0.25 + np.exp((X - 6)/3) )
    y = np.array(y)
    return y


# Heavy-tailed example

HUBER_EPS = 0.05   # contamination probability
HUBER_C = 10.0     # contaminated component scale multiplier


def huber_std_cdf(z, eps=HUBER_EPS, c=HUBER_C):
    """
    CDF of standardized Huber contamination:
        (1-eps) N(0,1) + eps N(0,c^2).
    """
    return (
        (1.0 - eps) * sp.stats.norm.cdf(z)
        + eps * sp.stats.norm.cdf(z / c)
    )


def huber_std_ppf(q, eps=HUBER_EPS, c=HUBER_C):
    """
    Quantile of standardized Huber contamination:
        (1-eps) N(0,1) + eps N(0,c^2).

    No closed form, so solve by root finding.
    """
    root_fun = lambda z: huber_std_cdf(z, eps=eps, c=c) - q

    # Wide bracket because contaminated component may be very diffuse.
    lower = -100.0 * c
    upper = 100.0 * c

    return sp.optimize.brentq(root_fun, lower, upper)

def sample_y_kurt(X):
    """
    Generate y from the Huber-contamination heavy-tailed example.

    Conditional distribution:
        Y | X=x ~ (1-eps) N(mu(x), scale(x)^2)
                 + eps N(mu(x), (c * scale(x))^2)
    """
    X = X[:, 0]

    loc_plot = 5.0 / (1.0 + np.exp(-X))
    scale_plot = 0.25 + np.exp((X - 6.0) / 3.0)

    # Draw contamination indicators
    is_contam = np.random.binomial(
        n=1,
        p=HUBER_EPS,
        size=X.shape
    ).astype(bool)

    # Effective scale for each observation
    scale_eff = scale_plot * np.where(is_contam, HUBER_C, 1.0)

    y = sp.stats.norm.rvs(
        loc=loc_plot,
        scale=scale_eff
    )

    return np.asarray(y)

# Same covariate distribution
def sample_X_het(n):
    X = np.random.uniform(-7,7,size = (n,1))
    return(X)
    

def sample_X_kurt(n):
    return(sample_X_het(n))


# Bivariate heteroscedastic Gaussian example
def _moments_mohet(X):
    """Return the conditional mean, marginal scales, and correlation."""
    x = np.asarray(X)[:, 0]

    mean = np.column_stack(
        (
            5.0 / (1.0 + np.exp(-x)),
            5.0 / (1.0 + np.exp(x)),
        )
    )
    scale = np.column_stack(
        (
            0.25 + np.exp((x - 6.0) / 3.0),
            0.25 + np.exp((-x - 6.0) / 3.0),
        )
    )
    corr = 0.8 * np.tanh(x / 2.0)

    return mean, scale, corr


def sample_y_mohet(X):
    """Generate the bivariate heteroscedastic Gaussian response."""
    mean, scale, corr = _moments_mohet(X)
    z = np.random.normal(size=mean.shape)

    y = np.empty_like(mean)
    y[:, 0] = mean[:, 0] + scale[:, 0] * z[:, 0]
    y[:, 1] = mean[:, 1] + scale[:, 1] * (
        corr * z[:, 0] + np.sqrt(1.0 - corr**2) * z[:, 1]
    )

    return y


def sample_X_mohet(n):
    """Generate the univariate covariate for the bivariate example."""
    return sample_X_het(n)

# ------------------------------------------------------------------
# Computing true values
# ------------------------------------------------------------------

def pred_truth_het(X, qs = (0.025,0.975)):
    X = X[:,0]
    loc_plot = 5/(1+np.exp(-X)) 
    scale_plot = (0.25+np.exp(((X - 6)/3)))
    mean_truth = loc_plot
    var_truth = scale_plot**2
    y_l_truth = sp.stats.norm.ppf(
        qs[0],
        loc=loc_plot,
        scale=scale_plot
    )

    y_h_truth = sp.stats.norm.ppf(
        qs[1],
        loc=loc_plot,
        scale=scale_plot
    )

    return mean_truth, var_truth, y_l_truth, y_h_truth


def pred_truth_mohet(X, qs=(0.025, 0.975), return_dependence=False):
    """True marginal moments and dependence for the bivariate example.
    """
    mean_truth, scale_truth, corr_truth = _moments_mohet(X)
    var_truth = scale_truth**2

    y_l_truth = sp.stats.norm.ppf(
        qs[0], loc=mean_truth, scale=scale_truth
    )
    y_h_truth = sp.stats.norm.ppf(
        qs[1], loc=mean_truth, scale=scale_truth
    )

    cov12_truth = corr_truth * scale_truth[:, 0] * scale_truth[:, 1]
    cov_truth = np.empty((mean_truth.shape[0], 2, 2), dtype=float)
    cov_truth[:, 0, 0] = var_truth[:, 0]
    cov_truth[:, 1, 1] = var_truth[:, 1]
    cov_truth[:, 0, 1] = cov12_truth
    cov_truth[:, 1, 0] = cov12_truth

    result = (
        mean_truth,
        var_truth,
        y_l_truth,
        y_h_truth,
    )
    if return_dependence:
        return result + (cov_truth, corr_truth)
    return result




def pred_truth_kurt(X, qs=(0.025, 0.975)):
    """
    True conditional mean, variance, and quantiles for the
    Huber-contamination heavy-tailed example.

    Conditional distribution:
        Y | X=x ~ (1-eps) N(mu(x), scale(x)^2)
                 + eps N(mu(x), (c * scale(x))^2)
    """
    X = X[:, 0]

    eps = HUBER_EPS
    c = HUBER_C

    loc_plot = 5.0 / (1.0 + np.exp(-X))
    scale_plot = 0.25 + np.exp((X - 6.0) / 3.0)

    # Mean: same for both mixture components
    mean_truth = loc_plot

    # Variance:
    # Var = scale^2 * [(1-eps) + eps*c^2]
    var_factor = (1.0 - eps) + eps * c**2
    var_truth = scale_plot**2 * var_factor

    # Quantiles:
    # If Z ~ (1-eps)N(0,1) + eps N(0,c^2),
    # then Y = loc + scale * Z.
    z_l = huber_std_ppf(qs[0], eps=eps, c=c)
    z_h = huber_std_ppf(qs[1], eps=eps, c=c)

    y_l_truth = loc_plot + scale_plot * z_l
    y_h_truth = loc_plot + scale_plot * z_h

    return mean_truth, var_truth, y_l_truth, y_h_truth


# ------------------------------------------------------------------
# Extension of Example 3 in Wade and Inácio (2025)
# with 3 spurious covariates
# ------------------------------------------------------------------

def _mean_mv(X):
    """
    Mean function for the multivariate covariate simulation.

    Active covariates:
        X[:, 0] = x1
        X[:, 1] = x2

    Spurious covariates:
        X[:, 2], X[:, 3], X[:, 4]
    have zero effect.
    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    signal = np.sin(x1 * x2 * np.pi / 2.0)

    mu = np.where(signal <= 0.0, 1.0, -1.0)

    return mu


def _scale_mv(X):
    """
    Conditional standard deviation: 0.1 or 1, depending on x1 and x2.
    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    signal = np.sin(x1 * x2 * np.pi / 2.0)
    scale = np.where(signal <= 0.0, 0.1, 1.)

    return scale


def sample_X_mv(n):
    """
    Generate covariates for the multivariate simulation with 3 spurious covariates.

    Columns:
        X[:, 0] = active x1 ~ U(-2, 2)
        X[:, 1] = active x2 ~ U(-2, 2)
        X[:, 2] = spurious ~ U(-2, 2)
        X[:, 3] = spurious ~ U(-2, 2)
        X[:, 4] = spurious ~ U(-2, 2)
    """
    X = np.random.uniform(-2.0, 2.0, size=(n, 5))
    return X


def sample_y_mv(X):
    """
    Generate y for the multivariate simulation with 3 spurious covariates.

    Only X[:, 0] and X[:, 1] affect the mean.
    """
    mu = _mean_mv(X)
    scale = _scale_mv(X)

    y = sp.stats.norm.rvs(
        loc=mu,
        scale=scale
    )

    return np.asarray(y)


def pred_truth_mv(X, qs=(0.025, 0.975)):
    """
    True conditional mean, variance, and quantiles for the multivariate simulation
    with 3 spurious covariates.
    """
    mu = _mean_mv(X)
    scale = _scale_mv(X)

    mean_truth = mu
    var_truth = scale**2

    y_l_truth = sp.stats.norm.ppf(
        qs[0],
        loc=mean_truth,
        scale=scale
    )

    y_h_truth = sp.stats.norm.ppf(
        qs[1],
        loc=mean_truth,
        scale=scale
    )

    return mean_truth, var_truth, y_l_truth, y_h_truth


# ------------------------------------------------------------------
# Utility functions for evaluating models
# ------------------------------------------------------------------

def test_logpdf_grid(
    X_test,
    y_test,
    x_plot,
    y_plot,
    pdf_NW
):
    """
    Average test log pdf by interpolating pdf_NW over (x_plot, y_plot).

    Args:
        X_test: shape (n_test,) or (n_test, 1)
        y_test: shape (n_test,)
        x_plot: shape (n_x,)
        y_plot: shape (n_y,)
        pdf_NW: shape (n_x, n_y)
    Returns:
        avg_logpdf
    """

    interp = RegularGridInterpolator(
        points=(x_plot, y_plot),
        values=pdf_NW,
        method="nearest",
        bounds_error = False,
        fill_value = None
    )

    pts = np.column_stack([X_test, y_test])

    pdf_test = interp(pts)

    logpdf_test = np.log(np.clip(pdf_test,a_min = 1e-12,a_max = np.inf))
    avg_logpdf = np.mean(logpdf_test)

    return avg_logpdf


def MSE(intervals,key,truth):
    """
    Compute MSE of posterior mean with true value
    """
    val = np.mean((intervals[f"{key}_mean"]- truth)**2)
    return float(val)

def MSE_med(intervals,key,truth):
    """
    Compute MSE of posterior median with true value
    """
    val = np.mean((intervals[f"{key}_med"]- truth)**2)
    return float(val)


def coverage(intervals, key, truth):
    """
    Compute avg coverage
    """
    val = np.mean(
        np.asarray(intervals[f"{key}_l"] <= truth)
        & np.asarray(intervals[f"{key}_h"] >= truth)
    )
    return float(val)


def interval_length(intervals, key):
    """
    Compute avg interval
    """
    val = np.mean(
        intervals[f"{key}_h"]-intervals[f"{key}_l"]
    )
    return float(val)



# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def block_tree(x):
    """Block until JAX computation is ready for arrays/pytrees."""
    return jax.tree_util.tree_map(
        lambda z: z.block_until_ready() if hasattr(z, "block_until_ready") else z,
        x,
    )


def time_call(fn):
    """Return (result, elapsed_seconds), blocking on JAX arrays."""
    t0 = time.perf_counter()
    out = fn()
    block_tree(out)
    t1 = time.perf_counter()
    return out, t1 - t0


def mean_test_logpdf(model, X_test, y_test):
    val = np.mean(model.test_log_pdf(X_test, y_test))
    val.block_until_ready()
    return float(val)


def summarize_results(df, id_cols=("seed", "Method")):
    """
    Average over seeds and compute standard errors.

    Expects df to have columns:
        seed, Method, metric columns...
    """
    value_cols = [c for c in df.columns if c not in id_cols]

    mean_df = (
        df.groupby("Method")[value_cols]
        .mean()
        .reset_index()
    )

    se_df = (
        df.groupby("Method")[value_cols]
        .sem()
        .reset_index()
    )

    # Pair mean columns with their standard errors, e.g. μ and μ_se.
    out = mean_df.copy()
    for col in value_cols:
        out[f"{col}_se"] = se_df[col]

    return out


def format_mean_se_table(summary_df, value_cols, method_col="Method", mean_fmt="{:.3g}", se_fmt="{:.2g}"):
    """
    Convert summary table with columns col and col_se into display table
    with entries 'mean (se)'.
    """
    out = pd.DataFrame()
    labels = {"Gauss": "RM (Gauss)", "SU": "RM (SU)"}
    out[method_col] = summary_df[method_col].map(lambda name: labels.get(name, name))

    for col in value_cols:
        out[col] = [
            f"{mean_fmt.format(m)} ({se_fmt.format(s)})"
            for m, s in zip(summary_df[col], summary_df[f"{col}_se"])
        ]

    return out
