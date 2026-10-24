###############################################################################
# Fit, score and time the scikit-learn Gaussian-process baseline.
###############################################################################

import time
import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)


def fit_gp(X, y, X_eval, cred_prob=0.95, n_restarts_optimizer=5, random_state=None):
    """Fit the GP comparator and summarize it at ``X_eval``.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    X_eval = np.asarray(X_eval)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(np.ones(X.shape[1]), (1e-3, 1e3))
        + WhiteKernel(max(1e-8, 0.1 * np.var(y)), (1e-8, 1e3))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    fit_start = time.perf_counter()
    gp.fit(X, y)
    fit_time = time.perf_counter() - fit_start

    post_start = time.perf_counter()
    mu_pred, y_sd = gp.predict(X_eval, return_std=True)
    # WhiteKernel noise is fitted on the normalized response scale.
    noise_var = float(gp.kernel_.k2.noise_level * gp._y_train_std**2)
    f_sd = np.sqrt(np.maximum(y_sd**2 - noise_var, 0.0))
    lower_prob = (1.0 - cred_prob) / 2.0
    z = norm.ppf(1.0 - lower_prob)
    # Estimate Var(Y | X=x) using fitted noise, excluding uncertainty in the mean.
    sigma2 = np.full_like(mu_pred, noise_var)

    moments = pd.DataFrame(
        {
            "μ_mean": mu_pred,
            "μ_med": mu_pred,
            "μ_l": mu_pred - z * f_sd,
            "μ_h": mu_pred + z * f_sd,
            "σ2_mean": sigma2,
            "σ2_med": sigma2,
            "σ2_l": sigma2,
            "σ2_h": sigma2,
        }
    )

    return {
        "model": gp,
        "moments": moments,
        "fit_time": fit_time,
        "postprocessing_time": time.perf_counter() - post_start,
        "noise_var": noise_var,
    }


def mean_test_logpdf_gp(gp, X_test, y_test):
    """Return the GP's mean posterior-predictive log density on test data."""
    mu_pred, y_sd = gp.predict(np.asarray(X_test), return_std=True)
    return float(np.mean(norm.logpdf(np.asarray(y_test).ravel(), mu_pred, y_sd)))
