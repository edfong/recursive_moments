"""Multivariate Gaussian heteroscedastic recursive moments."""

import jax
import jax.numpy as jnp
import numpy as np

from .moment import RMMomentBase
from .mogauss_fun import (
    compute_quantile,
    covariance_from_cholesky,
    fit_moments,
    fit_moments_perm,
    log_pdf,
    pr_moments,
    test_moments_perm,
)
from .mogauss_missing_fun import (
    conditional_last_log_pdf,
    fit_moments as fit_moments_missing,
    fit_moments_perm as fit_moments_perm_missing,
    test_moments_perm as test_moments_perm_missing,
)
from .Sxx_kern import logSxx_gauss, logSxx_gauss_self


def correlation_from_covariance(covariance):
    """Convert covariance matrices to correlation matrices.

    Leading dimensions are preserved, so this supports both point estimates
    with shape ``(..., d, d)`` and posterior samples with shape
    ``(B, ..., d, d)``.
    """
    covariance = jnp.asarray(covariance)
    variance = jnp.diagonal(covariance, axis1=-2, axis2=-1)
    scale = jnp.sqrt(jnp.maximum(variance, 0.0))
    denominator = scale[..., :, None] * scale[..., None, :]
    denominator = jnp.maximum(
        denominator, jnp.finfo(covariance.dtype).tiny
    )
    correlation = jnp.clip(covariance / denominator, -1.0, 1.0)

    # Make the exact mathematical diagonal explicit after numerical clipping.
    diagonal = jnp.arange(covariance.shape[-1])
    return correlation.at[..., diagonal, diagonal].set(1.0)


class RMMOGauss(RMMomentBase):
    """Multivariate-output Gaussian recursive moments.

    ``y`` must have shape ``(n_samples, n_outputs)``.  
    Set ``missing=True`` to enable missing-outcome updates represented by
    ``np.nan``. Those updates support d=3 only, with the third outcome
    possibly missing and the first two always observed.
    """

    fit_fn = staticmethod(fit_moments_perm)
    test_fn = staticmethod(test_moments_perm)
    quantile_fn = staticmethod(compute_quantile)
    logpdf_fn = staticmethod(log_pdf)
    pr_fn = staticmethod(pr_moments)

    logS_self_fn = staticmethod(logSxx_gauss_self)
    logS_cross_fn = staticmethod(logSxx_gauss)

    def __init__(
        self,
        optimize=True,
        rho_x=None,
        rho_y=0.9,
        n_perm_opt=10,
        n_perm_fit=10,
        initial_correlation=0.,
        missing=False,
    ):
        super().__init__(
            optimize=optimize,
            rho_x=rho_x,
            rho_y=rho_y,
            n_perm_opt=n_perm_opt,
            n_perm_fit=n_perm_fit,
        )
        self.initial_correlation = initial_correlation
        self.missing = missing
        if self.missing:
            self.fit_fn = fit_moments_perm_missing
            self.test_fn = test_moments_perm_missing

    def _backend_kwargs(self):
        return {"initial_correlation": self.initial_correlation}

    def _test_backend(self, log_S_test):
        # Prediction starts from the fitted L_init.
        return self.test_fn(self.model_perm_, log_S_test, ρ=self.rho_y_)

    def _make_optimization_loss(self, X_std, y_std):
        # Pre-permute once when optimizing with a single permutation.
        if self.n_perm_opt != 1:
            return super()._make_optimization_loss(X_std, y_std)

        # Match fit_moments_perm's default seed and split, including for n_perm=1.
        key = jax.random.split(jax.random.PRNGKey(121), 1)[0]
        ind = jax.random.permutation(key, y_std.shape[0])
        X_perm, y_perm = X_std[ind], y_std[ind]
        fit_fn = fit_moments_missing if self.missing else fit_moments

        def loss(x):
            rho = 1.0 / (1.0 + jnp.exp(x))
            log_Sn = self.logS_self_fn(X_perm, ρx=rho[:-1])
            _, log_preq = fit_fn(
                y_perm, log_Sn, rho[-1], self.initial_correlation,
                block_size=512,
            )
            return -log_preq

        return loss

    def _validate_y(self, y):
        """Check only the supported missing-outcome patterns."""
        y = jnp.asarray(y)
        d = y.shape[1]
        missing = np.isnan(np.asarray(y))
        has_missing = np.any(missing)
        if has_missing and not self.missing:
            raise ValueError(
                "y contains NaN outcomes but RMMOGauss was initialized with "
                "missing=False. Set missing=True to enable missing-outcome "
                "updates."
            )
        if self.missing:
            if d != 3:
                raise ValueError(
                    "The missing-outcome RMMOGauss backend supports only d=3."
                )
            if np.any(missing[:, :2]):
                raise NotImplementedError(
                    "For d=3, missing-outcome updates are implemented only "
                    "when the third component is missing; the first two "
                    "components must be observed."
                )
        if has_missing:
            if np.any(np.all(missing, axis=0)):
                raise ValueError(
                    "Each outcome coordinate must have at least one observed "
                    "training value."
                )
        return y

    def _fit_standardizers(self, X, y):
        """Fit X scalers and coordinate-wise output scalers."""
        X_loc = jnp.mean(X, axis=0)
        X_scale = jnp.std(X, axis=0)
        self.X_loc_ = X_loc
        self.X_scale_ = jnp.where(X_scale > 1e-12, X_scale, 1.0)

        if self.missing:
            self.y_loc_ = jnp.nanmean(y, axis=0)
            y_scale = jnp.nanstd(y, axis=0)
        else:
            self.y_loc_ = jnp.mean(y, axis=0)
            y_scale = jnp.std(y, axis=0)
        self.y_scale_ = jnp.where(y_scale > 1e-12, y_scale, 1.0)

    def _unstandardize_moments(self, moments_std):
        out = dict(moments_std)
        if "μ" in out:
            out["μ"] = self.y_loc_ + self.y_scale_ * out["μ"]
        if "L" in out:
            # If y_raw = D y_std + loc, then L_raw = D L_std.
            out["L"] = self.y_scale_[:, None] * out["L"]
        if "Σ" in out:
            out["Σ"] = (
                self.y_scale_[:, None]
                * out["Σ"]
                * self.y_scale_[None, :]
            )
        return out

    def fit(self, X, y):
        X_2d = self._as_2d(X)
        y_2d = self._validate_y(y)
        self.d_ = y_2d.shape[1]

        X_std, y_std, log_Sn = self._fit_common_setup(X_2d, y_2d)
        model_perm, preq_loglik = self._fit_backend(y_std, log_Sn)

        self.model_perm_ = model_perm
        if self.missing:
            observed = ~jnp.isnan(y_2d)
            log_jacobian = jnp.mean(
                jnp.sum(
                    observed * jnp.log(self.y_scale_)[None, :], axis=1
                )
            )
        else:
            log_jacobian = jnp.sum(jnp.log(self.y_scale_))
        self.preq_loglik_ = preq_loglik - log_jacobian
        return self

    def predict(self, X_test, qs=(0.025, 0.975)):
        prediction = super().predict(X_test, qs=qs)
        prediction["Σ"] = covariance_from_cholesky(prediction["L"])
        self.last_moments_["Σ"] = prediction["Σ"]
        return prediction

    def test_log_pdf(self, X_test, y_test, conditional=False):
        """Return a joint or last-output conditional Gaussian log density per row.
        
         NaN test outputs are marginalized by retaining only
        the observed entries and their corresponding covariance submatrix.

        With conditional=True, score only the last output given all preceding
        outputs. Supply complete test triplets, including the held-out truth.
        """
        prediction = self.predict(X_test)
        mean = np.asarray(prediction["μ"])
        covariance = np.asarray(prediction["Σ"])
        y = np.asarray(y_test)

        if np.any(np.all(np.isnan(y), axis=1)):
            raise ValueError(
                "Every test row must contain at least one observed output."
            )

        if conditional:
            if np.any(np.isnan(y)):
                raise ValueError("Conditional scoring requires the target and all preceding outputs.")
            return conditional_last_log_pdf(jnp.asarray(y), prediction)

        # Evaluate the joint density using the full Cholesky factor.
        if not np.any(np.isnan(y)):
            return self.logpdf_fn(
                jnp.asarray(y),
                prediction,
                **self._backend_kwargs(),
            )

        # For partially observed test rows, explicitly marginalize the
        # missing coordinates before evaluating the Gaussian density.
        log_density = np.empty(y.shape[0], dtype=np.float64)
        for row in range(y.shape[0]):
            observed = np.isfinite(y[row])
            residual = y[row, observed] - mean[row, observed]
            covariance_row = covariance[row][np.ix_(observed, observed)]
            covariance_row = 0.5 * (
                covariance_row + covariance_row.T
            )
            cholesky = np.linalg.cholesky(covariance_row)
            standardized = np.linalg.solve(cholesky, residual)
            log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
            dimension = observed.sum()
            log_density[row] = -0.5 * (
                dimension * np.log(2.0 * np.pi)
                + log_determinant
                + standardized @ standardized
            )

        return jnp.asarray(log_density)

    def pr(
        self,
        X_test=None,
        X_pr=None,
        X_range=None,
        T=4000,
        B=1000,
        seed=126,
        kde_bw=None,
        X_pr_method="uniform",
        return_intervals=True,
        qs=(0.025, 0.975),
    ):
        moments_pr = super().pr(
            X_test=X_test,
            X_pr=X_pr,
            X_range=X_range,
            T=T,
            B=B,
            seed=seed,
            kde_bw=kde_bw,
            X_pr_method=X_pr_method,
            return_intervals=False,
            qs=qs,
        )
        moments_pr = dict(moments_pr)
        moments_pr["Σ"] = covariance_from_cholesky(moments_pr["L"])
        moments_pr["R"] = correlation_from_covariance(moments_pr["Σ"])
        self.last_moments_pr_ = moments_pr

        if return_intervals:
            return moments_pr, self.pr_intervals(moments_pr, qs=qs)
        return moments_pr

    def pr_intervals(self, moments_pr, qs=(0.025, 0.975)):
        intervals = {}
        for key in ("μ", "Σ", "R"):
            if key in moments_pr:
                intervals.update(self._quantile_summary(moments_pr[key], key, qs))
        return intervals
