import numpy as np
import jax.numpy as jnp

from .base import RMCommonBase

class RMDensityBase(RMCommonBase):
    """
    Base class for nonparametric copula methods.

    Subclasses should define:

        optimize_fn
        fit_fn
        test_fn
        pr_fn
        logS_self_fn
        logS_cross_fn
    """

    fit_fn = None
    test_fn = None
    pr_fn = None

    def __init__(
        self,
        optimize=True,
        rho_x=None,
        rho_y=0.9,
        n_perm_opt=10,
        n_perm_fit=10,
    ):
        super().__init__(
            optimize=optimize,
            rho_x=rho_x,
            rho_y=rho_y,
            n_perm_opt=n_perm_opt,
        )

        self.n_perm_fit = n_perm_fit
 
    # ------------------------------------------------------------------
    # Backend wrappers
    # ------------------------------------------------------------------

    def _fit_backend(self, y_std, log_Sn):
        return self.fit_fn(
            y_std,
            log_Sn,
            ρ=self.rho_y_,
            n_perm=self.n_perm_fit
        )

    def _test_backend(self, y_rep_std, log_S_test_rep):
        return self.test_fn(
            y_rep_std,
            self.model_perm_,
            log_S_test_rep,
            self.rho_y_
        )

    def _pr_backend(self, log_pdf_cdf_std, log_S_test_pr_rep, B, seed):

        return self.pr_fn(
            log_pdf_cdf_std,
            log_S_test_pr_rep,
            self.n_,
            ρ=self.rho_y_,
            B=B,
            seed=seed
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit density model on standardized X and y.
        """
        X_std, y_std, log_Sn = self._fit_common_setup(X, y)

        model_perm, preq_loglik = self._fit_backend(y_std, log_Sn)

        self.model_perm_ = model_perm
        self.preq_loglik_ = preq_loglik - jnp.log(self.y_scale_) # Return prequential log likelihood on unstandardized scale

        return self

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _prepare_xy_grid(self, X_test, y_grid):
        """
        Given raw-scale X_test and raw-scale y_grid, construct standardized
        objects for backend density evaluation.
        """
        X_test_raw = self._as_2d(X_test)
        X_test_std = self._standardize_X(X_test_raw)

        y_grid_raw = jnp.asarray(y_grid)
        if y_grid_raw.ndim != 1:
            raise ValueError("y_grid must be one-dimensional.")

        y_grid_std = self._standardize_y(y_grid_raw)

        log_S_test = self._logS_cross_std(X_test_std, self.X_)

        n_x = X_test_raw.shape[0]
        n_y = y_grid_raw.shape[0]

        y_rep_std = jnp.tile(y_grid_std, n_x)
        log_S_test_rep = jnp.repeat(log_S_test, n_y, axis=0)

        return (
            X_test_raw,
            X_test_std,
            y_grid_raw,
            y_grid_std,
            log_S_test,
            y_rep_std,
            log_S_test_rep,
        )

    def _reshape_log_pdf_cdf(self, log_pdf_cdf, n_x, n_y):
        """
        Reshape flat outputs into grid arrays.

        Expected flat ordering is x-major:
            x_0 with all y_grid,
            x_1 with all y_grid,
            ...
        """
        out = dict(log_pdf_cdf)

        if "log_pdf" in out:
            out["log_pdf_grid"] = out["log_pdf"].reshape(n_x, n_y)
            out["pdf_grid"] = jnp.exp(out["log_pdf_grid"])

        if "log_cdf" in out:
            out["log_cdf_grid"] = out["log_cdf"].reshape(n_x, n_y)
            out["cdf_grid"] = jnp.exp(out["log_cdf_grid"])

        return out

    def _reshape_log_pdf_cdf_pr(self, log_pdf_cdf_pr, n_x, n_y):
        """
        Reshape PR outputs.

        Expected flat shape:
            log_pdf: (B, n_x * n_y)
            log_cdf: (B, n_x * n_y)
        """
        out = dict(log_pdf_cdf_pr)

        if "log_pdf" in out:
            B_actual = out["log_pdf"].shape[0]
            out["log_pdf_grid"] = out["log_pdf"].reshape(B_actual, n_x, n_y)
            out["pdf_grid"] = jnp.exp(out["log_pdf_grid"])

        if "log_cdf" in out:
            B_actual = out["log_cdf"].shape[0]
            out["log_cdf_grid"] = out["log_cdf"].reshape(B_actual, n_x, n_y)
            out["cdf_grid"] = jnp.exp(out["log_cdf_grid"])

        return out

    # ------------------------------------------------------------------
    # Numerical summaries from density grid
    # ------------------------------------------------------------------

    @staticmethod
    def density_grid_summaries(pdf_grid, cdf_grid, y_grid, qs=(0.025, 0.975)):
        pdf_np = np.asarray(pdf_grid)
        cdf_np = np.asarray(cdf_grid)
        y_np = np.asarray(y_grid)

        n_x, n_y = pdf_np.shape

        dy = float(np.mean(np.diff(y_np)))

        q_low, q_high = qs

        mean = np.sum(pdf_np * y_np[None, :] * dy, axis=1)
        var = np.sum(pdf_np * (y_np[None, :] - mean[:, None]) ** 2 * dy, axis=1)
        y_l = np.zeros(n_x)
        y_h = np.zeros(n_x)

        for j in range(n_x):
            idx_l = np.searchsorted(cdf_np[j], q_low) - 1
            idx_h = np.searchsorted(cdf_np[j], q_high) - 1

            idx_l = np.clip(idx_l, 0, n_y - 1)
            idx_h = np.clip(idx_h, 0, n_y - 1)

            y_l[j] = y_np[idx_l]
            y_h[j] = y_np[idx_h]

        return {
            "μ": mean,
            "σ2": var,
            "y_l": y_l,
            "y_h": y_h,
        }

    # ------------------------------------------------------------------
    # Predict density/CDF on grid
    # ------------------------------------------------------------------

    def predict(
        self,
        X_test,
        y_grid,
        return_summaries=True,
        qs=(0.025, 0.975),
    ):
        """
        Predict density/CDF over an X_test by y_grid mesh.

        Inputs X_test and y_grid are on the raw/original scale.
        Outputs are also on the raw/original y scale.
        """
        self._check_is_fitted()

        (
            X_test_raw,
            X_test_std,
            y_grid_raw,
            y_grid_std,
            log_S_test,
            y_rep_std,
            log_S_test_rep,
        ) = self._prepare_xy_grid(X_test, y_grid)

        # Backend returns standardized-y log_pdf/log_cdf.
        log_pdf_cdf_std_flat = self._test_backend(y_rep_std, log_S_test_rep)

        # Convert flat density output to raw-y scale.
        # log_pdf_raw = log_pdf_std - log(y_scale)
        # log_cdf_raw = log_cdf_std
        log_pdf_cdf_raw_flat = self._unstandardize_log_pdf_cdf_flat(
            log_pdf_cdf_std_flat
        )

        n_x = X_test_raw.shape[0]
        n_y = y_grid_raw.shape[0]

        out_std = self._reshape_log_pdf_cdf(
            log_pdf_cdf_std_flat,
            n_x=n_x,
            n_y=n_y,
        )
        out_raw = self._reshape_log_pdf_cdf(
            log_pdf_cdf_raw_flat,
            n_x=n_x,
            n_y=n_y,
        )

        predict_raw = {
        "pdf_grid": out_raw["pdf_grid"],
        "cdf_grid": out_raw["cdf_grid"],
        }

        if return_summaries:
            summaries = self.density_grid_summaries(
                out_raw["pdf_grid"],
                out_raw["cdf_grid"],
                y_grid_raw,
                qs=qs,
            )
            predict_raw.update(summaries)

        # Cache both raw and standardized versions.
        self.last_X_test_raw_ = X_test_raw
        self.last_X_test_ = X_test_std

        self.last_y_grid_raw_ = y_grid_raw
        self.last_y_grid_ = y_grid_std

        self.last_log_S_test_ = log_S_test
        self.last_y_rep_ = y_rep_std
        self.last_log_S_test_rep_ = log_S_test_rep

        self.last_log_pdf_cdf_std_ = out_std
        self.last_log_pdf_cdf_ = out_raw

        return predict_raw


    def test_log_pdf(
        self,
        X_test,
        y_test):
        """
        Compute log p(y_test[i] | X_test[i]) for paired test observations.

        Inputs X_test and y_test are on the raw/original scale.
        The returned log_pdf is also for the raw/original y-density.

        Parameters
        ----------
        X_test : array, shape (n_test, p) or (n_test,)
            Test covariates.

        y_test : array, shape (n_test,)
            Test responses paired with rows of X_test.

        Returns
        -------
        log_pdf : array, shape (n_test,)
            Raw-scale log density values.
        """
        self._check_is_fitted()

        # Raw inputs
        X_test_raw = self._as_2d(X_test)
        y_test_raw = jnp.asarray(y_test)

        if y_test_raw.ndim == 0:
            y_test_raw = y_test_raw[None]

        if y_test_raw.ndim != 1:
            raise ValueError("y_test must be one-dimensional.")

        if X_test_raw.shape[0] != y_test_raw.shape[0]:
            raise ValueError(
                "X_test and y_test must have the same number of observations. "
                f"Got X_test.shape[0]={X_test_raw.shape[0]} and "
                f"y_test.shape[0]={y_test_raw.shape[0]}."
            )

        # Standardize inputs
        X_test_std = self._standardize_X(X_test_raw)
        y_test_std = self._standardize_y(y_test_raw)

        # Similarity against training data
        log_S_test = self._logS_cross_std(X_test_std, self.X_)

        # Backend returns standardized-y log_pdf/log_cdf
        log_pdf_cdf_std = self._test_backend(y_test_std, log_S_test)

        # Convert to raw y-density scale:
        # log f_Y(y) = log f_Ystd(ystd) - log(y_scale)
        log_pdf_cdf_raw = self._unstandardize_log_pdf_cdf_flat(log_pdf_cdf_std)
        log_pdf = log_pdf_cdf_raw["log_pdf"]

    
        return log_pdf

    # ------------------------------------------------------------------
    # PR for density method
    # ------------------------------------------------------------------

    def pr(
        self,
        X_test=None,
        y_grid=None,
        X_pr=None,
        X_range = None,
        T=4000,
        B=1000,
        seed=126,
        kde_bw=None,
        X_pr_method="uniform",
        return_intervals=True,
        qs=(0.025, 0.975),
    ):
        """
        Predictive resampling for density method.

        Inputs X_test, y_grid, and X_pr are raw scale.
        """
        self._check_is_fitted()

        # --------------------------------------------------------------
        # Resolve X_test
        # --------------------------------------------------------------
        if X_test is None:
            if not hasattr(self, "last_X_test_raw_"):
                raise ValueError("X_test is None and no previous prediction exists.")
            X_test_raw = self.last_X_test_raw_
            X_test_std = self.last_X_test_
        else:
            X_test_raw = self._as_2d(X_test)
            X_test_std = self._standardize_X(X_test_raw)

        # --------------------------------------------------------------
        # Resolve y_grid
        # --------------------------------------------------------------
        if y_grid is None:
            if not hasattr(self, "last_y_grid_raw_"):
                raise ValueError("y_grid is None and no previous prediction exists.")
            y_grid_raw = self.last_y_grid_raw_
            y_grid_std = self.last_y_grid_
        else:
            y_grid_raw = jnp.asarray(y_grid)
            if y_grid_raw.ndim != 1:
                raise ValueError("y_grid must be one-dimensional.")
            y_grid_std = self._standardize_y(y_grid_raw)

        n_x = X_test_raw.shape[0]
        n_y = y_grid_raw.shape[0]

        # --------------------------------------------------------------
        # Resolve standardized log_pdf_cdf for backend PR
        # --------------------------------------------------------------
        if (
            hasattr(self, "last_log_pdf_cdf_std_")
            and np.array_equal(X_test_raw, self.last_X_test_raw_)
            and np.array_equal(y_grid_raw, self.last_y_grid_raw_)
        ):
            log_pdf_cdf_std = self.last_log_pdf_cdf_std_
        else:
            self.predict(
                X_test_raw,
                y_grid_raw,
                return_summaries=False,
            )
            log_pdf_cdf_std = self.last_log_pdf_cdf_std_
      
        # --------------------------------------------------------------
        # Generate/use X_pr
        # --------------------------------------------------------------
        if X_pr is None:
            X_pr_raw = self._default_X_pr(
                T=T,
                seed=seed,
                kde_bw=kde_bw,
                X_range = X_range,
                X_pr_method=X_pr_method,
            )
        else:
            X_pr_raw = self._as_2d(X_pr)

        X_pr_std = self._standardize_X(X_pr_raw)

        # --------------------------------------------------------------
        # Construct PR similarity grid on standardized X scale
        # --------------------------------------------------------------
        log_S_test_pr = self._logS_cross_std(X_test_std, X_pr_std)
        log_S_test_pr_rep = jnp.repeat(log_S_test_pr, n_y, axis=0)

        # Backend PR on standardized-y density object.
        log_pdf_cdf_pr_std_flat = self._pr_backend(
            log_pdf_cdf_std=log_pdf_cdf_std,
            log_S_test_pr_rep=log_S_test_pr_rep,
            B=B,
            seed=seed,
        )

        # Convert PR output to raw-y scale.
        log_pdf_cdf_pr_raw_flat = self._unstandardize_log_pdf_cdf_flat(
            log_pdf_cdf_pr_std_flat
        )

        out_pr_std = self._reshape_log_pdf_cdf_pr(
            log_pdf_cdf_pr_std_flat,
            n_x=n_x,
            n_y=n_y,
        )
        out_pr_raw = self._reshape_log_pdf_cdf_pr(
            log_pdf_cdf_pr_raw_flat,
            n_x=n_x,
            n_y=n_y,
        )

        self.last_X_pr_raw_ = X_pr_raw
        self.last_X_pr_ = X_pr_std
        self.last_log_S_test_pr_ = log_S_test_pr
        self.last_log_S_test_pr_rep_ = log_S_test_pr_rep

        self.last_log_pdf_cdf_pr_std_ = out_pr_std
        self.last_log_pdf_cdf_pr_ = out_pr_raw

        # Process outputs
        samples_pr_raw = {
            "pdf_grid": out_pr_raw["pdf_grid"],
            "cdf_grid": out_pr_raw["cdf_grid"],
        }

        if return_intervals:
            intervals = self.pr_intervals_moments(
                out_pr_raw["pdf_grid"],
                y_grid_raw,
                qs=qs,
            )
            return samples_pr_raw, intervals
        else: return samples_pr_raw


    @staticmethod
    def pr_intervals_moments(pdf_pr_grid, y_grid, qs=(0.025, 0.975)):
        pdf_np = np.asarray(pdf_pr_grid)
        y_np = np.asarray(y_grid)

        B, n_x, n_y = pdf_np.shape

        dy = float(np.mean(np.diff(y_np)))

        mu = np.sum(pdf_np * y_np[None, None, :] * dy, axis=2)

        centered = y_np[None, None, :] - mu[:, :, None]
        var = np.sum(pdf_np * centered**2 * dy, axis=2)

        q_low, q_high = qs

        return {
            "μ_l": np.quantile(mu, q_low, axis=0),
            "μ_h": np.quantile(mu, q_high, axis=0),
            "μ_mean": np.mean(mu, axis=0),

            "σ2_l": np.quantile(var, q_low, axis=0),
            "σ2_h": np.quantile(var, q_high, axis=0),
            "σ2_mean": np.mean(var, axis=0),

            "μ_pr": mu,
            "σ2_pr": var,
        }
