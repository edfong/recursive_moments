import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import minimize

try:
    from sklearn.base import BaseEstimator
except Exception:
    BaseEstimator = object


class RMCommonBase(BaseEstimator):
    """
    Common class shared by moment-based and density-based RM models.
    """

    logS_self_fn = None
    logS_cross_fn = None

    def __init__(
        self,
        optimize=True,
        rho_x=None,
        rho_y=0.9,
        n_perm_opt=10,
    ):
        # Constructor parameters omit the trailing "_" for sklearn compatibility.
        # Fitted attributes use the trailing "_".
        self.optimize = optimize
        self.rho_x = rho_x
        self.rho_y = rho_y
        self.n_perm_opt = n_perm_opt

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _as_2d(self, X):
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        return X

    def _make_rho_x(self, rho_x, p):
        """
        Ensure rho_x has shape (p,).
        """
        if rho_x is None:
            rho_x = 0.9

        rho_x = jnp.asarray(rho_x)

        if rho_x.ndim == 0:
            rho_x = jnp.full((p,), rho_x)

        return rho_x

    def _check_is_fitted(self):
        if not hasattr(self, "model_perm_"):
            raise RuntimeError("Model is not fitted yet. Call `.fit(X, y)` first.")

    # ------------------------------------------------------------------
    # Standardization helpers
    # ------------------------------------------------------------------

    def _fit_standardizers(self, X, y):
        """
        Fit standardization constants using training data.
        """
        X_loc = jnp.mean(X, axis=0)
        X_scale = jnp.std(X, axis=0)

        # Avoid division by zero for constant columns.
        X_scale = jnp.where(X_scale > 1e-12, X_scale, 1.0)

        y_loc = jnp.mean(y)
        y_scale = jnp.std(y)

        # Avoid division by zero for constant y.
        y_scale = jnp.where(y_scale > 1e-12, y_scale, 1.0)

        self.X_loc_ = X_loc
        self.X_scale_ = X_scale
        self.y_loc_ = y_loc
        self.y_scale_ = y_scale

    def _standardize_X(self, X):
        X = self._as_2d(X)
        return (X - self.X_loc_) / self.X_scale_

    def _unstandardize_X(self, X_std):
        return self.X_loc_ + self.X_scale_ * X_std

    def _standardize_y(self, y):
        return (jnp.asarray(y) - self.y_loc_) / self.y_scale_

    def _unstandardize_y(self, y_std):
        return self.y_loc_ + self.y_scale_ * y_std

    # ------------------------------------------------------------------
    # Moment transformation helpers
    # ------------------------------------------------------------------

    def _unstandardize_moments(self, moments_std):
        """
        Convert standardized-y moment dictionary to original-y scale.
        """
        out = dict(moments_std)

        sy = self.y_scale_
        my = self.y_loc_

        if "μ" in out:
            out["μ"] = my + sy * out["μ"]

        if "σ2" in out:
            out["σ2"] = sy**2 * out["σ2"]

        return out

    # ------------------------------------------------------------------
    # Density transformation helpers
    # ------------------------------------------------------------------

    def _unstandardize_log_pdf_cdf_flat(self, log_pdf_cdf_std):
        """
        Convert standardized-y log density/CDF output to original-y scale.
        """
        out = dict(log_pdf_cdf_std)

        if "log_pdf" in out:
            out["log_pdf"] = out["log_pdf"] - jnp.log(self.y_scale_)

        if "log_cdf" in out:
            out["log_cdf"] = out["log_cdf"]

        return out

    # ------------------------------------------------------------------
    # Similarity wrappers
    # ------------------------------------------------------------------

    def _logS_self_std(self, X_std):
        return self.logS_self_fn(X_std, ρx=self.rho_x_)

    def _logS_cross_std(self, X1_std, X2_std):
        return self.logS_cross_fn(X1_std, X2_std, ρx=self.rho_x_)

    # ------------------------------------------------------------------
    # Hyperparameter optimization
    # ------------------------------------------------------------------
    def _backend_kwargs(self):
        """
        Extra keyword arguments passed to backend JAX functions.

        Subclasses with additional hyperparameters can override this.
        Returns an empty dictionary by default.
        """
        return {}

    def _make_optimization_loss(self, X_std, y_std):
        """Build the prequential loss, retaining permutations even for one fit."""
        if not hasattr(self, "fit_fn") or self.fit_fn is None:
            raise RuntimeError(
                "Subclass must define fit_fn in order to optimize hyperparameters."
            )

        def loss(x):
            rho = 1.0 / (1.0 + jnp.exp(x))

            rho_x = rho[:-1]
            rho_y = rho[-1]

            log_Sn = self.logS_self_fn(X_std, ρx=rho_x)

            _, log_preq = self.fit_fn(
                y_std,
                log_Sn,
                ρ=rho_y,
                n_perm=self.n_perm_opt,
                **self._backend_kwargs(),
            )

            return -log_preq

        return loss

    def _optimize_hyperparams(self, X_std, y_std):
        """Optimize rho_x and rho_y using the model's prequential loss."""
        p = X_std.shape[1]
        loss = self._make_optimization_loss(X_std, y_std)
        loss_and_grad = jax.jit(jax.value_and_grad(loss))

        def scipy_objective(x_np):
            loss_val, grad = loss_and_grad(jnp.asarray(x_np))
            return float(loss_val), np.asarray(grad, dtype=np.float64)

        x0 = -2 * np.ones(p + 1)

        res = minimize(
            scipy_objective,
            x0,
            method="L-BFGS-B",
            jac=True,
        )

        print(f"Bandwidth optimization success: {res.success}")

        rho_vec = 1.0 / (1.0 + np.exp(res.x))

        rho_x = rho_vec[:-1]
        rho_y = rho_vec[-1]

        rho_x = np.clip(rho_x, 0.0, 0.99)
        rho_y = np.clip(rho_y, 0.1, 0.99)

        return rho_x, rho_y
    
    # ------------------------------------------------------------------
    # Common fit setup
    # ------------------------------------------------------------------

    def _fit_common_setup(self, X, y):
        """
        Stores raw data, fits standardizers, standardizes X and y,
        optionally optimizes rho, and computes log_Sn on standardized X.
        """
        X_raw = self._as_2d(X)
        y_raw = jnp.asarray(y)

        self.X_raw_ = X_raw
        self.y_raw_ = y_raw
        self.n_, self.p_ = X_raw.shape

        # Fit and apply standardization.
        self._fit_standardizers(X_raw, y_raw)

        X_std = self._standardize_X(X_raw)
        y_std = self._standardize_y(y_raw)

        self.X_ = X_std
        self.y_ = y_std

        # Hyperparameters.
        if self.optimize:
            rho_x, rho_y = self._optimize_hyperparams(X_std, y_std)
            self.rho_x_ = jnp.asarray(rho_x)
            self.rho_y_ = jnp.asarray(rho_y)
        else:
            self.rho_x_ = self._make_rho_x(self.rho_x, self.p_)
            self.rho_y_ = jnp.asarray(self.rho_y)

        # Training similarity matrix.
        log_Sn = self._logS_self_std(X_std)

        return X_std, y_std, log_Sn

    # ------------------------------------------------------------------
    # Default predictive-resampling design
    # ------------------------------------------------------------------

    def _default_X_pr(
        self,
        T,
        seed=None,
        kde_bw=None,
        X_range=None,
        X_pr_method="uniform",
    ):
        """Draw a predictive-resampling covariates.

        Parameters
        ----------
        X_pr_method : {"uniform", "gaussian", "kde"}
            ``"uniform"`` draws independently within the component-wise
            ``X_range`` and is the default. If no range is supplied, the
            observed training bounds are used. ``"gaussian"`` draws from a
            standard isotropic Gaussian on standardized X-space. ``"kde"``
            uses bootstrap sampling with jitter.

        Returns
        -------
        jax.Array
            Draws on the raw/original X scale
        """
        X_np = np.asarray(self.X_raw_)
        n, p = X_np.shape
        rng = np.random.default_rng(seed)

        if X_pr_method == "gaussian":
            X_pr_std = rng.normal(size=(T, p))
            X_pr = (
                np.asarray(self.X_loc_)
                + np.asarray(self.X_scale_) * X_pr_std
            )
            return jnp.asarray(X_pr)

        if X_range is None:
            X_min = np.min(X_np, axis=0)
            X_max = np.max(X_np, axis=0)
        elif isinstance(X_range, tuple) and len(X_range) == 2:
            X_min, X_max = X_range
            X_min = np.broadcast_to(np.asarray(X_min), (p,))
            X_max = np.broadcast_to(np.asarray(X_max), (p,))
        else:
            raise ValueError(
                "X_range must be None or a tuple (X_min, X_max)."
            )

        if np.any(X_max < X_min):
            raise ValueError("Each X_range upper bound must be >= its lower bound.")

        if X_pr_method == "uniform":
            X_pr = rng.uniform(X_min, X_max, size=(T, p))
        elif X_pr_method == "kde":
            idx = rng.choice(n, size=T, replace=True)
            X_pr = X_np[idx].copy()
            if kde_bw is None:
                kde_bw = n ** (-1.0 / 5.0)
            X_pr = X_pr + rng.normal(size=X_pr.shape) * kde_bw
            X_pr = np.clip(X_pr, X_min, X_max)
        else:
            raise ValueError(
                "X_pr_method must be one of 'uniform', 'gaussian', or 'kde'."
            )

        return jnp.asarray(X_pr)
