import jax.numpy as jnp
from .base import RMCommonBase


class RMMomentBase(RMCommonBase):
    """
    Base class for recursive-moment methods.
    """

    fit_fn = None
    test_fn = None
    logpdf_fn = None
    quantile_fn = None
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
            n_perm=self.n_perm_fit,
            **self._backend_kwargs(),
        )

    def _test_backend(self, log_S_test):
        return self.test_fn(
            self.model_perm_,
            log_S_test,
            ρ=self.rho_y_,
            **self._backend_kwargs(),
        )

    def _pr_backend(self, moments_test_std, log_S_test_pr, B, seed):
        return self.pr_fn(
            moments_test_std,
            log_S_test_pr,
            self.n_,
            ρ=self.rho_y_,
            B=B,
            seed=seed,
            **self._backend_kwargs(),
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_std, y_std, log_Sn = self._fit_common_setup(X, y)

        model_perm, preq_loglik = self._fit_backend(y_std, log_Sn)

        self.model_perm_ = model_perm
        self.preq_loglik_ = preq_loglik - jnp.log(self.y_scale_)  # Return prequential log likelihood on unstandardized scale

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X_test,qs=(0.025, 0.975)):
        self._check_is_fitted()

        X_test_raw = self._as_2d(X_test)
        X_test_std = self._standardize_X(X_test_raw)

        log_S_test = self._logS_cross_std(X_test_std, self.X_)

        moments_std = self._test_backend(log_S_test)
        moments_raw = self._unstandardize_moments(moments_std)

        self.last_X_test_raw_ = X_test_raw
        self.last_X_test_ = X_test_std
        self.last_log_S_test_ = log_S_test

        self.last_moments_std_ = moments_std
        self.last_moments_ = moments_raw

        # Compute prediction interval
        y_l = self.quantile_fn(
            moments_raw,
            α=qs[0],
            **self._backend_kwargs(),
        )

        y_h = self.quantile_fn(
            moments_raw,
            α=qs[1],
            **self._backend_kwargs(),
        )

        predict_raw = {**moments_raw, 'y_l': y_l, 'y_h': y_h}

        return predict_raw
        
    # ------------------------------------------------------------------
    # Test log pdf
    # ------------------------------------------------------------------

    def test_log_pdf(self, X_test, y_test):
        predict_raw = self.predict(X_test)
        return self.logpdf_fn(
            y_test,
            predict_raw,
            **self._backend_kwargs(),
        )
    

    # ------------------------------------------------------------------
    # Predictive resampling
    # ------------------------------------------------------------------

    def pr(
        self,
        X_test=None,
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
        self._check_is_fitted()

        # --------------------------------------------------------------
        # Resolve X_test
        # --------------------------------------------------------------
        if X_test is None:
            if not hasattr(self, "last_X_test_raw_"):
                raise ValueError(
                    "X_test is None and no previous prediction exists. "
                    "Call predict(X_test) first or pass X_test."
                )
            X_test_raw = self.last_X_test_raw_
            X_test_std = self.last_X_test_
            use_cached_moments = hasattr(self, "last_moments_std_")
        else:
            X_test_raw = self._as_2d(X_test)
            X_test_std = self._standardize_X(X_test_raw)
            use_cached_moments = False

        # --------------------------------------------------------------
        # Resolve moments_test on standardized scale
        # --------------------------------------------------------------
    
        if use_cached_moments:
            moments_test_std = self.last_moments_std_
        else:
            # This call caches both raw and standardized moments.
            self.predict(X_test_raw)
            moments_test_std = self.last_moments_std_
        
        # --------------------------------------------------------------
        # Resolve X_pr
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
        # PR backend on standardized scale
        # --------------------------------------------------------------
        log_S_test_pr = self._logS_cross_std(X_test_std, X_pr_std)

        moments_pr_std = self._pr_backend(
            moments_test_std=moments_test_std,
            log_S_test_pr=log_S_test_pr,
            B=B,
            seed=seed,
        )

        moments_pr_raw = self._unstandardize_moments(moments_pr_std)

        self.last_X_pr_raw_ = X_pr_raw
        self.last_X_pr_ = X_pr_std
        self.last_log_S_test_pr_ = log_S_test_pr

        self.last_moments_pr_std_ = moments_pr_std
        self.last_moments_pr_ = moments_pr_raw

        if return_intervals:
            intervals = self.pr_intervals(moments_pr_raw, qs=qs)
            return moments_pr_raw, intervals

        return moments_pr_raw

    # ------------------------------------------------------------------
    # Interval helper
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_summary(arr, name, qs):
        q_low, q_high = qs
        arr = jnp.asarray(arr)

        return {
            f"{name}_l": jnp.quantile(arr, q_low, axis=0),
            f"{name}_h": jnp.quantile(arr, q_high, axis=0),
            f"{name}_mean": jnp.mean(arr, axis=0),
        }

    def pr_intervals(self, moments_pr, qs=(0.025, 0.975)):
        """
        Mean and variance PR intervals on raw/original scale.
        """
        intervals = {}

        for key in ("μ", "σ2"):
            if key in moments_pr:
                intervals.update(self._quantile_summary(moments_pr[key], key, qs))

        return intervals
