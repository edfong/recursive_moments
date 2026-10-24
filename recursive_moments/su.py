import jax.numpy as jnp
import jax
import numpy as np
from scipy.optimize import minimize

from .moment import RMMomentBase
from .su_fun import (
    fit_moments_perm,
    test_moments_perm,
    pr_moments,
    compute_quantile,
    log_pdf,
)
from .Sxx_kern import (
    logSxx_gauss_self,
    logSxx_gauss,
)

def _inv_softplus_np(y):
    """
    Stable inverse of softplus for y > 0.
    Solves y = log(1 + exp(x)).
    """
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))

class RMSU(RMMomentBase):
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
        lambda_=50.0,
        optimize_lambda=True,
    ):
        if lambda_ <= 3.0:
            raise ValueError(
                f"RMSU requires lambda_ > 3. Got lambda_={lambda_}."
            )

        super().__init__(
            optimize=optimize,
            rho_x=rho_x,
            rho_y=rho_y,
            n_perm_opt=n_perm_opt,
            n_perm_fit=n_perm_fit,
        )

        self.lambda_ = float(lambda_)
        self.optimize_lambda = optimize_lambda

    def _backend_kwargs(self):
        """
        Pass fixed/current lambda into the JAX backend.
        """
        if self.lambda_ <= 3.0:
            raise ValueError(
                f"RMSU requires lambda_ > 3. Got lambda_={self.lambda_}."
            )

        return {"λ": jnp.asarray(self.lambda_)}

    def _optimize_hyperparams(self, X_std, y_std):
        """
        Jointly optimize rho_x, rho_y, and lambda using prequential log likelihood.

        Optimization vector:
            x = [x_rho_x_1, ..., x_rho_x_p, x_rho_y, eta_lambda]

        Transformations:
            rho = 1 / (1 + exp(x_rho))
            lambda = 3 + softplus(eta_lambda)

        Therefore lambda > 3 throughout optimization.
        """
        if not hasattr(self, "fit_fn") or self.fit_fn is None:
            raise RuntimeError(
                "Subclass must define fit_fn in order to optimize hyperparameters."
            )

        p = X_std.shape[1]

        # Initial unconstrained rho parameters.
        x0_rho = -2.0 * np.ones(p + 1, dtype=np.float64)

        if self.optimize_lambda:
            # lambda = 3 + softplus(eta)
            eta0 = _inv_softplus_np(self.lambda_ - 3.0)
            x0 = np.concatenate([x0_rho, np.array([eta0], dtype=np.float64)])
        else:
            x0 = x0_rho

        def loss(x):
            # ----------------------------------------------------------
            # rho part
            # ----------------------------------------------------------
            x_rho = x[: p + 1]

            rho = 1.0 / (1.0 + jnp.exp(x_rho))

            rho_x = rho[:-1]
            rho_y = rho[-1]

            # ----------------------------------------------------------
            # lambda part
            # ----------------------------------------------------------
            if self.optimize_lambda:
                eta_lambda = x[p + 1]
                λ_current = 3.0 + jax.nn.softplus(eta_lambda)
            else:
                λ_current = jnp.asarray(self.lambda_)

            # ----------------------------------------------------------
            # Prequential objective
            # ----------------------------------------------------------
            log_Sn = self.logS_self_fn(X_std, ρx=rho_x)

            _, log_preq = self.fit_fn(
                y_std,
                log_Sn,
                ρ=rho_y,
                λ=λ_current,
                n_perm=self.n_perm_opt,
            )

            return -log_preq

        loss_and_grad = jax.jit(jax.value_and_grad(loss))

        def scipy_objective(x_np):
            loss_val, grad = loss_and_grad(jnp.asarray(x_np))
            return float(loss_val), np.asarray(grad, dtype=np.float64)

        res = minimize(
            scipy_objective,
            x0,
            method="L-BFGS-B",
            jac=True,
        )

        print(f"Joint rho/lambda optimization success: {res.success}")

        x_opt = res.x

        # --------------------------------------------------------------
        # Extract optimized rho
        # --------------------------------------------------------------
        x_rho_opt = x_opt[: p + 1]

        rho_vec = 1.0 / (1.0 + np.exp(x_rho_opt))

        rho_x = rho_vec[:-1]
        rho_y = rho_vec[-1]

        rho_x = np.clip(rho_x, 0.0, 0.99)
        rho_y = np.clip(rho_y, 0.1, 0.98)

        # --------------------------------------------------------------
        # Extract optimized lambda
        # --------------------------------------------------------------
        if self.optimize_lambda:
            eta_lambda_opt = x_opt[p + 1]
            lambda_opt = 3.0 + float(jax.nn.softplus(jnp.asarray(eta_lambda_opt)))
            self.lambda_ = lambda_opt

            print(f"Optimized lambda: {self.lambda_:.4f}")

        return rho_x, rho_y
