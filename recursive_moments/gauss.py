from .moment import RMMomentBase
from .gauss_fun import (
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


class RMGauss(RMMomentBase):
    """Univariate Gaussian recursive moments."""

    fit_fn = staticmethod(fit_moments_perm)
    test_fn = staticmethod(test_moments_perm)
    quantile_fn = staticmethod(compute_quantile)
    logpdf_fn = staticmethod(log_pdf)
    pr_fn = staticmethod(pr_moments)

    logS_self_fn = staticmethod(logSxx_gauss_self)
    logS_cross_fn = staticmethod(logSxx_gauss)
