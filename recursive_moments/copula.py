from .density import RMDensityBase
from .copula_fun import (
    fit_copula_perm,
    test_copula_perm,
    pr_np,
)
from .Sxx_kern import (
    logSxx_gauss_self,
    logSxx_gauss,
)


class RMCopula(RMDensityBase):
    fit_fn = staticmethod(fit_copula_perm)
    test_fn = staticmethod(test_copula_perm)
    pr_fn = staticmethod(pr_np)

    logS_self_fn = staticmethod(logSxx_gauss_self)
    logS_cross_fn = staticmethod(logSxx_gauss)
