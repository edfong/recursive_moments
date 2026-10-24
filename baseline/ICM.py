"""Native GPflow ICM with task-indexed observations, following coregionalisation.
"""
import gpflow
import numpy as np


def augment_inputs(X, p):
    """One query per output, in row-major order: (x0,0), (x0,1), ... ."""
    return np.column_stack([np.repeat(X, p, axis=0), np.tile(np.arange(p), len(X))])


def augment_data(X, y):
    """Append output indices to X and y, retaining observed values only."""
    rows, tasks = np.nonzero(np.isfinite(y))
    return np.column_stack([X[rows], tasks]), np.column_stack([y[rows, tasks], tasks])


def make_icm(Xs, ys, num_inducing=256, seed=125):
    num_inducing = int(num_inducing)
    gpflow.config.set_default_float(np.float64)
    p, input_dims = ys.shape[1], list(range(Xs.shape[1]))
    # Setup kernels
    rbf = gpflow.kernels.SquaredExponential(lengthscales=np.ones(len(input_dims)), active_dims=input_dims)
    gpflow.set_trainable(rbf.variance, False)  # Coregion supplies the amplitude.
    coreg = gpflow.kernels.Coregion(output_dim=p, rank=p, active_dims=[len(input_dims)])
    coreg.W.assign(.1 * np.random.default_rng(seed).standard_normal((p, p))) # Random initialization to prevent saddle
    kernel = rbf * coreg
    likelihood = gpflow.likelihoods.SwitchedLikelihood(
        [gpflow.likelihoods.Gaussian() for _ in range(p)])
    # Setup inducing points
    candidates = np.unique(Xs, axis=0)
    indices = np.random.default_rng(seed).permutation(len(candidates))
    Z = augment_inputs(candidates[indices], p)[:num_inducing]
    # Input coordinates are learned; the categorical index has zero gradient.
    return gpflow.models.SVGP(kernel, likelihood, inducing_variable=Z)


def optimize_icm(model, data, max_iter=5000):
    loss = model.training_loss_closure(augment_data(*data))
    iteration = 0

    def record(intermediate_result):
        nonlocal iteration
        iteration += 1
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: objective = {-float(intermediate_result.fun):.6f}", flush=True)

    result = gpflow.optimizers.Scipy().minimize(
        loss, model.trainable_variables, method="L-BFGS-B", callback=record,
        options={"maxiter": max_iter})
    return result


def noise_covariance(model, scaling):
    variance = np.array([lik.variance.numpy() for lik in model.likelihood.likelihoods])
    return np.diag(variance * scaling["y_scale"]**2)


def predict_icm(model, X, scaling):
    """Raw-unit mean and within-row covariance, including cross-output covariance."""
    Xs = (X - scaling["x_mean"]) / scaling["x_scale"]
    p = len(scaling["y_scale"])
    mean, covariance = model.predict_f(augment_inputs(Xs, p), full_cov=True)
    # Retain each row's P x P block; extract from full covariance over 3N data points
    blocks = covariance.numpy()[0].reshape(len(Xs), p, len(Xs), p)
    rows = np.arange(len(Xs))
    covariance = blocks[rows, :, rows, :]
    scale = scaling["y_scale"]
    mean = scaling["y_mean"] + mean.numpy().reshape(-1, p) * scale
    covariance = covariance * np.outer(scale, scale)
    return mean, .5 * (covariance + covariance.swapaxes(-1, -2))


def test_log_pdf(model, X, y, scaling, last_dim_only=False):
    """Pointwise raw-unit joint log PDF, or DEW with its TMP/SLP already in training."""
    mean, covariance = predict_icm(model, X, scaling)
    covariance += noise_covariance(model, scaling)[None]
    if last_dim_only:
        # In our setting, the TMP/SLP measurements are already conditioned on during fitting.
        # No need for additional conditioning.
        mean, y, covariance = mean[:, -1:], y[:, -1:], covariance[:, -1:, -1:]
    chol = np.linalg.cholesky(covariance)
    residual = np.linalg.solve(chol, (y - mean)[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
    # Scaling the covariance above already includes the normalization Jacobian.
    return -.5 * (y.shape[1] * np.log(2*np.pi) + logdet + (residual**2).sum(-1))
