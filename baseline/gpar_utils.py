"""GPAR inducing grids and pointwise predictive scores."""

import numpy as np
import torch


def make_inducing_grid(X, spatial_points=10):
    """Setup regular inducing points on a grid according to main paper"""
    longitude = np.linspace(X[:, 0].min(), X[:, 0].max(), spatial_points)
    latitude = np.linspace(X[:, 1].min(), X[:, 1].max(), spatial_points)
    years = np.unique(X[:, 2])
    return np.column_stack([v.ravel() for v in np.meshgrid(
        longitude, latitude, years, indexing="ij")])


def conditional_log_pdf(model, Xs, y, y_train, last_dim_only=False):
    """Raw-y log p(y_j | x,y_<j,D), pointwise across rows; Xs is standardized.
    last_dim_only = True returns just the last conditional, with shape (N,1).
    y_train contains the raw training outputs, including missing values.
    """
    from gpar.regression import _construct_gpar

    terms = []
    with torch.no_grad():
        ys = model._normalise_y(torch.tensor(y, dtype=torch.float64)).numpy()
        log_scale = np.zeros(model.p)
        if model.normalise_y:
            train_y = model._transform_y(torch.tensor(y_train, dtype=torch.float64))
            # Match GPAR's population standard deviations and zero-variance fallback.
            std = torch.stack([
                torch.std(train_y[~torch.isnan(train_y[:, j]), j], unbiased=False)
                for j in range(model.p)])
            log_scale = np.log(torch.where(std > 0, std, torch.ones_like(std)).numpy())
        posterior = _construct_gpar(model, model.vs, model.m, model.p) | (model.x, model.y, model.w)
        for j, layer in enumerate(posterior.layers):
            if last_dim_only and j != model.p - 1:
                continue
            process, noise = layer()
            inputs = torch.tensor(np.column_stack([Xs, ys[:, :j]]), dtype=torch.float64)
            mean, variance = process(inputs, noise).marginals()
            mean, variance = mean.numpy(), variance.numpy()
            term = -.5 * (np.log(2 * np.pi * variance) + (ys[:, j] - mean)**2 / variance)
            terms.append(term - log_scale[j])  # Normalization Jacobian, once per output.
    return np.column_stack(terms)
