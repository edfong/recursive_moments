"""Fit the ICM and GPAR baselines on the shared US weather split."""

from pathlib import Path
from time import perf_counter
import sys

import numpy as np

from weather import load_weather as weather_setup

ICM_NUM_INDUCING = (64, 128, 256)
ICM_MAX_ITER = 5000
GPAR_SPATIAL_POINTS = (10, 20, None)  # None uses exact GPs.
GPAR_MAX_ITER = 500
GPAR_NUM_SAMPLES = 200


def _load_weather_split():
    weather, X, y = weather_setup.load_weather_data()
    return weather_setup.load_weather_split(weather, X, y)


def _save_results(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)


def _print_scores(result):
    print(f"Mean test log PDF, held-out rows (TMP/SLP/DEW): {result['mean_test_log_pdf']:.6f}", flush=True)
    print(f"Mean test log PDF, hidden DEW given TMP/SLP: {result['mean_dew_test_log_pdf']:.6f}", flush=True)


def run_weather_icm(max_iter=ICM_MAX_ITER, num_inducing=256):
    from baseline import ICM

    total_start = perf_counter()
    split = _load_weather_split()
    Xt, yt = split["X_train"], split["y_train"]
    scaling = dict(x_mean=Xt.mean(0), x_scale=Xt.std(0),
                   y_mean=np.nanmean(yt, axis=0), y_scale=np.nanstd(yt, axis=0))
    Xs = (Xt - scaling["x_mean"]) / scaling["x_scale"]
    ys = (yt - scaling["y_mean"]) / scaling["y_scale"]
    label = "Sparse GPflow ICM (SVGP)"
    print(f"{label}: {len(Xt)} training rows; {len(split['X_test'])} held-out rows; "
          f"{len(split['X_dew_test'])} hidden DEW values", flush=True)
    start = perf_counter()
    model = ICM.make_icm(Xs, ys, num_inducing, seed=weather_setup.SEED)
    num_inducing = len(model.inducing_variable.Z.numpy())
    opt = ICM.optimize_icm(model, (Xs, ys), max_iter)
    fit_seconds = perf_counter() - start
    path = weather_setup.icm_results_path(num_inducing)
    print(f"Fit: {fit_seconds:.1f}s; {opt.message}", flush=True)
    scores = ICM.test_log_pdf(model, split["X_test"], split["y_test"], scaling)
    dew_scores = ICM.test_log_pdf(model, split["X_dew_test"], split["y_dew_test"], scaling,
                                 last_dim_only=True)
    result = dict(mean_test_log_pdf=float(scores.mean()),
                  mean_dew_test_log_pdf=float(dew_scores.mean()), fit_seconds=fit_seconds)
    _save_results(path, result)
    _print_scores(result)
    longitude, latitude, grid = weather_setup.make_prediction_grid()
    grid_scaled = (grid - scaling["x_mean"]) / scaling["x_scale"]
    mean, _ = model.predict_f(ICM.augment_inputs(grid_scaled, yt.shape[1]))
    mean = scaling["y_mean"] + mean.numpy().reshape(-1, yt.shape[1]) * scaling["y_scale"]
    shape, p = (len(latitude), len(longitude)), yt.shape[1]
    result.update(
        longitude=longitude, latitude=latitude, output_names=weather_setup.OUTPUT_NAMES,
        mu_mean=mean.reshape(shape + (p,)),
        total_seconds=perf_counter() - total_start,
    )
    _save_results(path, result)
    print(f"Total including prediction: {result['total_seconds']:.1f}s; saved {path}", flush=True)
    return model, result


def run_weather_gpar(spatial_points=10, max_iter=GPAR_MAX_ITER,
                     num_samples=GPAR_NUM_SAMPLES, output_path=None):
    total_start = perf_counter()
    import torch
    from lab.torch import B
    from baseline import gpar_utils

    sys.path.insert(0, str(weather_setup.ROOT / "baseline" / "gpar"))
    from gpar import GPARRegressor

    B.epsilon = 1e-6  # Numerical jitter for the large inducing grids; smaller values cause numerical instability
    split = _load_weather_split()
    Xt, yt = split["X_train"], split["y_train"]
    # GPAR normalizes outputs, but has no built-in input standardizer.
    scaling = dict(x_mean=Xt.mean(0), x_scale=np.where(Xt.std(0) > 0, Xt.std(0), 1.))
    Xs = (Xt - scaling["x_mean"]) / scaling["x_scale"]
    Z = None if spatial_points is None else gpar_utils.make_inducing_grid(Xt, spatial_points)
    Zs = None if Z is None else (Z - scaling["x_mean"]) / scaling["x_scale"]
    name = "exact" if Z is None else f"m{len(Z)}"
    print(f"GPAR {name}: {len(Xt)} training / {len(split['X_test'])} test rows", flush=True)
    start = perf_counter()
    model = GPARRegressor(nonlinear=True, x_ind=Zs)
    model.fit(Xs, yt, fix=True, iters=max_iter, trace=True)
    fit_seconds = perf_counter() - start
    factors = gpar_utils.conditional_log_pdf(
        model, (split["X_test"] - scaling["x_mean"]) / scaling["x_scale"],
        split["y_test"], y_train=yt)
    scores = factors.sum(axis=1)
    dew_scores = gpar_utils.conditional_log_pdf(
        model, (split["X_dew_test"] - scaling["x_mean"]) / scaling["x_scale"],
        split["y_dew_test"], y_train=yt, last_dim_only=True)[:, 0]
    result = dict(
        fit_seconds=fit_seconds, mean_test_log_pdf=float(scores.mean()),
        mean_dew_test_log_pdf=float(dew_scores.mean()),
    )
    path = (weather_setup.gpar_results_path(None if Z is None else len(Z))
            if output_path is None else Path(output_path))
    # Preserve scores before the slower Monte Carlo map calculation.
    _save_results(path, result)
    print(f"Fit: {fit_seconds:.1f}s", flush=True)
    _print_scores(result)
    longitude, latitude, grid = weather_setup.make_prediction_grid()
    if num_samples < 2:
        raise ValueError("Use at least two posterior samples.")
    B.set_random_seed(weather_setup.SEED)
    with torch.no_grad():
        mean = model.predict(
            (grid - scaling["x_mean"]) / scaling["x_scale"],
            num_samples=num_samples, latent=False)
    shape = (len(latitude), len(longitude), yt.shape[1])
    result.update(mu_mean=mean.reshape(shape),
                  longitude=longitude, latitude=latitude, output_names=weather_setup.OUTPUT_NAMES)
    # Includes loading, fitting, scoring, checkpoints and all grid predictions.
    # Stops before this final results write; plotting runs separately.
    result["total_seconds"] = perf_counter() - total_start
    _save_results(path, result)
    print(f"Total GPAR time (including prediction): {result['total_seconds']:.2f}s", flush=True)
    print(f"Saved {path}", flush=True)
    return model, result


def run_weather_GPs(methods=("ICM", "GPAR"), icm_num_inducing=ICM_NUM_INDUCING,
                    gpar_spatial_points=GPAR_SPATIAL_POINTS):
    """Run the paper settings for the selected GP methods."""
    if not set(methods).issubset({"ICM", "GPAR"}):
        raise ValueError("Choose ICM and/or GPAR.")
    results = {}
    if "ICM" in methods:
        for m in icm_num_inducing:
            results[f"ICM_m{m}"] = run_weather_icm(num_inducing=m)[1]
    if "GPAR" in methods:
        for l in gpar_spatial_points:
            name = "GPAR_exact" if l is None else f"GPAR_l{l}"
            results[name] = run_weather_gpar(spatial_points=l)[1]
    return results


if __name__ == "__main__":
    results = run_weather_GPs()