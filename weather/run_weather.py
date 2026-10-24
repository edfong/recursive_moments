"""Fit multi-output and independent RM models on the shared US weather split.
"""

from time import perf_counter

import numpy as np
from weather import load_weather as weather_setup

# Settings
T = 4000
B = 1000


def run_weather():
    """Fit multi-output RM, then independent RM using its bandwidths."""
    import jax
    from recursive_moments import RMGauss, RMMOGauss

    total_start = perf_counter()
    weather, X, y = weather_setup.load_weather_data()
    split = weather_setup.load_weather_split(weather, X, y)
    X_train, y_train = split["X_train"], split["y_train"]
    longitude, latitude, x_plot = weather_setup.make_prediction_grid()

    model_mohet = RMMOGauss(optimize=True, n_perm_opt=1, n_perm_fit=10, missing=True)
    print(f"Fitting {len(X_train)} station-years "
          f"({len(np.unique(split['station_ids_train']))} stations, "
          f"{weather_setup.START_YEAR}-{weather_setup.END_YEAR}); "
          f"outputs: {', '.join(weather_setup.OUTPUT_NAMES)}", flush=True)
    print(f"Held-out rows: {len(split['X_test'])}; hidden training DEW: {len(split['X_dew_test'])}; "
          f"split seed: {split['split_seed']}", flush=True)
    start = perf_counter()
    model_mohet.fit(X_train, y_train)
    model_mohet.preq_loglik_.block_until_ready()
    fit_seconds = perf_counter() - start
    print(f"Fit: {fit_seconds:.1f}s", flush=True)
    print("rho_x:", model_mohet.rho_x_)
    print("rho_y:", model_mohet.rho_y_)
    print("preq_loglik:", model_mohet.preq_loglik_)

    # Scoring updates the prediction cache, so it should precede prediction.
    scores = np.asarray(model_mohet.test_log_pdf(
        split["X_test"], split["y_test"]), dtype=float)
    dew_scores = np.asarray(model_mohet.test_log_pdf(
        split["X_dew_test"], split["y_dew_test"], conditional=True), dtype=float)

    print(f"Mean test log PDF, held-out rows (TMP/SLP/DEW): {scores.mean():.6f}", flush=True)
    print(f"Mean test log PDF, hidden DEW given TMP/SLP: {dew_scores.mean():.6f}", flush=True)
    print(f"Predicting on a {weather_setup.GRID_SIZE} x {weather_setup.GRID_SIZE} grid "
          f"for {weather_setup.PREDICT_YEAR}", flush=True)
    jax.block_until_ready(model_mohet.predict(x_plot))

    print(f"Predictive resampling: T={T}, B={B}", flush=True)
    start = perf_counter()
    # Reuse the predictions cached above; X_test=None refers to x_plot.
    moments_pr = model_mohet.pr(
        X_test=None,
        X_range=(X_train.min(axis=0), X_train.max(axis=0)),
        T=T, B=B, seed=weather_setup.SEED, return_intervals=False,
    )
    jax.block_until_ready(moments_pr)
    total_seconds = perf_counter() - total_start
    print(f"Resampling: {perf_counter() - start:.1f}s", flush=True)

    shape = (len(latitude), len(longitude))
    d = len(weather_setup.OUTPUT_NAMES)
    mu = np.asarray(moments_pr["μ"])
    correlation = np.asarray(moments_pr["R"])
    # Summarize each function over PR draws
    multi_result = dict(
        longitude=longitude, latitude=latitude, output_names=weather_setup.OUTPUT_NAMES,
        mu_mean=mu.mean(axis=0, dtype=np.float64).reshape(shape + (d,)),
        mu_var=mu.var(axis=0, ddof=1, dtype=np.float64).reshape(shape + (d,)),
        r_mean=correlation.mean(axis=0, dtype=np.float64).reshape(shape + (d, d)),
        mean_test_log_pdf=float(scores.mean()), mean_dew_test_log_pdf=float(dew_scores.mean()),
        fit_seconds=fit_seconds, total_seconds=total_seconds,
    )
    path = weather_setup.rm_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **multi_result)
    print(f"Total including prediction/resampling: {total_seconds:.1f}s; saved {path}", flush=True)

    # Reuse the split and fitted bandwidths directly for the independent fits.
    total_start = perf_counter()
    shared_rho_x = np.asarray(model_mohet.rho_x_)
    shared_rho_y = float(model_mohet.rho_y_)
    jax.config.update("jax_enable_x64", shared_rho_x.dtype.name == "float64")
    dew = weather_setup.OUTPUT_NAMES.index("DEW")
    scores = []
    fit_seconds = 0.
    print("Fitting independent RMGauss models", flush=True)
    for j, output in enumerate(weather_setup.OUTPUT_NAMES):
        observed = np.isfinite(y_train[:, j])
        model = RMGauss(optimize=False, rho_x=shared_rho_x, rho_y=shared_rho_y,
                      n_perm_fit=model_mohet.n_perm_fit)
        print(f"Fitting {output}: {observed.sum()} observed training values", flush=True)
        start = perf_counter()
        model.fit(X_train[observed], y_train[observed, j])
        model.preq_loglik_.block_until_ready()
        fit_seconds += perf_counter() - start
        # Native test_log_pdf uses predictive moments in original response units.
        score = np.asarray(model.test_log_pdf(split["X_test"], split["y_test"][:, j]), dtype=float)
        scores.append(score)
        if j == dew:
            # Other outcomes cannot inform DEW under this independent model.
            dew_scores = np.asarray(model.test_log_pdf(
                split["X_dew_test"], split["y_dew_test"][:, dew]), dtype=float)
        print(f"{output}: mean held-out log PDF {score.mean():.6f}", flush=True)

    independent_result = dict(
        mean_test_log_pdf=float(np.column_stack(scores).sum(axis=1).mean()),
        mean_dew_test_log_pdf=float(dew_scores.mean()),
        fit_seconds=fit_seconds, total_seconds=perf_counter() - total_start,
    )
    path = weather_setup.independent_gauss_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **independent_result)
    print(f"Mean joint held-out test log PDF: {independent_result['mean_test_log_pdf']:.6f}\n"
          f"Mean hidden-DEW test log PDF: {independent_result['mean_dew_test_log_pdf']:.6f}\n"
          f"Fit: {fit_seconds:.1f}s; total: {independent_result['total_seconds']:.1f}s; saved {path}",
          flush=True)
    return {"multi-output": multi_result, "independent": independent_result}


if __name__ == "__main__":
    results = run_weather()
