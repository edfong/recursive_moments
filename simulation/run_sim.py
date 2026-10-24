from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import os
import numpy as np
import pandas as pd

from recursive_moments import RMGauss, RMCopula, RMSU
from simulation.sim_utils import (
    MSE,
    MSE_med,
    coverage,
    format_mean_se_table,
    interval_length,
    mean_test_logpdf,
    pred_truth_mv,
    summarize_results,
    test_logpdf_grid,
    time_call,
)
from baseline.GP_sim import fit_gp, mean_test_logpdf_gp


# ------------------------------------------------------------------
# Method registry/helpers
# ------------------------------------------------------------------

RM_METHODS = {
    "Gauss": RMGauss,
    "Cop": RMCopula,
    "SU": RMSU,
}

METHOD_ALIASES = {
    "Copula": "Cop",
    "RMCopula": "Cop",
    "RMGauss": "Gauss",
    "RMSU": "SU",
}
LOCAL_NW_TIMING_EXPERIMENTS = {
    ("kurt", 200),
    ("kurt", 1000),
    ("mv", 600),
}


def nw_runtime_path(setting, n):
    """Use local DDP timing for experiments with separate local timing results."""
    suffix = "_local" if (setting, n) in LOCAL_NW_TIMING_EXPERIMENTS else ""
    return f"{ROOT}/results/simulation/sim_{setting}_n{n}_runtime_summary_NW{suffix}.csv"


def normalize_methods(methods):
    """
    Normalize aliases and validate method names.
    """
    out = []
    valid = set(RM_METHODS.keys()) | {"NW", "GP"}

    for m in methods:
        m = METHOD_ALIASES.get(m, m)
        if m not in valid:
            raise ValueError(
                f"Unknown method '{m}'. Valid methods are {sorted(valid)} "
                f"plus aliases {sorted(METHOD_ALIASES.keys())}."
            )
        out.append(m)

    return out


def method_set_tag(methods):
    """Return a stable, filename-safe label for one comparison run."""
    return "-".join(method.lower() for method in methods)


def fit_rm_method(
    method,
    X,
    y,
    X_eval,
    y_plot,
    X_range,
    n_perm_opt,
    seed,
    predict=True,
    T=4000,
    B=1000,
):
    """
    Fit one RM method and return model, moments, and timing row.

    Handles the special predict/pr signatures for Copula. ``X_eval`` is the
    plotting grid except for multivariate simulations, which use ``X_test``.
    """
    ModelClass = RM_METHODS[method]

    model = ModelClass(
        optimize=True,
        n_perm_opt=n_perm_opt,
        n_perm_fit=10,
    )

    _, t_fit = time_call(lambda: model.fit(X, y))

    if method == "Cop" and predict:
        pred, t_pred = time_call(
            lambda: model.predict(
                X_eval,
                y_grid=y_plot,
                return_summaries=True,
            )
        )

        (pr_out, moments), t_pr = time_call(
            lambda: model.pr(
                X_test=X_eval,
                y_grid=y_plot,
                T=T,
                B=B,
                seed=126,
                return_intervals=True,
                X_range=X_range,
            )
        )

    elif predict:
        pred, t_pred = time_call(lambda: model.predict(X_eval))

        (_, moments), t_pr = time_call(
            lambda: model.pr(
                X_test=X_eval,
                T=T,
                B=B,
                seed=126,
                return_intervals=True,
                X_range=X_range,
            )
        )

    elif method == "Cop":
        pred = None
        t_pred = 0.0
        (pr_out, moments), t_pr = time_call(
            lambda: model.pr(
                X_test=X_eval,
                y_grid=y_plot,
                T=T,
                B=B,
                seed=126,
                return_intervals=True,
                X_range=X_range,
            )
        )

    else:
        pred = None
        t_pred = 0.0
        (_, moments), t_pr = time_call(
            lambda: model.pr(
                X_test=X_eval,
                T=T,
                B=B,
                seed=126,
                return_intervals=True,
                X_range=X_range,
            )
        )

    timing_row = {
        "seed": seed,
        "Method": method,
        "fit": t_fit,
        "predict": t_pred,
        "pr": t_pr,
    }

    return {
        "method": method,
        "model": model,
        "pred": pred,
        "moments": moments,
        "timing_row": timing_row,
    }


def fit_gp_method(X, y, X_eval, n_restarts_optimizer, seed):
    """Fit the Gaussian-process comparator through :mod:`GP_sim`."""
    out = fit_gp(
        X,
        y,
        X_eval,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )
    return {
        "method": "GP",
        "model": out["model"],
        "pred": None,
        "moments": out["moments"],
        "timing_row": {
            "seed": seed,
            "Method": "GP",
            "fit": out["fit_time"],
            "predict": 0.0,
            "pr": out["postprocessing_time"],
        },
    }


def load_nw_method(
    setting,
    n,
    seed,
    X_test,
    y_test,
    x_plot,
    y_plot,
):
    """
    Load precomputed NW outputs and return moments, test logpdf, and timing row.
    """
    runtime_NW = pd.read_csv(nw_runtime_path(setting, n))

    moments_NW = pd.read_csv(
        f"{ROOT}/results/simulation/sim_{setting}_seed{seed}_n{n}_moments_NW.csv"
    )

    logpdf_path = f"{ROOT}/results/simulation/sim_{setting}_seed{seed}_n{n}_test_logpdf_NW.csv"
    if os.path.exists(logpdf_path):
        test_logpdf_NW = float(np.mean(np.loadtxt(logpdf_path, delimiter=",")))
    else:
        pdf_NW = np.asarray(np.loadtxt(
            f"{ROOT}/results/simulation/sim_{setting}_seed{seed}_n{n}_pdf_NW.csv",
            delimiter=",",
        ))
        test_logpdf_NW = test_logpdf_grid(
            X_test=X_test[:, 0],
            y_test=y_test,
            x_plot=x_plot[:, 0],
            y_plot=y_plot,
            pdf_NW=pdf_NW,
        )

    runtime_seed_NW = runtime_NW.loc[runtime_NW["seed"] == seed]
    if runtime_seed_NW.empty and len(runtime_NW) == 1:
        runtime_seed_NW = runtime_NW
    runtime_seed_NW = runtime_seed_NW.iloc[0]

    nw_fit_time = float(runtime_seed_NW["posterior_sampling_seconds"])
    nw_predict_time = float(runtime_seed_NW["postprocessing_seconds"])

    timing_row = {
        "seed": seed,
        "Method": "NW",
        "fit": nw_fit_time,
        "predict": nw_predict_time,
        "pr": 0.0,
    }

    return {
        "method": "NW",
        "model": None,
        "pred": None,
        "moments": moments_NW,
        "test_logpdf": test_logpdf_NW,
        "timing_row": timing_row,
    }


def mse_for_method(method, moments, col, truth):
    """
    Evaluate the method-specific point estimates:
      - NW uses MSE for μ
      - NW uses MSE_med for σ2
      - RM methods use MSE
    """
    if method == "NW" and col != "μ":
        return MSE_med(moments, col, truth)
    return MSE(moments, col, truth)


def make_mse_df(seed, methods, results, truths):
    rows = []

    for method in methods:
        moments = results[method]["moments"]

        rows.append(
            {
                "seed": seed,
                "Method": method,
                "μ": mse_for_method(method, moments, "μ", truths["μ"]),
                "σ2": mse_for_method(method, moments, "σ2", truths["σ2"]),
            }
        )

    return pd.DataFrame(rows)


def make_coverage_df(seed, methods, results, truths):
    rows = []

    for method in methods:
        moments = results[method]["moments"]

        rows.append(
            {
                "seed": seed,
                "Method": method,
                "μ": coverage(moments, "μ", truths["μ"]),
                "σ2": coverage(moments, "σ2", truths["σ2"]),
            }
        )

    return pd.DataFrame(rows)


def make_interval_length_df(seed, methods, results):
    rows = []

    for method in methods:
        moments = results[method]["moments"]

        rows.append(
            {
                "seed": seed,
                "Method": method,
                "μ": interval_length(moments, "μ"),
                "σ2": interval_length(moments, "σ2"),
            }
        )

    return pd.DataFrame(rows)


def make_test_loglik_df(seed, methods, results, X_test, y_test):
    rows = []

    for method in methods:
        if method == "NW":
            test_logpdf = results[method]["test_logpdf"]
        elif method == "GP":
            test_logpdf = mean_test_logpdf_gp(
                results[method]["model"], X_test, y_test
            )
        else:
            test_logpdf = mean_test_logpdf(
                results[method]["model"],
                X_test,
                y_test,
            )

        rows.append(
            {
                "seed": seed,
                "Method": method,
                "test_logpdf": test_logpdf,
            }
        )

    return pd.DataFrame(rows)


def run_sim(setting, n, n_perm_opt, methods=None):
    # ------------------------------------------------------------------
    # Setup sim constants
    # ------------------------------------------------------------------

    if methods is None:
        methods = ["Gauss", "SU", "GP"]

    methods = normalize_methods(methods)
    methods_tag = method_set_tag(methods)

    is_mv = setting in ["mv"]
    if setting in ["het", "kurt"]:
        X_range = (np.array([-7]), np.array([7]))
    elif is_mv:
        X_range = (-2 * np.ones(5), 2 * np.ones(5))
    else:
        raise ValueError(f"Unknown setting: {setting}")

    x_plot = np.loadtxt(
        f"{ROOT}/data/simulation/sim_{setting}_x_grid.csv",
        delimiter=",",
    )
    if x_plot.ndim == 1:
        x_plot = x_plot.reshape(-1, 1)

    if not is_mv:
        truths = {
            "μ": np.loadtxt(f"{ROOT}/data/simulation/sim_{setting}_mean_truth.csv", delimiter=","),
            "σ2": np.loadtxt(f"{ROOT}/data/simulation/sim_{setting}_var_truth.csv", delimiter=","),
        }

    # Seeds
    seeds = list(range(125, 145))  # 20 seeds

    # ------------------------------------------------------------------
    # Storage across seeds
    # ------------------------------------------------------------------

    all_MSE = []
    all_coverage = []
    all_length = []
    all_test_loglik = []
    all_timing = []

    # ------------------------------------------------------------------
    # Main seed loop
    # ------------------------------------------------------------------

    for seed in seeds:
        print(f"\n================ Seed {seed} ================\n")

        np.random.seed(seed)

        # --------------------------------------------------------------
        # Load data
        # --------------------------------------------------------------

        X = np.loadtxt(
            f"{ROOT}/data/simulation/sim_{setting}_X_seed{seed}_n{n}.csv",
            delimiter=",",
        )
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = np.loadtxt(
            f"{ROOT}/data/simulation/sim_{setting}_y_seed{seed}_n{n}.csv",
            delimiter=",",
        )

        X_test = np.loadtxt(
            f"{ROOT}/data/simulation/sim_{setting}_X_test_seed{seed}_n{n}.csv",
            delimiter=",",
        )
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        y_test = np.loadtxt(
            f"{ROOT}/data/simulation/sim_{setting}_y_test_seed{seed}_n{n}.csv",
            delimiter=",",
        )

        y_plot = np.loadtxt(
            f"{ROOT}/data/simulation/sim_{setting}_y_grid_seed{seed}_n{n}.csv",
            delimiter=",",
        )

        if is_mv:
            mean_truth, var_truth, _, _ = pred_truth_mv(X_test)
            truths = {
                "μ": mean_truth,
                "σ2": var_truth,
            }

        X_eval = X_test if is_mv else x_plot

        # --------------------------------------------------------------
        # Fit/load requested methods
        # --------------------------------------------------------------

        results = {}
        timing_rows = []

        for method in methods:
            print(f"Running method: {method}")

            if method == "NW":
                out = load_nw_method(
                    setting=setting,
                    n=n,
                    seed=seed,
                    X_test=X_test,
                    y_test=y_test,
                    x_plot=x_plot,
                    y_plot=y_plot,
                )
            elif method == "GP":
                out = fit_gp_method(
                    X=X,
                    y=y,
                    X_eval=X_eval,
                    n_restarts_optimizer=n_perm_opt,
                    seed=seed,
                )
            else:
                out = fit_rm_method(
                    method=method,
                    X=X,
                    y=y,
                    X_eval=X_eval,
                    y_plot=y_plot,
                    X_range=X_range,
                    n_perm_opt=n_perm_opt,
                    seed=seed,
                    predict=not is_mv,
                    T=4000,
                    B=1000,
                )

            results[method] = out
            timing_rows.append(out["timing_row"])

        # --------------------------------------------------------------
        # Evaluation tables for this seed
        # --------------------------------------------------------------

        MSE_df = make_mse_df(
            seed=seed,
            methods=methods,
            results=results,
            truths=truths,
        )

        coverage_df = make_coverage_df(
            seed=seed,
            methods=methods,
            results=results,
            truths=truths,
        )

        interval_length_df = make_interval_length_df(
            seed=seed,
            methods=methods,
            results=results,
        )

        test_loglik_df = make_test_loglik_df(
            seed=seed,
            methods=methods,
            results=results,
            X_test=X_test,
            y_test=y_test,
        )

        timing_df = pd.DataFrame(timing_rows)
        timing_df["total"] = timing_df[["fit", "predict", "pr"]].sum(axis=1)

        # Store
        all_MSE.append(MSE_df)
        all_coverage.append(coverage_df)
        all_length.append(interval_length_df)
        all_test_loglik.append(test_loglik_df)
        all_timing.append(timing_df)

    # ------------------------------------------------------------------
    # Combine all seeds
    # ------------------------------------------------------------------

    MSE_all = pd.concat(all_MSE, ignore_index=True)
    coverage_all = pd.concat(all_coverage, ignore_index=True)
    length_all = pd.concat(all_length, ignore_index=True)
    test_loglik_all = pd.concat(all_test_loglik, ignore_index=True)
    timing_all = pd.concat(all_timing, ignore_index=True)

    # Save each comparison separately, so runs with different method sets
    # (e.g., fast methods versus slower methods) never overwrite one
    # another.
    os.makedirs(ROOT / "results" / "simulation", exist_ok=True)
    results_prefix = f"{ROOT}/results/simulation/sim_{setting}_n{n}_methods_{methods_tag}"
    MSE_all.to_csv(f"{results_prefix}_MSE_all.csv", index=False)
    coverage_all.to_csv(f"{results_prefix}_coverage_all.csv", index=False)
    length_all.to_csv(
        f"{results_prefix}_interval_length_all.csv", index=False
    )
    test_loglik_all.to_csv(
        f"{results_prefix}_test_loglik_all.csv", index=False
    )
    timing_all.to_csv(f"{results_prefix}_timing_all.csv", index=False)
    run_metadata = pd.DataFrame(
        {
            "setting": setting,
            "n": n,
            "method_set": methods_tag,
            "method": methods,
        }
    )
    run_metadata.to_csv(f"{results_prefix}_run_metadata.csv", index=False)

    # ------------------------------------------------------------------
    # Average result tables with standard errors
    # ------------------------------------------------------------------

    MSE_summary = summarize_results(MSE_all)
    coverage_summary = summarize_results(coverage_all)
    length_summary = summarize_results(length_all)
    test_loglik_summary = summarize_results(test_loglik_all)
    timing_summary = summarize_results(timing_all)

    MSE_table = format_mean_se_table(
        MSE_summary,
        ["μ", "σ2"],
        mean_fmt="{:.2g}",
        se_fmt="{:.1g}",
    )

    coverage_table = format_mean_se_table(
        coverage_summary,
        ["μ", "σ2"],
        mean_fmt="{:.2f}",
        se_fmt="{:.2f}",
    )

    length_table = format_mean_se_table(
        length_summary,
        ["μ", "σ2"],
        mean_fmt="{:.2g}",
        se_fmt="{:.1g}",
    )

    test_loglik_table = format_mean_se_table(
        test_loglik_summary,
        ["test_logpdf"],
        mean_fmt="{:.3f}",
        se_fmt="{:.2f}",
    )

    timing_table = format_mean_se_table(
        timing_summary,
        ["fit", "predict", "pr", "total"],
        mean_fmt="{:.2f}",
        se_fmt="{:.2f}",
    )

    # ------------------------------------------------------------------
    # Print average results
    # ------------------------------------------------------------------

    print("\n\n================ Average over seeds ================\n")

    print("Methods:", methods)
    print("\n")

    print("MSE: mean (SE)")
    print(MSE_table.to_string(index=False))
    print("\n")

    print("Coverage: mean (SE)")
    print(coverage_table.to_string(index=False))
    print("\n")

    print("Interval length: mean (SE)")
    print(length_table.to_string(index=False))
    print("\n")

    print("Test log pdf: mean (SE)")
    print(test_loglik_table.to_string(index=False))
    print("\n")

    print("Timing seconds: mean (SE)")
    print(timing_table.to_string(index=False))

    return {
        "MSE_all": MSE_all,
        "coverage_all": coverage_all,
        "length_all": length_all,
        "test_loglik_all": test_loglik_all,
        "timing_all": timing_all,
        "MSE_table": MSE_table,
        "coverage_table": coverage_table,
        "length_table": length_table,
        "test_loglik_table": test_loglik_table,
        "timing_table": timing_table,
        "methods": methods,
        "methods_tag": methods_tag,
        "results_prefix": results_prefix,
    }


# ------------------------------------------------------------------
# Choose setting/methods
# ------------------------------------------------------------------

if __name__ == "__main__":
    methods = ["Gauss", "SU", "GP"]
    run_sim("kurt", 200, 10, methods=methods)
    run_sim("kurt", 1000, 2, methods=methods)
    run_sim("mv", 600, 2, methods=methods)

    # The copula runs are run separately as they are much slower
    # methods = ["Cop"]
    # run_sim("kurt", 200, 10, methods=methods)
    # run_sim("kurt", 1000, 2, methods=methods)
