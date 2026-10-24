"""Combine saved simulation results and write CSV summaries and LaTeX tables."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

from simulation.sim_utils import (
    MSE,
    MSE_med,
    coverage,
    format_mean_se_table,
    interval_length,
    pred_truth_mv,
    summarize_results,
    test_logpdf_grid,
)


METHOD_ORDER = ["Gauss", "SU", "GP", "Cop", "NW"]
FAST_METHODS_TAG = "gauss-su-gp"
COP_METHODS_TAG = "cop"
METRICS = {
    "MSE": ["μ", "σ2"],
    "coverage": ["μ", "σ2"],
    "interval_length": ["μ", "σ2"],
    "test_loglik": ["test_logpdf"],
    "timing": ["fit", "predict", "pr", "total"],
}
LOCAL_NW_TIMING_EXPERIMENTS = {
    ("kurt", 200),
    ("kurt", 1000),
    ("mv", 600),
}
LOCAL_NW_TIMING_SEEDS = {125}

METHODS = ["GP", "NW", "Cop", "Gauss", "SU"]
METHOD_LABELS = {
    "GP": "GP",
    "NW": "DDP",
    "Cop": "Cop",
    "Gauss": "RM (Gauss)",
    "SU": "RM (SU)",
}
TABLE_METHODS = {
    "het": ["GP", "NW", "Cop", "Gauss"],
    "kurt": METHODS,
    "mv": ["GP", "NW", "Gauss", "SU"],
}
EXPECTED_SEEDS = set(range(125, 145))
RESULTS_DIR = ROOT / "results" / "simulation"
OUTPUT_PATH = RESULTS_DIR / "simulation_tables.tex"


def _nw_runtime_path(results_dir, setting, n):
    """Return the intended DDP timing file for an experiment."""
    suffix = "_local" if (setting, n) in LOCAL_NW_TIMING_EXPERIMENTS else ""
    return Path(results_dir) / f"sim_{setting}_n{n}_runtime_summary_NW{suffix}.csv"


def _load_method_set(results_dir, setting, n, methods_tag, result_name):
    """Load one table written by run_sim for a specified method set."""
    path = (
        Path(results_dir)
        / f"sim_{setting}_n{n}_methods_{methods_tag}_{result_name}_all.csv"
    )
    if not path.exists():
        return pd.DataFrame(), path
    table = pd.read_csv(path)
    # Select the paper's metrics.
    columns = ["seed", "Method"] + [
        name for name in METRICS[result_name] if name in table
    ]
    return table[columns], path


def _truths(setting, X_test=None, data_dir=ROOT / "data" / "simulation"):
    """Return the truth at the appropriate evaluation covariates."""
    if setting in ["mv"]:
        if X_test is None:
            raise ValueError("X_test is required for the multivariate setting.")
        mean, var, _, _ = pred_truth_mv(X_test)
        return {"μ": mean, "σ2": var}

    data_dir = Path(data_dir)
    return {
        "μ": np.loadtxt(data_dir / f"sim_{setting}_mean_truth.csv", delimiter=","),
        "σ2": np.loadtxt(data_dir / f"sim_{setting}_var_truth.csv", delimiter=","),
    }


def _load_nw_rows(setting, n, seeds, data_dir=ROOT / "data" / "simulation", results_dir=RESULTS_DIR):
    """Compute NW metrics from its raw MATLAB outputs."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    is_mv = setting in ["mv"]

    runtime_path = _nw_runtime_path(results_dir, setting, n)
    if (setting, n) in LOCAL_NW_TIMING_EXPERIMENTS and not runtime_path.exists():
        raise FileNotFoundError(
            f"Local DDP timing file not found: {runtime_path}. "
            "Run baseline/NormalizedWeights/run_sim.m locally first."
        )
    runtime = pd.read_csv(runtime_path) if runtime_path.exists() else None
    if runtime is not None and (setting, n) in LOCAL_NW_TIMING_EXPERIMENTS:
        if runtime.duplicated("seed").any() or set(runtime["seed"]) != LOCAL_NW_TIMING_SEEDS:
            raise ValueError(
                f"{runtime_path} must contain exactly the local timing seed "
                f"{sorted(LOCAL_NW_TIMING_SEEDS)}"
            )
    rows = {name: [] for name in METRICS}
    missing = []

    for seed in seeds:
        moments_path = results_dir / f"sim_{setting}_seed{seed}_n{n}_moments_NW.csv"
        logpdf_path = results_dir / f"sim_{setting}_seed{seed}_n{n}_test_logpdf_NW.csv"
        if not moments_path.exists():
            missing.append(f"moments (seed {seed})")
            continue

        X_test = np.loadtxt(
            data_dir / f"sim_{setting}_X_test_seed{seed}_n{n}.csv", delimiter=","
        )
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        y_test = np.loadtxt(
            data_dir / f"sim_{setting}_y_test_seed{seed}_n{n}.csv", delimiter=","
        )
        moments = pd.read_csv(moments_path)
        truths = _truths(setting, X_test=X_test, data_dir=data_dir)

        rows["MSE"].append(
            {
                "seed": seed,
                "Method": "NW",
                "μ": MSE(moments, "μ", truths["μ"]),
                "σ2": MSE_med(moments, "σ2", truths["σ2"]),
            }
        )
        rows["coverage"].append(
            {
                "seed": seed,
                "Method": "NW",
                "μ": coverage(moments, "μ", truths["μ"]),
                "σ2": coverage(moments, "σ2", truths["σ2"]),
            }
        )
        rows["interval_length"].append(
            {
                "seed": seed,
                "Method": "NW",
                "μ": interval_length(moments, "μ"),
                "σ2": interval_length(moments, "σ2"),
            }
        )

        if logpdf_path.exists():
            test_logpdf = float(np.mean(np.loadtxt(logpdf_path, delimiter=",")))
        elif not is_mv:
            x_plot = np.loadtxt(
                data_dir / f"sim_{setting}_x_grid.csv", delimiter=","
            ).reshape(-1, 1)
            y_grid = np.loadtxt(
                data_dir / f"sim_{setting}_y_grid_seed{seed}_n{n}.csv",
                delimiter=",",
            )
            pdf = np.loadtxt(
                results_dir / f"sim_{setting}_seed{seed}_n{n}_pdf_NW.csv",
                delimiter=",",
            )
            test_logpdf = test_logpdf_grid(
                X_test=X_test[:, 0],
                y_test=y_test,
                x_plot=x_plot[:, 0],
                y_plot=y_grid,
                pdf_NW=np.asarray(pdf),
            )
        else:
            missing.append(f"test log pdf (seed {seed})")
            test_logpdf = np.nan

        rows["test_loglik"].append(
            {"seed": seed, "Method": "NW", "test_logpdf": test_logpdf}
        )

        if runtime is not None:
            runtime_seed = runtime.loc[runtime["seed"] == seed]
            if not runtime_seed.empty:
                runtime_seed = runtime_seed.iloc[0]
                fit = float(runtime_seed["posterior_sampling_seconds"])
                predict = float(runtime_seed["postprocessing_seconds"])
                rows["timing"].append(
                    {
                        "seed": seed,
                        "Method": "NW",
                        "fit": fit,
                        "predict": predict,
                        "pr": 0.0,
                        "total": fit + predict,
                    }
                )

    if runtime is None:
        missing.append("runtime summary")
    return {name: pd.DataFrame(values) for name, values in rows.items()}, missing


def _combine_method_tables(setting, n, results_dir=RESULTS_DIR):
    """Collect the fast run and Cop-only run for every metric."""
    collected = {name: [] for name in METRICS}
    missing_files = []
    for methods_tag in ((FAST_METHODS_TAG,) if setting == "mv" else (FAST_METHODS_TAG, COP_METHODS_TAG)):
        for result_name in METRICS:
            table, path = _load_method_set(
                results_dir, setting, n, methods_tag, result_name
            )
            if table.empty:
                missing_files.append(path.name)
            else:
                collected[result_name].append(table)

    output = {}
    for result_name, tables in collected.items():
        output[result_name] = (
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        )
    return output, missing_files


def _order_and_validate(table, method_order):
    """Check for duplicate seed/method entries and apply a consistent order."""
    if table.empty:
        return table

    duplicates = table.duplicated(["seed", "Method"], keep=False)
    if duplicates.any():
        duplicate_rows = table.loc[duplicates, ["seed", "Method"]]
        raise ValueError(
            "Duplicate method results found:\n"
            + duplicate_rows.to_string(index=False)
        )

    table = table[table["Method"].isin(method_order)].copy()
    table["Method"] = pd.Categorical(
        table["Method"], categories=method_order, ordered=True
    )
    return table.sort_values(["Method", "seed"]).reset_index(drop=True)


def read_results(
    setting,
    n,
    seeds=range(125, 145),
    data_dir=ROOT / "data" / "simulation",
    results_dir=RESULTS_DIR,
    method_order=METHOD_ORDER,
    save_final=True,
):
    """Create final tables from fast, Cop-only, and raw NW simulation outputs."""
    seeds = list(seeds)
    method_order = list(method_order)
    print(f"\n================ Setting {setting}, n={n} ================")

    processed, missing_files = _combine_method_tables(setting, n, results_dir)
    nw_tables, nw_missing = _load_nw_rows(
        setting, n, seeds, data_dir=data_dir, results_dir=results_dir
    )

    final_tables = {}
    for result_name in METRICS:
        parts = [processed[result_name], nw_tables[result_name]]
        table = pd.concat(
            [part for part in parts if not part.empty], ignore_index=True
        )
        if result_name == "timing" and not table.empty:
            if "pr" not in table:
                table["pr"] = 0.0
            if "total" not in table:
                table["total"] = table[["fit", "predict", "pr"]].sum(axis=1)
        final_tables[result_name] = _order_and_validate(table, method_order)

    found_methods = set(final_tables["MSE"]["Method"].astype(str))
    missing_methods = [method for method in method_order if method not in found_methods]
    if missing_files:
        print("Missing processed files:")
        for filename in sorted(set(missing_files)):
            print(f"  - {filename}")
    if nw_missing:
        print("Missing NW outputs:")
        for item in sorted(set(nw_missing)):
            print(f"  - {item}")
    if missing_methods:
        print("Methods absent from final comparison:", ", ".join(missing_methods))

    results_dir = Path(results_dir)
    final_prefix = results_dir / f"sim_{setting}_n{n}_methods_final"
    if save_final:
        for result_name, table in final_tables.items():
            table.to_csv(f"{final_prefix}_{result_name}_all.csv", index=False)

    summaries = {
        name: summarize_results(table)
        for name, table in final_tables.items()
        if not table.empty
    }
    formatted_tables = {
        "MSE": format_mean_se_table(
            summaries["MSE"], METRICS["MSE"], mean_fmt="{:.2g}", se_fmt="{:.1g}"
        ),
        "coverage": format_mean_se_table(
            summaries["coverage"],
            METRICS["coverage"],
            mean_fmt="{:.2f}",
            se_fmt="{:.2f}",
        ),
        "interval_length": format_mean_se_table(
            summaries["interval_length"],
            METRICS["interval_length"],
            mean_fmt="{:.2g}",
            se_fmt="{:.1g}",
        ),
        "test_loglik": format_mean_se_table(
            summaries["test_loglik"],
            METRICS["test_loglik"],
            mean_fmt="{:.3f}",
            se_fmt="{:.2f}",
        ),
        "timing": format_mean_se_table(
            summaries["timing"],
            METRICS["timing"],
            mean_fmt="{:.2f}",
            se_fmt="{:.2f}",
        ),
    }

    for name, table in formatted_tables.items():
        print(f"\n{name.replace('_', ' ').title()}: mean (SE)")
        print(table.to_string(index=False))

    return {
        **final_tables,
        "summaries": summaries,
        "formatted_tables": formatted_tables,
        "missing_methods": missing_methods,
        "missing_files": missing_files,
        "nw_missing": nw_missing,
    }


def build_all_final_tables():
    """Build all requested experiment tables."""
    experiments = [
        ("kurt", 200),
        ("kurt", 1000),
        ("mv", 600),
    ]
    return {
        (setting, n): read_results(
            setting, n,
            method_order=[m for m in METHOD_ORDER if not (setting == "mv" and m == "Cop")],
        )
        for setting, n in experiments
    }


def load_summaries(setting, n, methods):
    """Validate and summarize the final CSVs just written by read_results."""
    summaries = {}
    for kind in METRICS:
        path = RESULTS_DIR / f"sim_{setting}_n{n}_methods_final_{kind}_all.csv"
        table = pd.read_csv(path)
        table = table[table["Method"].isin(methods)].copy()

        duplicates = table.duplicated(["seed", "Method"])
        if duplicates.any():
            raise ValueError(f"Duplicate seed/method rows in {path}")
        for method in methods:
            seeds = set(table.loc[table["Method"] == method, "seed"])
            expected = (
                LOCAL_NW_TIMING_SEEDS
                if kind == "timing" and method == "NW"
                and (setting, n) in LOCAL_NW_TIMING_EXPERIMENTS
                else EXPECTED_SEEDS
            )
            if seeds != expected:
                raise ValueError(
                    f"{path}: {method} has seeds {sorted(seeds)}, "
                    f"expected {sorted(expected)}"
                )

        value_columns = (
            ["μ", "σ2"] if kind in {"MSE", "coverage", "interval_length"}
            else [c for c in table if c not in {"seed", "Method"}]
        )
        summaries[kind] = table.groupby("Method")[value_columns].agg(
            ["mean", "sem"]
        )
    return summaries


def format_significant(value, digits):
    """Format a finite number with at most ``digits`` significant figures."""
    value = float(value)
    if value == 0:
        return "0.0"
    decimals = digits - 1 - int(np.floor(np.log10(abs(value))))
    rounded = round(value, decimals)
    if rounded != 0:
        decimals = digits - 1 - int(np.floor(np.log10(abs(rounded))))
    return f"{rounded:,.{max(decimals, 0)}f}"


def decimal_padding(text, places):
    """Return invisible padding for decimal places omitted from ``text``."""
    decimals = len(text.rsplit(".", 1)[1]) if "." in text else 0
    if decimals == 0:
        return rf"\hphantom{{.{'0' * places}}}"
    missing = places - decimals
    return "" if missing == 0 else rf"\hphantom{{{'0' * missing}}}"


def maybe_bold(text, bold):
    """Bold visible text while retaining its normal-font alignment width."""
    if not bold:
        return text
    return rf"\makebox[0pt][l]{{\textbf{{{text}}}}}\hphantom{{{text}}}"


def mean_se(
    summary,
    method,
    column,
    scale=1.0,
    decimals=2,
    sem_significant_digits=None,
    sem_alignment_decimals=None,
    bold=False,
):
    mean = summary.loc[method, (column, "mean")] * scale
    sem = summary.loc[method, (column, "sem")] * scale
    sem_text = (
        format_significant(sem, sem_significant_digits)
        if sem_significant_digits is not None
        else f"{sem:.{decimals}f}"
    )
    sem_padding = (
        decimal_padding(sem_text, sem_alignment_decimals)
        if sem_alignment_decimals is not None
        else ""
    )
    value = f"{mean:.{decimals}f} ({sem_text}){sem_padding}"
    return maybe_bold(value, bold)


def runtime(summary, method, bold=False):
    value = summary.loc[method, ("total", "mean")]
    if value > 10:
        text = f"{value:.0f}"
    else:
        text = f"{value:,.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def best_methods(summaries, methods):
    """Best estimation/prediction values; UQ columns are not ranked alone."""
    return {
        "mse_mu": summaries["MSE"].loc[methods, ("μ", "mean")].idxmin(),
        "mse_var": summaries["MSE"].loc[methods, ("σ2", "mean")].idxmin(),
        "log_score": summaries["test_loglik"]
        .loc[methods, ("test_logpdf", "mean")]
        .idxmax(),
        "runtime": summaries["timing"].loc[methods, ("total", "mean")].idxmin(),
    }


def panel_rows(setting, n, mse_mu_scale, mse_var_scale, methods):
    summaries = load_summaries(setting, n, methods)
    best = best_methods(summaries, methods)
    rows = [rf"\multicolumn{{9}}{{l}}{{\textit{{$n={n}$}}}} \\"]
    for method in methods:
        values = [
            METHOD_LABELS[method],
            mean_se(
                summaries["MSE"], method, "μ", mse_mu_scale, 2,
                bold=method == best["mse_mu"],
            ),
            mean_se(
                summaries["MSE"], method, "σ2", mse_var_scale, 2,
                bold=method == best["mse_var"],
            ),
            mean_se(
                summaries["coverage"], method, "μ",
                decimals=2, sem_significant_digits=1,
                sem_alignment_decimals=3,
            ),
            mean_se(
                summaries["interval_length"], method, "μ",
                decimals=2, sem_significant_digits=1,
                sem_alignment_decimals=3,
            ),
            "/" if method == "GP" else mean_se(
                summaries["coverage"], method, "σ2",
                decimals=2, sem_significant_digits=1,
                sem_alignment_decimals=3,
            ),
            "/" if method == "GP" else mean_se(
                summaries["interval_length"], method, "σ2",
                decimals=2, sem_significant_digits=1,
                sem_alignment_decimals=3,
            ),
            mean_se(
                summaries["test_loglik"], method, "test_logpdf",
                decimals=2, sem_significant_digits=1,
                sem_alignment_decimals=3,
                bold=method == best["log_score"],
            ),
            runtime(summaries["timing"], method, method == best["runtime"]),
        ]
        rows.append(" & ".join(values) + r" \\")
    return rows


def make_table(
    setting,
    title,
    label,
    sample_sizes,
    mse_mu_power,
    mse_var_power,
):
    methods = TABLE_METHODS[setting]
    mu_scale = 10.0**mse_mu_power
    var_scale = 10.0**mse_var_power
    if mse_mu_power:
        mu_header = rf"$\mathrm{{MSE}}_\mu\times 10^{{{mse_mu_power}}}$"
    else:
        mu_header = r"$\mathrm{MSE}_\mu$"
    if mse_var_power:
        var_header = rf"$\mathrm{{MSE}}_{{\sigma^2}}\times 10^{{{mse_var_power}}}$"
    else:
        var_header = r"$\mathrm{MSE}_{\sigma^2}$"

    rows = []
    for index, n in enumerate(sample_sizes):
        if index:
            rows.append(r"\addlinespace[4pt]")
        rows += panel_rows(setting, n, mu_scale, var_scale, methods)

    caption = (
        f"{title}. Boldface marks the smallest MSE and run-time and the largest "
        "held-out log score for each sample size; standard errors in "
        "parentheses."
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (
            r"& \multicolumn{2}{c}{Estimation} "
            r"& \multicolumn{2}{c}{Mean UQ} "
            r"& \multicolumn{2}{c}{Variance UQ} "
            r"& \multicolumn{1}{c}{Prediction} "
            r"& \multicolumn{1}{c}{Run-time} \\"
        ),
        (
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
            r"\cmidrule(lr){8-8}\cmidrule(lr){9-9}"
        ),
        (
            rf"Method & \multicolumn{{1}}{{c}}{{{mu_header}}} "
            rf"& \multicolumn{{1}}{{c}}{{{var_header}}} "
            r"& \multicolumn{1}{c}{$\mathrm{Cov}_{\mu}$} "
            r"& \multicolumn{1}{c}{$\mathrm{Len}_{\mu}$} "
            r"& \multicolumn{1}{c}{$\mathrm{Cov}_{\sigma^2}$} "
            r"& \multicolumn{1}{c}{$\mathrm{Len}_{\sigma^2}$} "
            r"& \multicolumn{1}{c}{Avg $\log p(y_{\mathrm{test}})$} "
            r"& \multicolumn{1}{c}{Total (s)} \\"
        ),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def write_latex_tables():
    """Write the publication tables using the combined simulation results."""
    tables = [
        "% Requires \\usepackage{booktabs,graphicx}",
        make_table(
            "kurt",
            "Heavy-tailed contamination simulation",
            "tab:sim-kurt",
            sample_sizes=[200, 1000],
            mse_mu_power=2,
            mse_var_power=0,
        ),
        make_table(
            "mv",
            "Multivariate simulation",
            "tab:sim-mv",
            sample_sizes=[600],
            mse_mu_power=1,
            mse_var_power=1,
        ),
    ]
    OUTPUT_PATH.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")



def main():
    build_all_final_tables()
    write_latex_tables()


if __name__ == "__main__":
    main()
