"""Print the weather comparison and write its LaTeX table from saved results."""

import re

import numpy as np
import pandas as pd

from weather import load_weather as weather_setup

OUTPUT_PATH = weather_setup.ROOT / "results" / "weather" / "weather_table.tex"


def read_results():
    paths = [
        ("RM (Multi-output)", weather_setup.rm_results_path()),
        ("RM (Independent)", weather_setup.independent_gauss_results_path()),
        *[(f"ICM (m={m})", weather_setup.icm_results_path(m)) for m in (64, 128, 256)],
        *[(label, weather_setup.gpar_results_path(m)) for label, m in
          [("GPAR (l=10)", 1000), ("GPAR (l=20)", 4000), ("GPAR (exact)", None)]],
    ]
    rows = []
    for label, path in paths:
        with np.load(path) as saved:
            rows.append({
                "Method": label,
                "Full": float(saved["mean_test_log_pdf"]),
                "DEW": float(saved["mean_dew_test_log_pdf"]),
                "Fit (s)": float(saved["fit_seconds"]) if "fit_seconds" in saved else np.nan,
                "Total (s)": float(saved["total_seconds"]) if "total_seconds" in saved else np.nan,
            })
    return pd.DataFrame(rows)


def write_latex_table(results):
    table = results.copy()
    table.loc[table["Method"] == "RM (Independent)", ["Fit (s)", "Total (s)"]] = np.nan
    best = {"Full": table["Full"].max(), "DEW": table["DEW"].max(),
            "Fit (s)": table["Fit (s)"].min(), "Total (s)": table["Total (s)"].min()}
    lines = [r"""\begin{table}[t]
\scriptsize
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.12}
\caption{Weather data results. ICM is run with 5000 optimizer steps. RM (Independent) plugs in the same hyperparameters as RM (Multi-output), so we do not report run-times.
Total run-time includes method-specific prediction time on the $50 \times 50$ longitude-latitude grid for the year 2021.}
\centering
\label{tab:weather_results}
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{Avg test log pdf}
& \multicolumn{2}{c}{Run-time (s)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method
& \multicolumn{1}{c}{Full}
& \multicolumn{1}{c}{DEW}
& \multicolumn{1}{c}{Fit}
& \multicolumn{1}{c}{Total} \\
\midrule"""]
    for _, row in table.iterrows():
        method = re.sub(r"\(([ml]=\d+)\)", r"($\1$)", row["Method"])
        if row["Method"] in ("RM (Independent)", "ICM (m=256)"):
            method += r" \vspace{1mm}"
        cells = [method]
        for column, digits in (("Full", 3), ("DEW", 3), ("Fit (s)", 0), ("Total (s)", 0)):
            value = row[column]
            text = "/" if pd.isna(value) else f"{value:.{digits}f}"
            if value == best[column]:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    results = read_results()
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    write_latex_table(results)
