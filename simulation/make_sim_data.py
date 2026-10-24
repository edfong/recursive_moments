from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import os
import numpy as np
import jax.numpy as jnp

from simulation.sim_utils import (
    pred_truth_het,
    pred_truth_kurt,
    pred_truth_mohet,
    pred_truth_mv,
    sample_X_het,
    sample_X_kurt,
    sample_X_mohet,
    sample_X_mv,
    sample_y_het,
    sample_y_kurt,
    sample_y_mohet,
    sample_y_mv,
)


def make_sim_data(setting,n):    
    # ------------------------------------------------------------------
    # Setup sim constants
    # ------------------------------------------------------------------

    if setting in ["het", "kurt", "mohet"]:
        X_range = (np.array([-7]), np.array([7]))
    elif setting in ["mv"]:
        X_range = (-2 * np.ones(5), 2 * np.ones(5))
    else:
        raise ValueError(f"Unknown setting: {setting}")

    n_plot_x = 50
    if setting in ["mv"]:
        x1, x2 = np.meshgrid(
            np.linspace(X_range[0][0], X_range[1][0], n_plot_x),
            np.linspace(X_range[0][1], X_range[1][1], n_plot_x),
            indexing="ij",
        )
        x_plot = np.column_stack(
            (x1.ravel(), x2.ravel(), np.zeros((n_plot_x**2, 3)))
        )
    else:
        x_plot = jnp.linspace(X_range[0], X_range[1], n_plot_x).reshape(-1, 1)
    n_plot_y = 200
    n_test = 2000

    # Seeds
    seeds = list(range(125, 145))  # 20 seeds

    # Save fixed x grid
    os.makedirs(ROOT / "data" / "simulation", exist_ok=True)
    np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_x_grid.csv", np.asarray(x_plot), delimiter=",")

    # Truth fixed over seeds because it depends only on x_plot.
    if setting == "het":
        mean_truth, var_truth, y_l_truth, y_h_truth = pred_truth_het(x_plot)
    elif setting == "kurt":
        mean_truth, var_truth, y_l_truth, y_h_truth = pred_truth_kurt(x_plot)
    elif setting == "mohet":
        (
            mean_truth,
            var_truth,
            y_l_truth,
            y_h_truth,
            cov_truth,
            corr_truth,
        ) = pred_truth_mohet(x_plot, return_dependence=True)
    elif setting in ["mv"]:
        mean_truth, var_truth, y_l_truth, y_h_truth = pred_truth_mv(x_plot)

    np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_mean_truth.csv", np.asarray(mean_truth), delimiter=",")
    np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_var_truth.csv", np.asarray(var_truth), delimiter=",")
    np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_y_l_truth.csv", np.asarray(y_l_truth), delimiter=",")
    np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_y_h_truth.csv", np.asarray(y_h_truth), delimiter=",")

    if setting == "mohet":
        # Columns are Sigma_11, Sigma_22, and Sigma_12 respectively.
        cov_components = np.column_stack(
            (cov_truth[:, 0, 0], cov_truth[:, 1, 1], cov_truth[:, 0, 1])
        )
        np.savetxt(
            f"{ROOT}/data/simulation/sim_{setting}_sd_truth.csv",
            np.sqrt(np.asarray(var_truth)),
            delimiter=",",
        )
        np.savetxt(
            f"{ROOT}/data/simulation/sim_{setting}_cov_truth.csv",
            cov_components,
            delimiter=",",
        )
        np.savetxt(
            f"{ROOT}/data/simulation/sim_{setting}_corr_truth.csv",
            np.asarray(corr_truth),
            delimiter=",",
        )

    # ------------------------------------------------------------------
    # Main seed loop
    # ------------------------------------------------------------------

    for seed in seeds:
        print(f"\n================ Seed {seed} ================\n")

        np.random.seed(seed)

        # --------------------------------------------------------------
        # Generate data
        # --------------------------------------------------------------
        if setting == "het":
            X = sample_X_het(n)
            y = sample_y_het(X)

            X_test = sample_X_het(n_test)
            y_test = sample_y_het(X_test)

        elif setting == "kurt":
            X = sample_X_kurt(n)
            y = sample_y_kurt(X)

            X_test = sample_X_kurt(n_test)
            y_test = sample_y_kurt(X_test)

        elif setting == "mohet":
            X = sample_X_mohet(n)
            y = sample_y_mohet(X)

            X_test = sample_X_mohet(n_test)
            y_test = sample_y_mohet(X_test)

        elif setting in ["mv"]:
            X = sample_X_mv(n)
            y = sample_y_mv(X)

            X_test = sample_X_mv(n_test)
            y_test = sample_y_mv(X_test)

        # y-grid is seed-specific because y range changes.
        if setting == "mohet":
            y_plot = np.column_stack(
                [
                    np.linspace(
                        np.min(y[:, j]) - np.std(y[:, j]),
                        np.max(y[:, j]) + np.std(y[:, j]),
                        n_plot_y,
                    )
                    for j in range(y.shape[1])
                ]
            )
        else:
            y_plot = jnp.linspace(
                np.min(y) - np.std(y),
                np.max(y) + np.std(y),
                n_plot_y,
            )

        # --------------------------------------------------------------
        # Save seed-specific data
        # --------------------------------------------------------------
        np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_X_seed{seed}_n{n}.csv", np.asarray(X), delimiter=",")
        np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_y_seed{seed}_n{n}.csv", np.asarray(y), delimiter=",")
        np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_X_test_seed{seed}_n{n}.csv", np.asarray(X_test), delimiter=",")
        np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_y_test_seed{seed}_n{n}.csv", np.asarray(y_test), delimiter=",")
        np.savetxt(f"{ROOT}/data/simulation/sim_{setting}_y_grid_seed{seed}_n{n}.csv", np.asarray(y_plot), delimiter=",")



# ------------------------------------------------------------------
# Choose settings
# ------------------------------------------------------------------
if __name__ == "__main__":
    make_sim_data("het", 200)
    make_sim_data("kurt", 200)
    make_sim_data("kurt", 1000)
    make_sim_data("mohet", 200)
    make_sim_data("mv", 600)
