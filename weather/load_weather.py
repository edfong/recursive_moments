"""Shared weather settings, paths, data, holdout and prediction grid.
Run this script to prepare the split without fitting a model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START_YEAR = 2014
END_YEAR = 2023
YEAR_RANGE = f"{START_YEAR}_{END_YEAR}"
INPUT_NAMES = ["LONGITUDE", "LATITUDE", "YEAR"]
OUTPUT_NAMES = ["TMP", "SLP", "DEW"]
PREDICT_YEAR = 2021
GRID_SIZE = 50
GRID_BOUNDS = (-126, 23, -65, 51)  # west, south, east, north
TEST_FRACTION = 0.20
DEW_MISSING_FRACTION = 0.20
MISSING_TAG = f"deweast{100 * DEW_MISSING_FRACTION:.0f}"
SPLIT_SEED = 125
SEED = 125  # Model fitting/resampling seed, independent of the split RNG.

DATA_PATH = ROOT / "data" / "weather" / f"weather_us_annual_{YEAR_RANGE}.csv"
SPLIT_PATH = ROOT / "data" / "weather" / (
    f"weather_us_annual_{YEAR_RANGE}_holdout{100 * TEST_FRACTION:.0f}"
    f"_{MISSING_TAG}_seed{SPLIT_SEED}.npz"
)
WORLD_PATH = ROOT / "data" / "weather" / "ne_110m_admin_0_countries.zip"


def rm_results_path():
    return ROOT / "results" / "weather" / f"weather_us_{YEAR_RANGE}_predict_{PREDICT_YEAR}_{MISSING_TAG}.npz"


def icm_results_path(num_inducing):
    name = f"m{int(num_inducing)}"
    return ROOT / "results" / "weather" / f"weather_icm_us_{YEAR_RANGE}_predict_{PREDICT_YEAR}_{MISSING_TAG}_{name}.npz"


def independent_gauss_results_path():
    return ROOT / "results" / "weather" / f"weather_independent_rm_us_{YEAR_RANGE}_{MISSING_TAG}.npz"


def gpar_results_path(num_inducing):
    name = "exact" if num_inducing is None else f"m{num_inducing}"
    return ROOT / "results" / "weather" / f"weather_gpar_us_{YEAR_RANGE}_{MISSING_TAG}_{name}.npz"


def load_weather_data():
    weather = pd.read_csv(DATA_PATH, dtype={"STATION": str})
    weather = weather.loc[weather["YEAR"].between(START_YEAR, END_YEAR)].copy()
    X = weather[INPUT_NAMES].to_numpy(dtype=float)
    y = weather[OUTPUT_NAMES].to_numpy(dtype=float)
    return weather, X, y


def load_weather_split(weather, X, y):
    """Hold out 20% of rows, then 20% of DEW where training longitude > -100.

    X_test/y_test are wholly held-out rows. X_dew_test/y_dew_test retain the
    triplets needed to score DEW given TMP/SLP; only their DEW is hidden in training.
    Saved indices refer to the original CSV, not positions within the training set.
    """
    if SPLIT_PATH.exists():
        with np.load(SPLIT_PATH) as saved:
            train_indices, test_indices = saved["train_indices"], saved["test_indices"]
            dew_test_indices = saved["dew_test_indices"]
    else:
        train_indices, test_indices = split_rows(weather)
        east = train_indices[X[train_indices, 0] > -100.]
        dew_test_indices = np.sort(np.random.default_rng(SPLIT_SEED).choice(
            east, size=round(DEW_MISSING_FRACTION * len(east)), replace=False))
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SPLIT_PATH, train_indices=train_indices, test_indices=test_indices,
                            dew_test_indices=dew_test_indices, split_seed=SPLIT_SEED,
                            test_fraction=TEST_FRACTION, dew_missing_fraction=DEW_MISSING_FRACTION,
                            censor_rule="LONGITUDE > -100")

    y_train = y[train_indices].copy()
    y_train[np.isin(train_indices, dew_test_indices), OUTPUT_NAMES.index("DEW")] = np.nan
    station_ids = weather["STATION"].to_numpy(dtype=str)
    return dict(
        X_train=X[train_indices], y_train=y_train,
        X_test=X[test_indices], y_test=y[test_indices],
        X_dew_test=X[dew_test_indices], y_dew_test=y[dew_test_indices],
        station_ids_train=station_ids[train_indices],
        split_seed=SPLIT_SEED, split_path=str(SPLIT_PATH.resolve()),
    )


def split_rows(weather):
    """Seed-125 80/20 split, stratified by year."""
    rng = np.random.default_rng(SPLIT_SEED)
    years = weather["YEAR"].to_numpy()
    unique_years, counts = np.unique(years, return_counts=True)
    # Keep the requested total test size, allocating rounding remainders across years.
    target = counts * TEST_FRACTION
    test_counts = np.floor(target).astype(int)
    extra = round(len(weather) * TEST_FRACTION) - test_counts.sum()
    order = rng.permutation(len(counts))
    order = order[np.argsort(-(target - test_counts)[order], kind="stable")]
    test_counts[order[:extra]] += 1
    test_indices = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(years == year), size=count, replace=False)
        for year, count in zip(unique_years, test_counts)
    ]))
    train_indices = np.setdiff1d(np.arange(len(weather)), test_indices)
    return train_indices, test_indices


def make_prediction_grid():
    west, south, east, north = GRID_BOUNDS
    longitude = np.linspace(west, east, GRID_SIZE)
    latitude = np.linspace(south, north, GRID_SIZE)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude, indexing="xy")
    grid = np.column_stack([
        lon_grid.ravel(), lat_grid.ravel(), np.full(lon_grid.size, PREDICT_YEAR),
    ])
    return longitude, latitude, grid


if __name__ == "__main__":
    weather, X, y = load_weather_data()
    split = load_weather_split(weather, X, y)
    print(f"Training rows: {len(split['X_train'])}; held-out rows: {len(split['X_test'])}; "
          f"hidden training DEW: {len(split['X_dew_test'])}")
    print(f"Split: {split['split_path']}")
