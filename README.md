# Recursive Moment Updates for Bayesian Regression

This folder contains the code for the experiments for the paper "Recursive moment updates for scalable Bayesian nonparametric regression". Simulation data
and precomputed results are not distributed; please generate them using the scripts
below. The processed annual weather dataset and split are included.
Methodological and implementation details can be found in the supplementary material.

## Contents

- `recursive_moments/`: Gaussian, SU, copula and multi-output recursive models.
- `baseline/`: GP helpers and experiment-specific normalized-weights MATLAB code.
- `simulation/`: data generation, fitting, table generation, utilities and
  `plots.ipynb` for the three main-paper figures and Example 3.1 timing.
- `weather/`: preprocessing, fitting, plotting and result-reading utilities.
- `data/`: generated simulation inputs in `simulation/`; annual weather CSV,
  split, map and source metadata in `weather/`.
- `results/`: generated simulation outputs and LaTeX tables in `simulation/`,
  and weather summaries, scores, timings and LaTeX table in `weather/`.
- `plots/`: simulation figures in `simulation/` and weather maps in `weather/`.

## Installation

Run the following commands from the repository root. We recommend using a virtual environment:

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

Import the package with `from recursive_moments import RMGauss, RMSU, RMCopula, RMMOGauss`.

`requirements.txt` lists the main runtime and plotting dependencies.

## Simulations

First generate the simulation inputs:

```sh
python -m simulation.make_sim_data
```

Then run the main paper settings for RM (Gauss), RM (SU) and GP:

```sh
python -m simulation.run_sim
```

The slower copula runs can be uncommented in `simulation/run_sim.py`.

After completing these and the normalized-weights runs below, generate the
results table which is stored `results/simulation/simulation_tables.tex`:

```sh
python -m simulation.read_results
```

Simulation plots are produced in `simulation/plots.ipynb`. For the
normalized-weights baseline, use `baseline/NormalizedWeights/run_sim.m` (see below).

## Baselines

### Normalized weights

Download the [BayesXMix NormalizedWeights files](https://github.com/sarawade/BayesXMix/tree/main/NormalizedWeights)
into the existing `baseline/NormalizedWeights/` folder.
After generating the simulation data, run `run_sim.m` from that folder in MATLAB.

### GPflow and GPAR

For the weather GP baselines, install the following separately:

- [GPflow](https://gpflow.github.io/GPflow/develop/installation.html), with compatible
  TensorFlow and TensorFlow Probability versions.
- [GPAR](https://github.com/wesselb/gpar) and its dependencies, including PyTorch.

Separate environments may be useful for these baselines.

## Weather results

The curated US weather dataset, `weather_us_annual_2014_2023.csv`, contains
annual averages of temperature, dew point and sea-level pressure for each station
from 2014–2023, derived from [WEATHER-5K](https://github.com/taohan10200/WEATHER-5K).
See the main paper for more details.
The dataset and exact holdout/missingness split are included in `data/weather/`.
All methods use `weather/load_weather.py`. 

Fit the RM models first:

```sh
python -m weather.run_weather
```

Run the GP baselines with:

```sh
python -m weather.run_weather_GPs
```

After fitting all methods, generate the table and plots:

```sh
python -m weather.read_results
python -m weather.plot_weather
python -m weather.plot_weather_GPs
```

## Optional weather preprocessing

The supplied dataset suffices for the experiments. To rebuild it, download
the hourly data linked from the [WEATHER-5K repository](https://github.com/taohan10200/WEATHER-5K)
and place the station CSVs under `data/weather/WEATHER-5K/global_weather_stations/`, then run:

```sh
python -m weather.make_annual_us_dataset --output weather_rebuilt.csv
```

The fixed station list (`weather_source/us_stations.csv`) and
[NOAA ISD catalog](https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv) snapshot
(`weather_source/isd-history.csv`, retrieved 2026-09-08) are in `data/weather/`.
Additional preprocessing details are given in the supplementary material.

## References

- Wade, S. and Inácio, V. (2025). Bayesian dependent mixture models: A predictive
  comparison and survey. *Statistical Science*, 40(1), 81–108.
  [GitHub](https://github.com/sarawade/BayesXMix).
- Requeima, J., Tebbutt, W., Bruinsma, W. and Turner, R. E. (2019). The Gaussian
  process autoregressive regression model (GPAR). *AISTATS*, 1860–1869.
  [GitHub](https://github.com/wesselb/gpar).
- Han, T., Wen, Z., Chen, Z., Du, D., Guo, S. and Bai, L. (2026). Benchmarking
  Physics-Informed Time-Series Models for Operational Global Station Weather
  Forecasting. *ICML*.
  [WEATHER-5K GitHub](https://github.com/taohan10200/WEATHER-5K).
- Kelso, N. V. and Patterson, T. (2010). Introducing Natural Earth Data -
  naturalearthdata.com. *Geographia Technica*, 5, 82–89.
  [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/).
- Smith, A., Lott, N. and Vose, R. (2011). The integrated surface database: Recent
  developments and partnerships. *Bulletin of the American Meteorological Society*,
  92(6), 704–708.
  [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).
