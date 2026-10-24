"""Plot the paper's TMP mean-function and DEW correlation maps.

Requires geopandas and matplotlib. Country outlines are read from the local
data/weather/ne_110m_admin_0_countries.zip file, which can be downloaded from
https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/?
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from shapely import intersects_xy
from shapely.geometry import box
from shapely.plotting import patch_from_polygon

from weather.load_weather import WORLD_PATH, rm_results_path

PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "weather"
US_BOUNDS = (-125, 24, -66, 50)  # west, south, east, north; excludes Alaska/Hawaii


def load_us_map():
    world = gpd.read_file(WORLD_PATH).to_crs("EPSG:4326")
    us = world.loc[world["ADMIN"] == "United States of America"].copy()
    us.geometry = us.geometry.intersection(box(*US_BOUNDS))
    return us


def plot_function(data, us, values, name, unit, filename):
    longitude, latitude = data["longitude"], data["latitude"]
    mean, variance = values
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    inside = intersects_xy(us.geometry.iloc[0], lon_grid, lat_grid)
    # This mask selects color limits only. Map boundaries use vector clipping below.
    lower = mean[inside].min()
    upper = mean[inside].max()
    variance_upper = max(float(variance[inside].max()), np.finfo(float).eps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), layout="compressed")
    maps = []
    titles = ("Posterior mean", "Posterior variance")
    for i, (ax, values, title) in enumerate(zip(axes, (mean, variance), titles)):
        maps.append(ax.pcolormesh(
            longitude, latitude, values, shading="gouraud",
            cmap="coolwarm" if i == 0 else "viridis",
            vmin=lower if i == 0 else 0, vmax=upper if i == 0 else variance_upper,
            zorder=1,
        ))
        # Clip partial grid cells at the actual border, not at their centres.
        country_clip = patch_from_polygon(us.geometry.iloc[0], transform=ax.transData)
        maps[-1].set_clip_path(country_clip)
        # Draw the vector outline on top; its shape is independent of grid resolution.
        us.boundary.plot(ax=ax, color="0.1", linewidth=1.0, zorder=2)
        ax.set(
            title=f"{title} of {name}", xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
            xlim=(longitude[0], longitude[-1]), ylim=(latitude[0], latitude[-1]),
        )
        if i == 1:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(maps[0], ax=axes[0], label=unit)
    fig.colorbar(maps[1], ax=axes[1], label=f"{unit}²")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved maps to {filename}", flush=True)
    return fig, axes


def plot_weather(path=None):
    path = rm_results_path() if path is None else Path(path)
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}

    us = load_us_map()
    outputs = data["output_names"].tolist()
    figures = {}
    j = outputs.index("TMP")
    figures["TMP"] = plot_function(
        data, us, [data[key][..., j] for key in ("mu_mean", "mu_var")],
        r"TMP $\mu(x)$", "°C", PLOTS_DIR / f"{path.stem}_mean_tmp.pdf",
    )
    dew = outputs.index("DEW")
    others = [outputs.index(output) for output in ("TMP", "SLP")]
    filename = PLOTS_DIR / f"{path.stem}_correlation_dew_means.pdf"
    figures["DEW correlations"] = plot_maps(
        data, us, [data["r_mean"][..., i, dew] for i in others],
        [f"Posterior mean of {outputs[i]}-DEW $r(x)$" for i in others],
        filename,
    )
    plt.show()
    return figures


def plot_maps(data, us, values, titles, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), layout="compressed")
    longitude, latitude = data["longitude"], data["latitude"]
    for i, (ax, value, title) in enumerate(zip(axes, values, titles)):
        mesh = ax.pcolormesh(
            longitude, latitude, value, shading="gouraud", zorder=1,
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        mesh.set_clip_path(patch_from_polygon(us.geometry.iloc[0], transform=ax.transData))
        us.boundary.plot(ax=ax, color="0.1", linewidth=1, zorder=2)
        ax.set(title=title, xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
               xlim=(longitude[0], longitude[-1]), ylim=(latitude[0], latitude[-1]))
        if i == 1:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=list(axes))
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved {filename}", flush=True)
    return fig, axes


if __name__ == "__main__":
    figures = plot_weather()
