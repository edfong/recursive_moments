"""Compare posterior mean TMP maps for exact GPAR and ICM with 256 inducing points."""

import matplotlib.pyplot as plt
import numpy as np
from shapely import intersects_xy
from shapely.plotting import patch_from_polygon

from weather.load_weather import gpar_results_path, icm_results_path
from weather.plot_weather import PLOTS_DIR, load_us_map


def plot_weather_GPs():
    us = load_us_map()
    maps = []
    for title, path in (("GPAR (exact)", gpar_results_path(None)),
                        ("ICM (m = 256)", icm_results_path(256))):
        with np.load(path, allow_pickle=False) as saved:
            j = saved["output_names"].tolist().index("TMP")
            longitude, latitude = saved["longitude"], saved["latitude"]
            mean = saved["mu_mean"][..., j]
        inside = intersects_xy(us.geometry.iloc[0], *np.meshgrid(longitude, latitude))
        maps.append((title, longitude, latitude, mean, inside))

    # Use the same colour scale for both models, based on locations inside the US.
    lower = min(mean[inside].min() for _, _, _, mean, inside in maps)
    upper = max(mean[inside].max() for _, _, _, mean, inside in maps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), layout="compressed")
    for ax, (title, longitude, latitude, mean, _) in zip(axes, maps):
        mesh = ax.pcolormesh(longitude, latitude, mean, shading="gouraud",
                             cmap="coolwarm", vmin=lower, vmax=upper, zorder=1)
        mesh.set_clip_path(patch_from_polygon(us.geometry.iloc[0], transform=ax.transData))
        us.boundary.plot(ax=ax, color="0.1", linewidth=1, zorder=2)
        ax.set(title=title, xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
               xlim=(longitude[0], longitude[-1]), ylim=(latitude[0], latitude[-1]))
        ax.set_aspect("equal", adjustable="box")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.colorbar(mesh, ax=list(axes), label="Posterior mean TMP (°C)")
    path = PLOTS_DIR / "weather_GPs_mean_tmp.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}", flush=True)
    plt.show()
    return fig, axes


if __name__ == "__main__":
    fig, axes = plot_weather_GPs()
