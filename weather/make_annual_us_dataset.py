"""Annual TMP/DEW/SLP means for the same 1033 contiguous-US stations, 2014-2023.

The station list uses NOAA country US and lower-48 state codes, plus
the Pensacola station whose state code is missing. 
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "weather"
DATA_DIR = ROOT / "WEATHER-5K" / "global_weather_stations"
STATIONS_PATH = ROOT / "weather_source" / "us_stations.csv"
CONTIGUOUS_STATES = set(
    "AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT "
    "NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)
PENSACOLA_ID = "72222099999"  # NOAA name: PENSACOLA FL; missing STATE.
START_YEAR = 2014
END_YEAR = 2023
COORDINATES = ["LONGITUDE", "LATITUDE"]
OUTPUTS = ["TMP", "DEW", "SLP"]
COLUMNS = ["STATION"] + COORDINATES + ["YEAR"] + OUTPUTS
START_TIME = pd.Timestamp(f"{START_YEAR}-01-01")
END_TIME = pd.Timestamp(f"{END_YEAR}-12-31 23:00")


def select_us_stations(stations):
    """Select by NOAA CTRY/STATE and station ID, without a coordinate filter."""
    keep = stations["CTRY"].eq("US") & (
        stations["STATE"].isin(CONTIGUOUS_STATES)
        | stations["STATION"].eq(PENSACOLA_ID)
    )
    return stations.loc[keep].copy()


def annual_station(path):
    """Read one station once and return its ten annual-average rows."""
    frame = pd.read_csv(path, usecols=["DATE"] + COORDINATES + OUTPUTS,
                        parse_dates=["DATE"])
    frame = frame.loc[frame["DATE"].between(START_TIME, END_TIME)]
    grouped = frame.groupby(frame["DATE"].dt.year)
    result = grouped[COORDINATES].first().join(grouped[OUTPUTS].mean())
    result.index.name = "YEAR"
    result = result.reset_index()
    result["STATION"] = path.stem
    return result[COLUMNS]


def prepare_us_annual(output_csv=None):
    """Return 10330 station-year rows (2014-2023); optionally save a new CSV."""
    destination = ROOT / Path(output_csv) if output_csv is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError(destination)
    stations = pd.read_csv(STATIONS_PATH, dtype={"STATION": str})
    stations = select_us_stations(stations)
    rows = []
    for i, station in enumerate(stations["STATION"], 1):
        rows.append(annual_station(DATA_DIR / f"{station}.csv"))
        if i % 100 == 0 or i == len(stations):
            print(f"Averaged {i}/{len(stations)} stations", flush=True)
    result = pd.concat(rows, ignore_index=True)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False, mode="x")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default=f"weather_us_annual_{START_YEAR}_{END_YEAR}.csv")
    args = parser.parse_args()
    result = prepare_us_annual(args.output)
    print(f"Saved {len(result)} station-year rows to {ROOT / Path(args.output)}")
