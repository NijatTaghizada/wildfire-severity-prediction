"""Download NASA FIRMS VIIRS yearly US files and keep California vegetation-fire detections.

Uses the public yearly country files (no MAP_KEY needed):
  https://firms.modaps.eosdis.nasa.gov/data/country/<sensor>/<year>/<sensor>_<year>_United_States.csv
VIIRS S-NPP covers 2012-, NOAA-20 (named 'viirs-jpss1' on the server) covers 2018-.

Output: data/interim/firms_viirs_ca.csv
"""
import pandas as pd
from tqdm import tqdm

from common import FIRST_YEAR, INTERIM, LAST_YEAR, in_california

BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
SENSORS = {"viirs-snpp": 2012, "viirs-jpss1": 2018}

frames = []
jobs = [(s, y) for s, start in SENSORS.items() for y in range(max(start, FIRST_YEAR), LAST_YEAR + 1)]
for sensor, year in tqdm(jobs, desc="FIRMS yearly files"):
    url = f"{BASE}/{sensor}/{year}/{sensor}_{year}_United_States.csv"
    df = pd.read_csv(url, dtype={"acq_time": str, "version": str})
    df = df[(df["type"] == 0) & in_california(df["longitude"], df["latitude"])]
    tqdm.write(f"{sensor} {year}: {len(df):,} California vegetation-fire detections")
    frames.append(df)

firms = pd.concat(frames, ignore_index=True)
firms["acq_time"] = firms["acq_time"].str.zfill(4)
firms["acq_datetime"] = pd.to_datetime(firms["acq_date"] + " " + firms["acq_time"], format="%Y-%m-%d %H%M")
firms = firms.sort_values("acq_datetime").reset_index(drop=True)

out = INTERIM / "firms_viirs_ca.csv"
firms.to_csv(out, index=False)
print(f"\nSaved {len(firms):,} detections to {out}")
print(firms.groupby([firms["acq_datetime"].dt.year, "satellite"]).size().unstack(fill_value=0))
