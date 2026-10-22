"""Weather and fuel-dryness features from gridMET (4 km, daily) at each fire's first detection.

Point queries go to the University of Idaho THREDDS NetCDF Subset Service, so no account is needed.
Only days BEFORE the first detection (Pacific local date) are used: the previous day's values
plus short look-back summaries. Responses are cached in data/interim/gridmet_cache/ so the
script can be stopped and resumed.

Output: data/interim/features_gridmet.csv
"""
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from common import INTERIM, PROCESSED

NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{v}_1979_CurrentYear_CONUS.nc"
VARS = {  # dataset id: (variable name in file, days of look-back)
    "tmmx": ("daily_maximum_temperature", 7),
    "rmin": ("daily_minimum_relative_humidity", 7),
    "vs": ("daily_mean_wind_speed", 7),
    "th": ("daily_mean_wind_direction", 1),
    "vpd": ("daily_mean_vapor_pressure_deficit", 7),
    "fm100": ("dead_fuel_moisture_100hr", 1),
    "fm1000": ("dead_fuel_moisture_1000hr", 1),
    "erc": ("daily_mean_energy_release_component-g", 7),
    "bi": ("daily_mean_burning_index_g", 1),
    "pr": ("precipitation_amount", 60),
}
# NCSS returns the stored (packed) integers; real value = raw * scale + offset (from the files' metadata)
PACKING = {"tmmx": (0.1, 220.0), "rmin": (0.1, 0.0), "vs": (0.1, 0.0), "th": (1.0, 0.0), "vpd": (0.01, 0.0),
           "fm100": (0.1, 0.0), "fm1000": (0.1, 0.0), "erc": (1.0, 0.0), "bi": (1.0, 0.0), "pr": (0.1, 0.0)}
WORKERS = 8
CACHE = INTERIM / "gridmet_cache"
CACHE.mkdir(exist_ok=True)

ev = pd.read_csv(PROCESSED / "events_labeled.csv", parse_dates=["first_detect_local"])
ev["prev_day"] = ev["first_detect_local"].dt.normalize() - pd.Timedelta(days=1)

session = requests.Session()


def fetch(v, row):
    name, back = VARS[v]
    start = row.prev_day - pd.Timedelta(days=back - 1)
    params = {"var": name, "latitude": f"{row.lat:.4f}", "longitude": f"{row.lon:.4f}",
              "time_start": f"{start:%Y-%m-%d}T00:00:00Z", "time_end": f"{row.prev_day:%Y-%m-%d}T00:00:00Z",
              "accept": "csv"}
    for attempt in range(5):
        try:
            r = session.get(NCSS.format(v=v), params=params, timeout=60)
            r.raise_for_status()
            vals = pd.read_csv(io.StringIO(r.text)).iloc[:, -1].tolist()
            return row.event_id, vals
        except Exception:
            time.sleep(2 ** attempt)
    return row.event_id, None


for v in VARS:
    cache_file = CACHE / f"{v}.jsonl"
    done = set()
    if cache_file.exists():
        done = {json.loads(line)["event_id"] for line in cache_file.open()}
    todo = [r for r in ev.itertuples() if r.event_id not in done]
    if not todo:
        continue
    with cache_file.open("a") as out, ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(fetch, v, r) for r in todo]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"gridMET {v}"):
            eid, vals = f.result()
            if vals is not None:
                out.write(json.dumps({"event_id": int(eid), "vals": vals}) + "\n")
                out.flush()

# --- Summarise into features ------------------------------------------------------
series = {v: {d["event_id"]: [x * PACKING[v][0] + PACKING[v][1] for x in d["vals"]]
              for d in map(json.loads, (CACHE / f"{v}.jsonl").open())} for v in VARS}


def last(v, eid):
    vals = series[v].get(eid)
    return vals[-1] if vals else np.nan


def stat(v, eid, fn, n=None):
    vals = series[v].get(eid)
    return fn(vals[-n:] if n else vals) if vals else np.nan


def days_since_rain(eid, mm=2.5):
    vals = series["pr"].get(eid)
    if not vals:
        return np.nan
    wet = [i for i, p in enumerate(reversed(vals)) if p >= mm]
    return wet[0] + 1 if wet else len(vals) + 1  # 61 means "no rain in the last 60 days"


feats = pd.DataFrame({"event_id": ev["event_id"]})
e = feats["event_id"]
feats["tmax_c_prev"] = [last("tmmx", i) - 273.15 for i in e]
feats["rhmin_prev"] = [last("rmin", i) for i in e]
feats["wind_ms_prev"] = [last("vs", i) for i in e]
feats["wind_dir_prev"] = [last("th", i) for i in e]
feats["vpd_prev"] = [last("vpd", i) for i in e]
feats["fm100_prev"] = [last("fm100", i) for i in e]
feats["fm1000_prev"] = [last("fm1000", i) for i in e]
feats["erc_prev"] = [last("erc", i) for i in e]
feats["bi_prev"] = [last("bi", i) for i in e]
feats["erc_7d_mean"] = [stat("erc", i, np.mean) for i in e]
feats["vpd_7d_mean"] = [stat("vpd", i, np.mean) for i in e]
feats["tmax_c_7d_mean"] = [stat("tmmx", i, np.mean) - 273.15 for i in e]
feats["wind_ms_7d_max"] = [stat("vs", i, np.max) for i in e]
feats["precip_mm_30d"] = [stat("pr", i, np.sum, 30) for i in e]
feats["days_since_rain"] = [days_since_rain(i) for i in e]
feats.to_csv(INTERIM / "features_gridmet.csv", index=False)
print(f"Saved gridMET features for {feats.dropna().shape[0]:,} / {len(feats):,} fires")
print(feats.describe().T[["mean", "min", "max"]].round(2).to_string())
