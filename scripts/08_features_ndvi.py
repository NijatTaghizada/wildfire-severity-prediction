"""Vegetation greenness (MODIS MOD13Q1 NDVI, 250 m, 16-day) before each fire.

Uses the ORNL DAAC MODIS web service (no account). A composite labelled with date A covers
A .. A+15 days, so only composites that ended before the first detection are used; otherwise
the value could already include the burn scar.

Output: data/interim/features_ndvi.csv
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from common import INTERIM, PROCESSED

API = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
BANDS = {"ndvi": "250m_16_days_NDVI"}  # EVI tracks NDVI closely and would double the (slow) requests
SCALE, FILL = 0.0001, -3000
WORKERS = 6
CACHE = INTERIM / "ndvi_cache.jsonl"


def modis_date(ts):
    return f"A{ts.year}{ts.dayofyear:03d}"


def fetch(row):
    end = row.detect_day - pd.Timedelta(days=16)   # composite must have started 16+ days earlier
    start = end - pd.Timedelta(days=48)             # a few composites, in case the latest is cloudy
    out = {"event_id": int(row.event_id)}
    for name, band in BANDS.items():
        params = {"latitude": f"{row.lat:.5f}", "longitude": f"{row.lon:.5f}", "band": band,
                  "startDate": modis_date(start), "endDate": modis_date(end),
                  "kmAboveBelow": 0, "kmLeftRight": 0}
        for attempt in range(5):
            try:
                r = requests.get(API, params=params, headers={"Accept": "application/json"}, timeout=60)
                r.raise_for_status()
                out[name] = [(s["calendar_date"], s["data"][0]) for s in r.json()["subset"]]
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return None
    return out


ev = pd.read_csv(PROCESSED / "events_labeled.csv", parse_dates=["first_detect_local"])
ev["detect_day"] = ev["first_detect_local"].dt.normalize()
done = {json.loads(line)["event_id"] for line in CACHE.open()} if CACHE.exists() else set()
todo = [r for r in ev.itertuples() if r.event_id not in done]
with CACHE.open("a") as f, ThreadPoolExecutor(WORKERS) as pool:
    for fut in tqdm(as_completed([pool.submit(fetch, r) for r in todo]), total=len(todo), desc="MODIS NDVI"):
        res = fut.result()
        if res:
            f.write(json.dumps(res) + "\n")
            f.flush()

detect_day = ev.set_index("event_id")["detect_day"]
rows = []
for line in CACHE.open():
    d = json.loads(line)
    row = {"event_id": d["event_id"]}
    for name in BANDS:
        # latest valid composite that ended before the detection day
        valid = [(pd.Timestamp(day), v) for day, v in d.get(name, []) if v != FILL
                 and pd.Timestamp(day) + pd.Timedelta(days=15) < detect_day[d["event_id"]]]
        valid.sort()
        row[name] = valid[-1][1] * SCALE if valid else np.nan
        row[f"{name}_age_days"] = (detect_day[d["event_id"]] - valid[-1][0]).days if valid else np.nan
    rows.append(row)
feats = pd.DataFrame(rows)
feats.to_csv(INTERIM / "features_ndvi.csv", index=False)
print(f"Saved NDVI for {feats['ndvi'].notna().sum():,} / {len(ev):,} fires")
print(feats.describe().T[["mean", "min", "max"]].round(3).to_string())
