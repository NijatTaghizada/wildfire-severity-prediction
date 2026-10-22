"""Terrain and fuel-type features from LANDFIRE (30 m) at each fire's first-pass centre.

Uses the public LANDFIRE ImageServer getSamples endpoint (batches of points, no account):
  - LF2020 elevation, slope (degrees), aspect
  - LF2016 Remap FBFM40 fuel model (Scott & Burgan 40)

Note on fuel timing: LF2016 describes roughly 2016 vegetation. For fires in 2012-2016 it can already
show their own burn scar, so in those years fuel type partly reflects the outcome. The planned test
years (2023-2024) are unaffected, and the train-years effect can only hurt, not inflate, test scores.

Output: data/interim/features_landfire.csv
"""
import json
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from common import INTERIM, PROCESSED

BASE = "https://lfps.usgs.gov/arcgis/rest/services"
LAYERS = {
    "elevation_m": "Landfire_Topo/LF2020_Elev_CONUS",
    "slope_deg": "Landfire_Topo/LF2020_SlpD_CONUS",
    "aspect_deg": "Landfire_Topo/LF2020_Asp_CONUS",
    "fbfm40": "Landfire_LF2016/LF2016_FBFM40_CONUS",
}
BATCH = 200

# Scott & Burgan code ranges -> fuel family
FAMILIES = [(91, 99, "nonburnable"), (101, 109, "grass"), (121, 124, "grass_shrub"), (141, 149, "shrub"),
            (161, 165, "timber_understory"), (181, 189, "timber_litter"), (201, 204, "slash_blowdown")]


def family(code):
    for lo, hi, name in FAMILIES:
        if lo <= code <= hi:
            return name
    return "unknown"


def samples(layer, pts):
    geom = {"points": pts, "spatialReference": {"wkid": 4326}}
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE}/{layer}/ImageServer/getSamples", timeout=120, data={
                "geometryType": "esriGeometryMultipoint", "geometry": json.dumps(geom),
                "returnFirstValueOnly": "true", "f": "json"})
            r.raise_for_status()
            out = [np.nan] * len(pts)
            for s in r.json()["samples"]:
                try:
                    out[s["locationId"]] = float(s["value"])
                except (TypeError, ValueError):
                    pass
            return out
        except Exception:
            time.sleep(2 ** attempt)
    return [np.nan] * len(pts)


ev = pd.read_csv(PROCESSED / "events_labeled.csv")
pts = ev[["lon", "lat"]].round(5).values.tolist()
feats = pd.DataFrame({"event_id": ev["event_id"]})
for col, layer in LAYERS.items():
    vals = []
    for i in tqdm(range(0, len(pts), BATCH), desc=col):
        vals += samples(layer, pts[i:i + BATCH])
    feats[col] = vals

# Aspect is circular, so split it into north-south and east-west components. Flat ground has aspect -1.
a = np.radians(feats["aspect_deg"].where(feats["aspect_deg"] >= 0))
feats["aspect_northness"] = np.cos(a).fillna(0)
feats["aspect_eastness"] = np.sin(a).fillna(0)
feats["fuel_family"] = feats["fbfm40"].map(lambda c: family(int(c)) if pd.notna(c) else "unknown")
feats.to_csv(INTERIM / "features_landfire.csv", index=False)
print(f"Saved LANDFIRE features; missing values per column:\n{feats.isna().sum().to_string()}")
print(feats["fuel_family"].value_counts().to_string())
