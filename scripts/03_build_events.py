"""Group FIRMS detections into fire events and describe each event's first satellite pass.

Two detections belong to the same event when they are within LINK_KM of each other and
at most GAP_DAYS calendar days apart. Links are built two ways:
  - within the same day: every pair within LINK_KM
  - across days: each detection links to its nearest detection from the previous GAP_DAYS days
Connected components of that graph are the events.

Only the first overpass (detections within FIRST_PASS_MIN of the earliest one) is used for
model features. Columns prefixed `post_` describe the whole event and are for validation only.
They use information from after first detection and must never be model inputs.

Output: data/interim/events.csv, data/interim/firms_event_ids.csv
"""
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

from common import INTERIM

LINK_KM = 1.0
GAP_DAYS = 2
FIRST_PASS_MIN = 30

firms = pd.read_csv(INTERIM / "firms_viirs_ca.csv", parse_dates=["acq_datetime"])
n = len(firms)

# California Albers (EPSG:3310) so distances are in km.
x, y = Transformer.from_crs(4326, 3310, always_xy=True).transform(firms["longitude"].values, firms["latitude"].values)
xy = np.column_stack([x, y]) / 1000.0
day = (firms["acq_datetime"].dt.normalize() - pd.Timestamp("2012-01-01")).dt.days.values

src, dst = [], []
order = np.argsort(day, kind="stable")
day_sorted = day[order]
bounds = np.searchsorted(day_sorted, np.arange(day_sorted.min(), day_sorted.max() + 2))
first_day = day_sorted.min()
for d in tqdm(range(day_sorted.min(), day_sorted.max() + 1), desc="Linking days"):
    lo, hi = bounds[d - first_day], bounds[d - first_day + 1]
    if lo == hi:
        continue
    today = order[lo:hi]
    tree = cKDTree(xy[today])
    pairs = tree.query_pairs(LINK_KM, output_type="ndarray")
    src.append(today[pairs[:, 0]])
    dst.append(today[pairs[:, 1]])

    plo = bounds[max(d - GAP_DAYS - first_day, 0)]
    if plo < lo:
        prev = order[plo:lo]
        dist, idx = cKDTree(xy[prev]).query(xy[today], distance_upper_bound=LINK_KM)
        ok = np.isfinite(dist)
        src.append(today[ok])
        dst.append(prev[idx[ok]])

src, dst = np.concatenate(src), np.concatenate(dst)
graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
n_events, labels = connected_components(graph, directed=False)
firms["event_id"] = labels
print(f"{n:,} detections -> {n_events:,} events ({len(src):,} links)")

# --- First-pass features -------------------------------------------------------
t0 = firms.groupby("event_id")["acq_datetime"].transform("min")
first = firms[firms["acq_datetime"] <= t0 + pd.Timedelta(minutes=FIRST_PASS_MIN)]
g = first.groupby("event_id")
events = pd.DataFrame({
    "first_detect_utc": g["acq_datetime"].min(),
    "lat": g["latitude"].mean(),
    "lon": g["longitude"].mean(),
    "satellite": g["satellite"].first(),
    "daynight": g["daynight"].first(),
    "fp_n_pixels": g.size(),
    "fp_frp_sum": g["frp"].sum(),
    "fp_frp_max": g["frp"].max(),
    "fp_frp_mean": g["frp"].mean(),
    "fp_bright_ti4_max": g["bright_ti4"].max(),
    "fp_bright_ti4_mean": g["bright_ti4"].mean(),
    "fp_bright_ti5_mean": g["bright_ti5"].mean(),
    "fp_scan_mean": g["scan"].mean(),   # pixel size grows toward the swath edge, which affects FRP
    "fp_track_mean": g["track"].mean(),
    "fp_share_high_conf": g["confidence"].apply(lambda c: (c == "h").mean()),
})

# --- Whole-event summary (validation only) ---------------------------------------
ga = firms.groupby("event_id")
events["post_n_detections"] = ga.size()
events["post_last_detect_utc"] = ga["acq_datetime"].max()
events["post_duration_days"] = (events["post_last_detect_utc"] - events["first_detect_utc"]).dt.total_seconds() / 86400
events["post_frp_sum"] = ga["frp"].sum()

events = events.sort_values("first_detect_utc").reset_index()
events.to_csv(INTERIM / "events.csv", index=False)
firms[["acq_datetime", "latitude", "longitude", "frp", "event_id"]].to_csv(INTERIM / "firms_event_ids.csv", index=False)

print(f"Saved {len(events):,} events")
print("Detections per event:", events["post_n_detections"].describe(percentiles=[.5, .9, .99]).round(1).to_dict())
print("Events per year:", events.groupby(events["first_detect_utc"].dt.year).size().to_dict())
