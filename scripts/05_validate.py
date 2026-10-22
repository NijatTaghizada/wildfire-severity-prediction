"""Sanity checks on the labelled events.

1. Well-known fires: were they matched, and is the matched event's first detection sensible?
2. Big FPA fires that got no match: why?
3. Largest events: do they chain unrelated fires together?
"""
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from common import BIG_FIRE_ACRES, INTERIM, PROCESSED

pd.set_option("display.width", 200)
events = pd.read_csv(INTERIM / "events.csv", parse_dates=["first_detect_utc", "post_last_detect_utc"])
lab = pd.read_csv(PROCESSED / "events_labeled.csv", parse_dates=["first_detect_utc", "discovery_date"])
fpa = pd.read_csv(INTERIM / "fpa_fod_ca.csv", parse_dates=["discovery_date"])

print("== 1. Known fires ==")
known = ["CAMP", "DIXIE", "DOE", "THOMAS", "CREEK", "CALDOR", "RANCH", "TUBBS", "WOOLSEY", "PARK", "RIM", "KING"]
cols = ["fire_name", "complex_name", "discovery_date", "fire_size", "first_detect_utc", "detect_lag_days",
        "match_km", "fp_n_pixels", "fp_frp_sum", "post_n_detections"]
k = lab[lab["fire_name"].isin(known) & (lab["fire_size"] >= 50000)]
print(k[cols].to_string(index=False))
missing = sorted(set(fpa.loc[fpa.fire_name.isin(known) & (fpa.fire_size >= 50000) & (fpa.fire_year >= 2012),
                             "fire_name"]) - set(k.fire_name))
print("Known 50k+ acre fires not matched:", missing or "none")

print("\n== 2. Unmatched big fires ==")
big = fpa[(fpa.fire_size >= BIG_FIRE_ACRES) & (fpa.fire_year >= 2012) & ~fpa.fod_id.isin(lab.fod_id)].copy()
det = pd.read_csv(INTERIM / "firms_event_ids.csv", parse_dates=["acq_datetime"])
to_km = Transformer.from_crs(4326, 3310, always_xy=True)
dx, dy = to_km.transform(det.longitude.values, det.latitude.values)
bx, by = to_km.transform(big.longitude.values, big.latitude.values)
tree = cKDTree(np.column_stack([dx, dy]) / 1000)
t0 = events.set_index("event_id")["first_detect_utc"]
local_day = (t0.dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None).dt.normalize())
taken = set(lab["event_id"])
reasons = []
for i, near in enumerate(tree.query_ball_point(np.column_stack([bx, by]) / 1000, 5.0)):
    if not near:
        reasons.append("no detection within 5 km ever")
        continue
    # Only events that were already burning near the ignition point within 3 days of discovery
    close = det.iloc[near]
    close = close[(close.acq_datetime - big.discovery_date.values[i]).abs() <= pd.Timedelta(days=3)]
    if close.empty:
        reasons.append("no detection within 5 km and 3 days of discovery")
        continue
    eids = close.event_id.unique()
    lags = ((local_day.loc[eids].values - big.discovery_date.values[i]) / np.timedelta64(1, "D")).astype(int)
    if all(l < -1 for l in lags):
        reasons.append("detections belong to an event that started >1 day before discovery")
    elif all(e in taken for e, l in zip(eids, lags) if -1 <= l <= 2):
        reasons.append("event already matched to another FPA fire")
    else:
        reasons.append("other")
big["reason"] = reasons
print(big["reason"].value_counts().to_string())

neg_big = lab[(lab.label_big_fire == 0) & (lab.post_n_detections > 500)]
print(f"\nLabel-0 events with >500 detections (possible mislabels): {len(neg_big)}")
if len(neg_big):
    print(neg_big[["fire_name", "fire_size", "discovery_date", "post_n_detections"]].head(15).to_string(index=False))

print("\n== 3. Largest events ==")
top = events.nlargest(8, "post_n_detections")
for _, e in top.iterrows():
    names = lab.loc[lab.event_id == e.event_id, ["fire_name", "fire_size"]].values.tolist()
    print(f"event {e.event_id}: {e.post_n_detections:,} det, {e.post_duration_days:.0f} days, "
          f"start {e.first_detect_utc:%Y-%m-%d} at ({e.lat:.2f},{e.lon:.2f}) -> {names}")
