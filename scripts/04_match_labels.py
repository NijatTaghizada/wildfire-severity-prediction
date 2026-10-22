"""Match fire events to FPA FOD records and attach the big-fire label.

A candidate match needs:
  - FPA ignition point within MATCH_KM of any detection from the event's first MATCH_HOURS hours
    (not the first-pass centre: fast fires like Camp 2018 have already spread >5 km by then)
  - first detection (Pacific local date) between LAG_MIN and LAG_MAX days after FPA discovery date
  - the event is not far bigger than the FPA fire could produce (lightning sieges put tiny fires
    right next to big ones; a 0.1-acre fire cannot account for 129,000 detections)
Candidates are assigned one-to-one, best (closest in space and time) first.
The size check reads whole-event detection counts, so it shapes which record labels an event,
but those counts are never model inputs.
Events with no FPA match are dropped: they are mostly agricultural or prescribed burns,
industrial heat sources, or later fragments of an already matched fire.

Output: data/processed/events_labeled.csv
"""
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from common import BIG_FIRE_ACRES, INTERIM, PROCESSED

MATCH_KM = 5.0
MATCH_HOURS = 24
LAG_MIN, LAG_MAX = -1, 2  # detection may precede discovery by a day (remote fires) or trail it by up to 2
DAY_PENALTY_KM = 2.0      # 1 day of date difference counts like 2 km of distance when ranking
ACRES_PER_PIXEL = 35      # one 375 m VIIRS pixel
MAX_DET_PER_PIXEL = 50    # big fires here show 1-5 detections per pixel of final area; 50 is very generous

events = pd.read_csv(INTERIM / "events.csv", parse_dates=["first_detect_utc", "post_last_detect_utc"])
fpa = pd.read_csv(INTERIM / "fpa_fod_ca.csv", parse_dates=["discovery_date", "cont_date"])

events["first_detect_local"] = (events["first_detect_utc"].dt.tz_localize("UTC")
                                .dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
ev_day = events["first_detect_local"].dt.normalize()

# Detections from each event's first MATCH_HOURS hours.
det = pd.read_csv(INTERIM / "firms_event_ids.csv", parse_dates=["acq_datetime"])
ei_of = pd.Series(events.index, index=events["event_id"])
det["ei"] = ei_of.loc[det["event_id"]].values
det = det[det["acq_datetime"] <= events["first_detect_utc"].values[det["ei"]] + np.timedelta64(MATCH_HOURS, "h")]

to_km = Transformer.from_crs(4326, 3310, always_xy=True)
dx, dy = to_km.transform(det["longitude"].values, det["latitude"].values)
fx, fy = to_km.transform(fpa["longitude"].values, fpa["latitude"].values)
det_tree = cKDTree(np.column_stack([dx, dy]) / 1000.0)
det_ei = det["ei"].values

cand = []
for fi, near in enumerate(det_tree.query_ball_point(np.column_stack([fx, fy]) / 1000.0, MATCH_KM)):
    if not near:
        continue
    near = np.array(near)
    dist = np.hypot(dx[near] - fx[fi], dy[near] - fy[fi]) / 1000.0
    best = pd.Series(dist).groupby(det_ei[near]).min()  # nearest detection per candidate event
    lag = ((ev_day.values[best.index] - fpa["discovery_date"].values[fi]) / np.timedelta64(1, "D")).astype(int)
    for ei, d, l in zip(best.index, best.values, lag):
        if LAG_MIN <= l <= LAG_MAX:
            cand.append((d + DAY_PENALTY_KM * abs(l), ei, fi, d, l))

cand = pd.DataFrame(cand, columns=["cost", "ei", "fi", "match_km", "detect_lag_days"]).sort_values("cost")
max_det = MAX_DET_PER_PIXEL * np.maximum(1, fpa["fire_size"].values[cand["fi"]] / ACRES_PER_PIXEL)
plausible = events["post_n_detections"].values[cand["ei"]] <= max_det
print(f"Dropped {(~plausible).sum():,} of {len(cand):,} candidate pairs where the event is far too big for the fire")
cand = cand[plausible]
used_e, used_f, keep = set(), set(), []
for row in cand.itertuples(index=False):
    if row.ei in used_e or row.fi in used_f:
        continue
    used_e.add(row.ei)
    used_f.add(row.fi)
    keep.append(row)
m = pd.DataFrame(keep)

fpa_cols = ["fod_id", "fire_name", "complex_name", "mtbs_id", "discovery_date", "discovery_time", "fire_size",
            "fire_size_class", "nwcg_cause_classification", "nwcg_general_cause"]
out = events.iloc[m["ei"]].reset_index(drop=True)
out = pd.concat([out, m[["match_km", "detect_lag_days"]].reset_index(drop=True),
                 fpa.iloc[m["fi"]][fpa_cols].reset_index(drop=True)], axis=1)
out["label_big_fire"] = (out["fire_size"] >= BIG_FIRE_ACRES).astype(int)
out = out.sort_values("first_detect_utc").reset_index(drop=True)
out.to_csv(PROCESSED / "events_labeled.csv", index=False)

big_total = (fpa["fire_size"] >= BIG_FIRE_ACRES) & (fpa["discovery_date"].dt.year >= 2012)
print(f"Events: {len(events):,}  matched: {len(out):,}  positives: {out.label_big_fire.sum():,} "
      f"({out.label_big_fire.mean():.1%})")
print(f"Share of FPA 1,000+ acre fires (2012-2024) that got a match: "
      f"{out.label_big_fire.sum() / big_total.sum():.0%} of {big_total.sum()}")
print("Detection lag (days after discovery):", out["detect_lag_days"].value_counts().sort_index().to_dict())
print("Size class:", out["fire_size_class"].value_counts().sort_index().to_dict())
print(out.groupby(out["first_detect_utc"].dt.year)["label_big_fire"].agg(["size", "sum"]).T.to_string())
