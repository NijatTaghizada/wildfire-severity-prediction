"""Merge labels and all feature sources into the final training table.

Model inputs are only things known at the first satellite detection. Everything describing what
happened afterwards (post_* columns, final size, cause, containment) stays as metadata.

Output: data/processed/training_table.csv
"""
import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED

SPLITS = {"train": (2012, 2020), "val": (2021, 2022), "test": (2023, 2024)}

lab = pd.read_csv(PROCESSED / "events_labeled.csv", parse_dates=["first_detect_utc", "first_detect_local"])
df = lab
for name in ("gridmet", "landfire", "ndvi"):
    df = df.merge(pd.read_csv(INTERIM / f"features_{name}.csv"), on="event_id", how="left", validate="1:1")

t = df["first_detect_local"]
df["year"] = t.dt.year
df["month"] = t.dt.month
df["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
df["local_hour"] = t.dt.hour + t.dt.minute / 60
df["is_night"] = (df["daynight"] == "N").astype(int)
df["is_noaa20"] = (df["satellite"] == "N20").astype(int)
df["split"] = "none"
for split, (lo, hi) in SPLITS.items():
    df.loc[df["year"].between(lo, hi), "split"] = split
df["hard_case"] = (df["fp_n_pixels"] <= 3).astype(int)  # still small when first seen

FEATURES = [
    # satellite, first pass only
    "fp_n_pixels", "fp_frp_sum", "fp_frp_max", "fp_frp_mean", "fp_bright_ti4_max", "fp_bright_ti4_mean",
    "fp_bright_ti5_mean", "fp_scan_mean", "fp_track_mean", "fp_share_high_conf", "is_night", "is_noaa20",
    # weather and fuel dryness (gridMET, days before detection)
    "tmax_c_prev", "rhmin_prev", "wind_ms_prev", "wind_dir_prev", "vpd_prev", "fm100_prev", "fm1000_prev",
    "erc_prev", "bi_prev", "erc_7d_mean", "vpd_7d_mean", "tmax_c_7d_mean", "wind_ms_7d_max",
    "precip_mm_30d", "days_since_rain",
    # vegetation, fuel, terrain
    "ndvi", "fbfm40", "fuel_family", "elevation_m", "slope_deg", "aspect_northness", "aspect_eastness",
    # place and time
    "lat", "lon", "month", "doy_sin", "doy_cos", "local_hour",
]
META = ["event_id", "fod_id", "fire_name", "complex_name", "first_detect_local", "year", "split", "hard_case",
        "detect_lag_days", "match_km", "fire_size", "fire_size_class", "nwcg_general_cause", "post_n_detections"]
LABEL = "label_big_fire"

out = df[META + FEATURES + [LABEL]]
out.to_csv(PROCESSED / "training_table.csv", index=False)

print(f"Saved {len(out):,} rows x {len(FEATURES)} features to {PROCESSED / 'training_table.csv'}")
print("\nRows and big fires per split (all / hard cases):")
s = out.groupby("split").agg(rows=(LABEL, "size"), big=(LABEL, "sum"),
                             hard_rows=("hard_case", "sum"),
                             hard_big=(LABEL, lambda y: int(y[out.loc[y.index, "hard_case"] == 1].sum())))
print(s.loc[list(SPLITS)].to_string())
missing = out[FEATURES].isna().mean().loc[lambda m: m > 0].round(3)
print("\nMissing share per feature:", missing.to_dict() or "none")
