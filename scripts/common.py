"""Shared paths and helpers for the data pipeline."""
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
for d in (RAW, INTERIM, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

# Study period. FPA FOD 7th edition (the labels) ends in 2024.
FIRST_YEAR, LAST_YEAR = 2012, 2024

BIG_FIRE_ACRES = 1000  # NWCG size class F and above; also the MTBS cutoff in the West

CA_BOUNDARY = RAW / "california_boundary.geojson"  # Census TIGERweb, simplified to ~100 m


def california():
    with open(CA_BOUNDARY) as f:
        geom = shape(json.load(f)["features"][0]["geometry"])
    shapely.prepare(geom)
    return geom


def in_california(lon, lat):
    """Vectorised point-in-California test."""
    return shapely.contains_xy(california(), np.asarray(lon), np.asarray(lat))
