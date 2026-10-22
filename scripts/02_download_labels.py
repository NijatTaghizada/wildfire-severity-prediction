"""Download the label datasets and extract the California part.

- FPA FOD 7th edition (Short 2026, doi:10.2737/RDS-2013-0009.7): every reported US wildfire 1992-2024
  with discovery date, ignition point and final size.
- CAL FIRE FRAP fire perimeters: final outlines of California fires, used to sanity-check labels.

Outputs:
  data/interim/fpa_fod_ca.csv
  data/interim/calfire_perimeters_ca.geojson  (2012-2024 only, slimmed)
"""
import json
import sqlite3
import subprocess
import zipfile

import pandas as pd

from common import FIRST_YEAR, INTERIM, LAST_YEAR, RAW

FPA_URL = "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.7/RDS-2013-0009.7_Data_Format4_SQLITE.zip"
CALFIRE_URL = ("https://gis.data.cnra.ca.gov/api/download/v1/items/"
               "c3c10388e3b24cec8a954ba10458039d/geojson?layers=0")


def fetch(url, dest):
    if dest.exists():
        print(f"Already have {dest.name}")
        return
    print(f"Downloading {dest.name} ...")
    subprocess.run(["curl", "-sSL", "-o", str(dest), url], check=True)


# --- FPA FOD ---------------------------------------------------------------
fpa_zip = RAW / "fpa_fod_7th_sqlite.zip"
fetch(FPA_URL, fpa_zip)
fpa_db = RAW / "fpa_fod" / "Data" / "FPA_FOD_20260615.sqlite"
if not fpa_db.exists():
    zipfile.ZipFile(fpa_zip).extractall(RAW / "fpa_fod")

cols = ["FOD_ID", "FIRE_NAME", "COMPLEX_NAME", "MTBS_ID", "FIRE_YEAR", "DISCOVERY_DATE", "DISCOVERY_TIME",
        "CONT_DATE", "FIRE_SIZE", "FIRE_SIZE_CLASS", "LATITUDE", "LONGITUDE", "NWCG_CAUSE_CLASSIFICATION",
        "NWCG_GENERAL_CAUSE", "NWCG_REPORTING_AGENCY", "OWNER_DESCR"]
with sqlite3.connect(fpa_db) as con:
    # Include the year before so fires discovered in late Dec 2011 can match early Jan 2012 detections.
    fpa = pd.read_sql(f"select {', '.join(cols)} from Fires where STATE = 'CA' "
                      f"and FIRE_YEAR between {FIRST_YEAR - 1} and {LAST_YEAR}", con)

# Dates are stored as Julian day numbers.
for c in ("DISCOVERY_DATE", "CONT_DATE"):
    fpa[c] = pd.to_datetime(fpa[c] - 2440587.5, unit="D").dt.normalize()
fpa.columns = fpa.columns.str.lower()
fpa.to_csv(INTERIM / "fpa_fod_ca.csv", index=False)
print(f"FPA FOD California {FIRST_YEAR - 1}-{LAST_YEAR}: {len(fpa):,} fires, "
      f"{(fpa.fire_size >= 1000).sum():,} of 1,000+ acres")

# --- CAL FIRE perimeters ----------------------------------------------------
cf_raw = RAW / "calfire_perimeters.geojson"
fetch(CALFIRE_URL, cf_raw)
with open(cf_raw) as f:
    gj = json.load(f)
keep = ("YEAR_", "FIRE_NAME", "INC_NUM", "ALARM_DATE", "CONT_DATE", "GIS_ACRES", "CAUSE", "COMPLEX_NAME")
feats = []
for ft in gj["features"]:
    p = ft["properties"]
    if ft["geometry"] and p.get("YEAR_") and FIRST_YEAR <= int(p["YEAR_"]) <= LAST_YEAR:
        feats.append({"type": "Feature", "geometry": ft["geometry"], "properties": {k: p.get(k) for k in keep}})
with open(INTERIM / "calfire_perimeters_ca.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": feats}, f)
print(f"CAL FIRE perimeters {FIRST_YEAR}-{LAST_YEAR}: {len(feats):,} fires")
