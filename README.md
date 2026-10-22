# Wildfire Severity Prediction

**When a satellite first sees a new fire in California, can we tell whether it will grow past 1,000 acres?**

This is the follow-up to our ISEF 2025 project, [Wildfire-Predictiveness](https://github.com/NijatTaghizada/Wildfire-Predictiveness), done through [React Right](https://nijattaghizada.github.io/reactright-website/).
That study predicted Fire Radiative Power (FRP) from brightness measured on the same satellite pixel, which mostly describes a fire that is already burning.
Here each row is a whole fire and every input is known at the moment of first detection. The target is what the fire becomes.

## Data

| Role | Dataset | Coverage |
|---|---|---|
| Fire detections | NASA FIRMS VIIRS 375 m active fires, S-NPP and NOAA-20 ([yearly country files](https://firms.modaps.eosdis.nasa.gov/download/)) | California, 2012–2024 |
| Label (final fire size) | FPA FOD 7th edition, Short (2026), [doi:10.2737/RDS-2013-0009.7](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.7) | US wildfires 1992–2024 |
| Label check | CAL FIRE FRAP fire perimeters ([data.ca.gov](https://lab.data.ca.gov/dataset/california-fire-perimeters-all)) | California |
| State boundary | US Census TIGERweb | — |

Planned next: weather and fuel dryness (gridMET), vegetation (MODIS NDVI), fuel type (LANDFIRE), terrain (3DEP), sampled with Google Earth Engine at each fire's first detection.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd scripts
../.venv/bin/python 01_download_firms.py    # ~850 MB streamed, keeps California vegetation fires
../.venv/bin/python 02_download_labels.py   # FPA FOD (~230 MB) and CAL FIRE perimeters (~250 MB)
../.venv/bin/python 03_build_events.py      # groups detections into fire events
../.venv/bin/python 04_match_labels.py      # links events to FPA FOD records, adds the label
../.venv/bin/python 05_validate.py          # sanity checks
```

1. **Detections.** Every VIIRS detection inside California with `type == 0` (vegetation fire): 1,504,354 detections.
2. **Fire events.** Detections within 1 km of each other are linked. A link can also reach back up to 2 days, to the nearest earlier detection. Connected groups become events: 75,222 events, most of them single-pixel farm burns or flare-ups. Features come only from the **first overpass** (detections within 30 minutes of the first one). Columns starting with `post_` describe the whole event and are for validation only, never model inputs.
3. **Labels.** Each event is matched one-to-one to an FPA FOD fire whose ignition point is within 5 km of one of the event's first-day detections, and whose discovery date is between 1 day after and 2 days before the first detection. Matches where the event is far bigger than the reported fire are rejected; this happens when tiny lightning fires sit next to a big one. Unmatched events are dropped. `label_big_fire = 1` when the final size is at least 1,000 acres (NWCG size class F and up).

## Current dataset: `data/processed/events_labeled.csv`

| | |
|---|---|
| Fires (rows) | **5,565** |
| Big fires (≥ 1,000 acres) | **487 (8.8%)** |
| FPA FOD big fires in California 2012–2024 that were matched | 76% of 639 |
| Years | 2012–2024 (planned split: train 2012–2020, validate 2021–2022, test 2023–2024) |

All eleven well-known fires checked are matched to the right event: Rim, King, Thomas, Ranch, Camp, Woolsey, Doe (August Complex), Creek, Dixie, Caldor and Park.

### Known limitations

- **Missed big fires (24%).** These had no detection within 5 km and 3 days of discovery (96), shared a satellite event with another fire in the same complex (39), or were never detected nearby (7).
- **First detection can be late.** Some fires are already large when first seen: Thomas showed 826 pixels on the first pass. 417 fires were first detected 2 days after discovery. The model should also be evaluated on the harder subset: fires that were still small (≤ 3 pixels) when first seen, which is 4,597 fires, 170 of them big.
- **Location precision.** FPA FOD ignition points can be as coarse as a 1-square-mile survey section.
- **Disproportionate years.** 2020 and 2021 hold about half of all detections. Splitting by year keeps the extreme seasons from leaking between train and test.
