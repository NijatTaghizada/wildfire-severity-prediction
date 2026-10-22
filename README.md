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
| Weather and fuel dryness | gridMET, Abatzoglou (2013), via the [University of Idaho THREDDS server](http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_MET_aggregated_catalog.html) | 4 km, daily |
| Vegetation greenness | MODIS MOD13Q1 NDVI via the [ORNL DAAC MODIS web service](https://modis.ornl.gov/data/modis_webservice.html) | 250 m, 16-day |
| Fuel type and terrain | LANDFIRE LF2016 FBFM40 fuel models; LF2020 elevation, slope and aspect ([LANDFIRE services](https://lfps.usgs.gov/arcgis/rest/services)) | 30 m |

None of these sources needs an account. Every feature is sampled at the centre of the fire's first satellite pass, using only data from before the detection. Weather comes from the previous day, plus 7-day, 30-day or 60-day look-backs. NDVI comes from the latest 16-day composite that ended before the fire.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd scripts
../.venv/bin/python 01_download_firms.py    # ~850 MB streamed, keeps California vegetation fires
../.venv/bin/python 02_download_labels.py   # FPA FOD (~230 MB) and CAL FIRE perimeters (~250 MB)
../.venv/bin/python 03_build_events.py      # groups detections into fire events
../.venv/bin/python 04_match_labels.py      # links events to FPA FOD records, adds the label
../.venv/bin/python 05_validate.py          # sanity checks
../.venv/bin/python 06_features_gridmet.py  # ~55k point queries, about 1 hour; resumable
../.venv/bin/python 07_features_landfire.py # terrain and fuel, a few minutes
../.venv/bin/python 08_features_ndvi.py     # ~5.5k queries, 1-2 hours; resumable
../.venv/bin/python 09_build_training_table.py
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

### Training table: `data/processed/training_table.csv`

The same 5,565 fires with **40 inputs** added. 12 are satellite first-pass inputs, 15 weather and fuel dryness, 6 vegetation, fuel and terrain, and 6 place and time. Fewer than 0.1% of values are missing.
The `split` column assigns train 2012–2020, validation 2021–2022 and test 2023–2024. `hard_case` marks fires that were still ≤ 3 pixels when first seen.

| Split | Fires | Big fires | Hard cases | Big hard cases |
|---|---|---|---|---|
| train | 3,603 | 368 | 2,957 | 120 |
| val | 938 | 53 | 810 | 24 |
| test | 1,024 | 66 | 830 | 26 |

Median values, small vs. big fires: first-pass pixels 1 vs. 7; previous-day ERC 66 vs. 75; minimum humidity 20% vs. 15%; days since rain 42 vs. 53; slope 4° vs. 15°; elevation 274 m vs. 803 m.

All eleven well-known fires checked are matched to the right event: Rim, King, Thomas, Ranch, Camp, Woolsey, Doe (August Complex), Creek, Dixie, Caldor and Park.

### Known limitations

- **Missed big fires (24%).** These had no detection within 5 km and 3 days of discovery (96), shared a satellite event with another fire in the same complex (39), or were never detected nearby (7).
- **First detection can be late.** Some fires are already large when first seen: Thomas showed 826 pixels on the first pass. 417 fires were first detected 2 days after discovery. The model should also be evaluated on the harder subset: fires that were still small (≤ 3 pixels) when first seen, which is 4,597 fires, 170 of them big.
- **Fuel map timing.** LF2016 shows roughly 2016 vegetation, so for 2012–2016 fires it can already contain their own burn scar. The 2023–2024 test fires are unaffected.
- **Location precision.** FPA FOD ignition points can be as coarse as a 1-square-mile survey section.
- **Disproportionate years.** 2020 and 2021 hold about half of all detections. Splitting by year keeps the extreme seasons from leaking between train and test.
