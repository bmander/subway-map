"""Project-wide constants: the modeled day, data sources, and paths."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- The day we model -------------------------------------------------------
# Wednesday 2020-02-05: a high-ridership, non-holiday, non-school-recess weekday a
# month before COVID, covered by the archived Fall-2019 ("FA19") GTFS pick.
SERVICE_DATE = dt.date(2020, 2, 5)

# Times are seconds after midnight of SERVICE_DATE (GTFS-style; may exceed 24h).
# Demand (entries) is modeled over a 30h span so that every exit-count interval we
# fit is fully covered by modeled trips; the visualization shows the 24h service day
# 03:00 -> 27:00, which is the subway's quietest boundary.
DEMAND_START_S = 0
DEMAND_END_S = 30 * 3600
VIEW_START_S = 3 * 3600
VIEW_END_S = 27 * 3600
# Longest plausible journey; trains are kept this far past the demand window.
ROUTING_HORIZON_S = 3 * 3600

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

# --- Sources -------------------------------------------------------------------
SOCRATA = "https://data.ny.gov/resource"
DATASET_TURNSTILE_2020 = "py8k-a8wg"   # MTA Subway Turnstile Usage Data: 2020 (retired)
DATASET_HOURLY_2020_2024 = "wujg-7c2s" # MTA Subway Hourly Ridership: 2020-2024
DATASET_STATIONS = "39hk-dx4f"         # MTA Subway Stations (gtfs_stop_id <-> complex_id)
# Validation only, never used in fitting: MTA's own modeled OD estimates. They start in
# 2021, so we compare the nearest like-for-like slice (February Wednesdays, 2023) by
# *shares*, since post-pandemic volumes and commute patterns differ from 2020.
DATASET_OD_2023 = "uhf3-t34z"          # MTA Subway Origin-Destination Ridership Estimate: 2023
OD_REF_MONTH, OD_REF_DOW = 2, "Wednesday"

# Static GTFS in effect on SERVICE_DATE: the MTA feed as archived by the Wayback
# Machine on 2020-04-26 (pick FA19, valid 2020-01-02..2020-05-01). The `id_` flag asks
# for the original bytes; the archive still serves them gzip-wrapped.
GTFS_URL = (
    "https://web.archive.org/web/20200426183023id_/"
    "http://web.mta.info/developers/data/nyct/subway/google_transit.zip"
)


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
