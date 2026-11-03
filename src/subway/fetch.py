"""Download raw inputs into data/raw. Every function is idempotent (skips cached files)."""

from __future__ import annotations

import datetime as dt
import gzip
import io
import time
import zipfile
from pathlib import Path

import requests

from . import config as C

PAGE = 50_000


def _get(url: str, params: dict[str, str] | None = None, tries: int = 4) -> requests.Response:
    """GET with a polite backoff on 429/5xx (honours Retry-After)."""
    for i in range(tries):
        r = requests.get(url, params=params, timeout=300)
        if r.status_code != 429 and r.status_code < 500:
            break
        wait = float(r.headers.get("Retry-After", 30 * (i + 1)))
        print(f"  {r.status_code} from {url.split('/')[2]}; retrying in {wait:.0f}s")
        time.sleep(wait)
    r.raise_for_status()
    return r


def _socrata_csv(dataset: str, params: dict[str, str], dest: Path, order: str) -> Path:
    """Page through a Socrata CSV query and concatenate the pages into `dest`."""
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    offset, header, n = 0, None, 0
    with tmp.open("w", encoding="utf-8") as out:
        while True:
            q = {**params, "$order": order, "$limit": str(PAGE), "$offset": str(offset)}
            r = _get(f"{C.SOCRATA}/{dataset}.csv", params=q)
            lines = r.text.splitlines(keepends=True)
            if not lines:
                break
            if header is None:
                header = lines[0]
                out.write(header)
            elif lines[0] != header:
                raise RuntimeError(f"header changed between pages of {dataset}")
            rows = lines[1:]
            out.writelines(rows)
            n += len(rows)
            if len(rows) < PAGE:
                break
            offset += PAGE
    tmp.rename(dest)
    print(f"  {dest.name}: {n:,} rows")
    return dest


def gtfs() -> Path:
    dest = C.RAW / "gtfs_subway.zip"
    if dest.exists():
        return dest
    r = _get(C.GTFS_URL)
    body = r.content
    if body[:2] == b"\x1f\x8b":  # Wayback serves this capture gzip-wrapped
        body = gzip.decompress(body)
    zipfile.ZipFile(io.BytesIO(body)).testzip()  # raises if not a valid zip
    dest.write_bytes(body)
    print(f"  {dest.name}: {len(body):,} bytes")
    return dest


def stations() -> Path:
    return _socrata_csv(C.DATASET_STATIONS, {}, C.RAW / "stations.csv", order="gtfs_stop_id")


def _ts(d: dt.date, hour: int = 0) -> str:
    return (dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%S")


def hourly_entries() -> Path:
    """Hourly subway entries per station complex, summed over fare types.

    Covers the demand window [DEMAND_START_S, DEMAND_END_S).
    """
    d = C.SERVICE_DATE
    lo = _ts(d, C.DEMAND_START_S // 3600)
    hi = _ts(d, C.DEMAND_END_S // 3600)
    params = {
        "$select": (
            "transit_timestamp, station_complex_id, station_complex, borough, "
            "latitude, longitude, sum(ridership) as entries, sum(transfers) as transfers"
        ),
        "$where": f"transit_mode='subway' and transit_timestamp >= '{lo}' and transit_timestamp < '{hi}'",
        "$group": "transit_timestamp, station_complex_id, station_complex, borough, latitude, longitude",
    }
    return _socrata_csv(
        C.DATASET_HOURLY_2020_2024, params, C.RAW / f"hourly_entries_{d:%Y%m%d}.csv",
        order="station_complex_id, transit_timestamp",
    )


def turnstiles(days_before: int = 4, days_after: int = 3) -> Path:
    """Raw cumulative turnstile register reads for a week around the service date.

    The surrounding days give each turnstile a history to judge outliers against.
    """
    d = C.SERVICE_DATE
    lo, hi = d - dt.timedelta(days=days_before), d + dt.timedelta(days=days_after + 1)
    params = {"$where": f"date >= '{_ts(lo)}' and date < '{_ts(hi)}'"}
    return _socrata_csv(
        C.DATASET_TURNSTILE_2020, params, C.RAW / f"turnstiles_{lo:%Y%m%d}_{hi:%Y%m%d}.csv",
        order="c_a, unit, scp, date, time, description",
    )


def od_reference() -> list[Path]:
    """MTA modeled OD (validation only): avg February-Wednesday trips per complex pair,
    for the whole day and for the 7-10 AM peak (entry hour)."""
    out = []
    for name, extra in (("day", ""), ("am", " and hour_of_day >= 7 and hour_of_day < 10")):
        params = {
            "$select": "origin_station_complex_id as o, destination_station_complex_id as d, "
                       "sum(estimated_average_ridership) as trips",
            "$where": f"month = {C.OD_REF_MONTH} and day_of_week = '{C.OD_REF_DOW}'{extra}",
            "$group": "origin_station_complex_id, destination_station_complex_id",
        }
        out.append(_socrata_csv(C.DATASET_OD_2023, params, C.RAW / f"od_2023_{name}.csv", order="o, d"))
    return out


def all_() -> None:
    C.ensure_dirs()
    for f in (gtfs, stations, hourly_entries, turnstiles, od_reference):
        print(f"fetch {f.__name__}")
        f()
