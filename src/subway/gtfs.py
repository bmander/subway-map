"""Build the scheduled network for the modeled day from static GTFS.

The network is at *parent station* granularity. A "connection" is one train moving
between consecutive stops: (from_stop, to_stop, dep, arr, trip). Times are seconds
after midnight of SERVICE_DATE. Trips from the previous and next service days are
included (shifted by -24h / +24h), so the timetable is continuous across midnight.

Stations are grouped into fare-control complexes (where turnstile and hourly data
live) using the MTA stations table; transfers come from GTFS transfers.txt.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C

EXCLUDED_ROUTES = {"SI"}  # Staten Island Railway: separate system, no turnstile data
# Seconds between passing fare control and being able to board, and vice versa.
ACCESS_S = 90
EGRESS_S = 90
# Default change time at a station when transfers.txt has no self-transfer row.
DEFAULT_CHANGE_S = 180


def _read(z: zipfile.ZipFile, name: str) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(z.read(name)), infer_schema_length=0)


def _hms(col: str) -> pl.Expr:
    p = pl.col(col).str.split(":")
    return (
        p.list.get(0).cast(pl.Int64) * 3600 + p.list.get(1).cast(pl.Int64) * 60 + p.list.get(2).cast(pl.Int64)
    )


def active_services(cal: pl.DataFrame, cal_dates: pl.DataFrame, day: dt.date) -> set[str]:
    dow = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][day.weekday()]
    ymd = day.strftime("%Y%m%d")
    on = set(
        cal.filter((pl.col(dow) == "1") & (pl.col("start_date") <= ymd) & (pl.col("end_date") >= ymd))["service_id"]
    )
    ex = cal_dates.filter(pl.col("date") == ymd)
    on |= set(ex.filter(pl.col("exception_type") == "1")["service_id"])
    on -= set(ex.filter(pl.col("exception_type") == "2")["service_id"])
    return on


@dataclass
class Network:
    stops: pl.DataFrame       # idx, stop_id, stop_name, lat, lon, complex_idx (-1 if none)
    complexes: pl.DataFrame   # idx, complex_id, name, lat, lon
    trips: pl.DataFrame       # idx, trip_id, route_id, direction_id, shape_id, day_offset
    # connections, sorted by departure time
    c_dep: np.ndarray
    c_arr: np.ndarray
    c_from: np.ndarray
    c_to: np.ndarray
    c_trip: np.ndarray
    c_seq: np.ndarray         # position of the connection within its trip (0-based)
    trip_conns: np.ndarray    # connection ids grouped by trip, in trip order
    trip_first: np.ndarray    # trip t's connections are trip_conns[trip_first[t]:trip_first[t+1]]
    change_s: np.ndarray      # per stop: time to change trains at this stop
    xfer_ptr: np.ndarray      # CSR over stops: walking transfers to *other* stops
    xfer_to: np.ndarray
    xfer_s: np.ndarray
    cx_ptr: np.ndarray        # CSR over complexes: member stops
    cx_stops: np.ndarray

    @property
    def n_stops(self) -> int:
        return self.stops.height

    @property
    def n_complexes(self) -> int:
        return self.complexes.height

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.stops.write_parquet(d / "stops.parquet")
        self.complexes.write_parquet(d / "complexes.parquet")
        self.trips.write_parquet(d / "trips.parquet")
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez(d / "arrays.npz", **arrays)

    @classmethod
    def load(cls, d: Path) -> "Network":
        a = np.load(d / "arrays.npz")
        return cls(
            stops=pl.read_parquet(d / "stops.parquet"),
            complexes=pl.read_parquet(d / "complexes.parquet"),
            trips=pl.read_parquet(d / "trips.parquet"),
            **{k: a[k] for k in a.files},
        )


def _csr(src: np.ndarray, dst: np.ndarray, val: np.ndarray, n: int):
    order = np.lexsort((dst, src))
    src, dst, val = src[order], dst[order], val[order]
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, src + 1, 1)
    return np.cumsum(ptr), dst.astype(np.int32), val.astype(np.int32)


def build(gtfs_zip: Path, stations: pl.DataFrame, hourly: pl.DataFrame,
          t_min: int, t_max: int) -> Network:
    """Connections departing in [t_min, t_max) seconds after SERVICE_DATE midnight."""
    z = zipfile.ZipFile(gtfs_zip)
    stops = _read(z, "stops.txt")
    trips = _read(z, "trips.txt").filter(~pl.col("route_id").is_in(EXCLUDED_ROUTES))
    st = _read(z, "stop_times.txt").select(["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    cal, cal_dates = _read(z, "calendar.txt"), _read(z, "calendar_dates.txt")
    transfers = _read(z, "transfers.txt")

    # Trips running on (day-1, day, day+1), with their time offset.
    parts = []
    for off in (-1, 0, 1):
        svc = active_services(cal, cal_dates, C.SERVICE_DATE + dt.timedelta(days=off))
        parts.append(trips.filter(pl.col("service_id").is_in(svc)).with_columns(pl.lit(off * 86400).alias("day_offset")))
    trips = pl.concat(parts)

    # Stop times -> parent stations, times in modeled-day seconds.
    parent = stops.select(["stop_id", pl.coalesce(pl.col("parent_station"), pl.col("stop_id")).alias("parent")])
    st = (
        st.join(trips.select(["trip_id", "day_offset"]), on="trip_id")
        .join(parent, on="stop_id")
        .with_columns(
            (_hms("arrival_time") + pl.col("day_offset")).alias("arr"),
            (_hms("departure_time") + pl.col("day_offset")).alias("dep"),
            pl.col("stop_sequence").cast(pl.Int64),
        )
        .sort(["trip_id", "day_offset", "stop_sequence"])
    )
    # Keep trips that touch the window at all.
    span = st.group_by(["trip_id", "day_offset"]).agg(pl.col("dep").min().alias("t0"), pl.col("arr").max().alias("t1"))
    keep = span.filter((pl.col("t1") >= t_min) & (pl.col("t0") < t_max)).select(["trip_id", "day_offset"])
    st = st.join(keep, on=["trip_id", "day_offset"])
    trips = trips.join(keep, on=["trip_id", "day_offset"]).sort(["route_id", "trip_id", "day_offset"])
    trips = trips.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))

    # Parent stations that are served, with complex membership.
    served = st["parent"].unique().sort()
    par = (
        stops.filter(pl.col("stop_id").is_in(served))
        .select(["stop_id", "stop_name", pl.col("stop_lat").cast(pl.Float64).alias("lat"),
                 pl.col("stop_lon").cast(pl.Float64).alias("lon")])
        .sort("stop_id").with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    )
    cx_names = hourly.group_by("station_complex_id").agg(
        pl.col("station_complex").first().alias("name"), pl.col("latitude").first().alias("lat"),
        pl.col("longitude").first().alias("lon"))
    cx = (
        stations.select(["gtfs_stop_id", "complex_id"]).unique()
        .join(cx_names, left_on="complex_id", right_on="station_complex_id")  # drops SIR etc.
    )
    complexes = (
        cx.select(["complex_id", "name", "lat", "lon"]).unique("complex_id")
        .sort(pl.col("complex_id").cast(pl.Int64, strict=False), "complex_id")
        .with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    )
    par = (
        par.join(cx.select(["gtfs_stop_id", "complex_id"]), left_on="stop_id", right_on="gtfs_stop_id", how="left")
        .join(complexes.select(["complex_id", pl.col("idx").alias("complex_idx")]), on="complex_id", how="left")
        .with_columns(pl.col("complex_idx").fill_null(-1).cast(pl.Int32))
        .sort("idx")
    )
    stop_idx = dict(zip(par["stop_id"], par["idx"]))

    # Connections between consecutive stops of each trip.
    st = st.join(trips.select(["trip_id", "day_offset", "idx"]), on=["trip_id", "day_offset"]).with_columns(
        pl.col("parent").replace_strict(stop_idx, return_dtype=pl.Int32).alias("s")
    )
    cn = (
        st.with_columns(
            pl.col("s").shift(-1).over("idx").alias("s_to"),
            pl.col("arr").shift(-1).over("idx").alias("arr_to"),
        )
        .drop_nulls("s_to")
        .with_columns((pl.col("stop_sequence").rank("ordinal").over("idx") - 1).cast(pl.Int32).alias("seq"))
        .select([pl.col("idx").alias("trip"), "seq", pl.col("s").alias("from"), pl.col("s_to").alias("to"),
                 "dep", pl.col("arr_to").alias("arr")])
    )
    cn = cn.filter((pl.col("arr") >= t_min) & (pl.col("dep") < t_max))
    bad = cn.filter(pl.col("arr") < pl.col("dep")).height
    if bad:
        raise ValueError(f"{bad} connections arrive before they depart")
    cn = cn.sort(["dep", "arr", "trip", "seq"]).with_row_index("cid")

    # Per-trip ordered connection lists (CSR).
    by_trip = cn.sort(["trip", "seq"])
    n_trips = trips.height
    counts = np.bincount(by_trip["trip"].to_numpy(), minlength=n_trips)
    trip_first = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    # Re-number seq to be contiguous within the (window-clipped) trip.
    tc = by_trip["cid"].to_numpy().astype(np.int64)
    c_seq = np.empty(cn.height, dtype=np.int32)
    c_seq[tc] = np.arange(cn.height) - np.repeat(trip_first[:-1], counts)

    # Transfers: self-transfer -> change time; others -> walking edges.
    tr = transfers.filter(pl.col("from_stop_id").is_in(stop_idx.keys()) & pl.col("to_stop_id").is_in(stop_idx.keys()))
    tr = tr.with_columns(
        pl.col("from_stop_id").replace_strict(stop_idx, return_dtype=pl.Int32).alias("a"),
        pl.col("to_stop_id").replace_strict(stop_idx, return_dtype=pl.Int32).alias("b"),
        pl.col("min_transfer_time").cast(pl.Int32).alias("t"),
    )
    change = np.full(par.height, DEFAULT_CHANGE_S, dtype=np.int32)
    selfs = tr.filter(pl.col("a") == pl.col("b"))
    change[selfs["a"].to_numpy()] = selfs["t"].to_numpy()
    walk = tr.filter(pl.col("a") != pl.col("b"))
    xfer_ptr, xfer_to, xfer_s = _csr(walk["a"].to_numpy(), walk["b"].to_numpy(), walk["t"].to_numpy(), par.height)

    member = par.filter(pl.col("complex_idx") >= 0)
    cx_ptr, cx_stops, _ = _csr(member["complex_idx"].to_numpy(), member["idx"].to_numpy(),
                               np.zeros(member.height), complexes.height)

    return Network(
        stops=par.select(["idx", "stop_id", "stop_name", "lat", "lon", "complex_idx"]),
        complexes=complexes,
        trips=trips.select(["idx", "trip_id", "route_id", "direction_id", "shape_id", "day_offset"]),
        c_dep=cn["dep"].to_numpy().astype(np.int32),
        c_arr=cn["arr"].to_numpy().astype(np.int32),
        c_from=cn["from"].to_numpy().astype(np.int32),
        c_to=cn["to"].to_numpy().astype(np.int32),
        c_trip=cn["trip"].to_numpy().astype(np.int32),
        c_seq=c_seq,
        trip_conns=tc,
        trip_first=trip_first,
        change_s=change,
        xfer_ptr=xfer_ptr, xfer_to=xfer_to, xfer_s=xfer_s,
        cx_ptr=cx_ptr, cx_stops=cx_stops,
    )
