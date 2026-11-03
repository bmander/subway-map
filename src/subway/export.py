"""Write the visualization dataset (web/data/day.json).

  * segs: one polyline per stop-to-stop track segment (following GTFS shapes),
    shared by every trip that runs over it;
  * trips: per trip touching the view window, its route, headsign, stations, and per
    segment the geometry index, departure/arrival time and modeled load aboard;
  * route colors, station positions, hourly entries per complex, and the
    system-wide number of passengers aboard trains per minute.
The browser expands trips x segments into timestamped paths at load time.

Geometry: for each consecutive stop pair served by a trip that has a shape, the stop
positions are projected (monotonically) onto the shape and the shape's vertices
between them become that pair's polyline. Pairs are shared across trips, so trips
without a shape_id reuse geometry from trips that have one; a straight segment is the
last resort.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from .gtfs import Network

MAX_SNAP_M = 250.0  # a stop farther than this from a shape doesn't use that shape
SIMPLIFY_M = 2.0    # Douglas-Peucker tolerance; GTFS shapes have a vertex every ~8 m
M_PER_DEG = 111_320.0


def _read(z: zipfile.ZipFile, name: str) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(z.read(name)), infer_schema_length=0)


def _xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local equirectangular metres (good to <0.1% across NYC)."""
    k = np.cos(np.radians(40.7))
    return np.stack([lon * k * M_PER_DEG, lat * M_PER_DEG], axis=-1)


def simplify(xy: np.ndarray, tol: float) -> np.ndarray:
    """Indices of the vertices Douglas-Peucker keeps (endpoints always kept)."""
    keep = np.zeros(len(xy), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        ab = b - a
        L = np.hypot(*ab)
        p = xy[i + 1:j] - a
        d = np.abs(ab[0] * p[:, 1] - ab[1] * p[:, 0]) / L if L > 0 else np.hypot(p[:, 0], p[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def segment_geometry(net: Network, gtfs_zip: Path) -> dict[tuple[int, int], np.ndarray]:
    """(from_stop_idx, to_stop_idx) -> [n, 2] array of (lon, lat) incl. both stops."""
    z = zipfile.ZipFile(gtfs_zip)
    sh = _read(z, "shapes.txt").with_columns(
        pl.col("shape_pt_lat").cast(pl.Float64), pl.col("shape_pt_lon").cast(pl.Float64),
        pl.col("shape_pt_sequence").cast(pl.Int64),
    ).sort(["shape_id", "shape_pt_sequence"])
    shapes = {
        sid: (g["shape_pt_lon"].to_numpy(), g["shape_pt_lat"].to_numpy())
        for (sid,), g in sh.group_by(["shape_id"], maintain_order=True)
    }
    s_lon, s_lat = net.stops["lon"].to_numpy(), net.stops["lat"].to_numpy()
    s_xy = _xy(s_lat, s_lon)

    geo: dict[tuple[int, int], np.ndarray] = {}
    trips = net.trips.select(["idx", "shape_id"]).drop_nulls("shape_id")
    done_patterns: set[tuple] = set()
    for t, sid in trips.iter_rows():
        if sid not in shapes:
            continue
        cs = net.trip_conns[net.trip_first[t]:net.trip_first[t + 1]]
        if len(cs) == 0:
            continue
        stops = [int(net.c_from[cs[0]])] + [int(net.c_to[c]) for c in cs]
        key = (sid, tuple(stops))
        if key in done_patterns:
            continue
        done_patterns.add(key)
        lon, lat = shapes[sid]
        v_xy = _xy(lat, lon)
        idx, prev, ok = [], 0, True
        for s in stops:
            d = np.linalg.norm(v_xy[prev:] - s_xy[s], axis=1)
            j = int(np.argmin(d))
            if d[j] > MAX_SNAP_M:
                ok = False
                break
            idx.append(prev + j)
            prev = prev + j
        if not ok:
            continue
        for k in range(len(stops) - 1):
            a, b = stops[k], stops[k + 1]
            if (a, b) in geo:
                continue
            mid = np.stack([lon[idx[k] + 1:idx[k + 1]], lat[idx[k] + 1:idx[k + 1]]], axis=1)
            g = np.vstack([[s_lon[a], s_lat[a]], mid, [s_lon[b], s_lat[b]]])
            geo[(a, b)] = g[simplify(_xy(g[:, 1], g[:, 0]), SIMPLIFY_M)]
    return geo


def _clock(s: int) -> str:
    return f"{s // 3600 % 24:02d}:{s % 3600 // 60:02d}"


def write(out_dir: Path) -> Path:
    net = Network.load(C.PROCESSED / "network")
    loads = np.load(C.PROCESSED / "loads.npy")
    gtfs_zip = C.RAW / "gtfs_subway.zip"
    z = zipfile.ZipFile(gtfs_zip)
    routes = _read(z, "routes.txt")
    headsign = dict(_read(z, "trips.txt").select(["trip_id", "trip_headsign"]).iter_rows())
    geo = segment_geometry(net, gtfs_zip)

    lo, hi = C.VIEW_START_S, C.VIEW_END_S
    s_lon, s_lat = net.stops["lon"].to_numpy(), net.stops["lat"].to_numpy()
    route_ids = sorted(net.trips["route_id"].unique().to_list())
    r_index = {r: i for i, r in enumerate(route_ids)}
    seg_index: dict[tuple[int, int], int] = {}
    segs: list[list[float]] = []
    n_straight = 0

    def seg_of(a: int, b: int) -> int:
        nonlocal n_straight
        k = seg_index.get((a, b))
        if k is None:
            g = geo.get((a, b))
            if g is None:
                g = np.array([[s_lon[a], s_lat[a]], [s_lon[b], s_lat[b]]])
                n_straight += 1
            k = seg_index[(a, b)] = len(segs)
            segs.append(np.round(g, 5).ravel().tolist())
        return k

    out_trips = []
    for t, trip_id, route_id in net.trips.select(["idx", "trip_id", "route_id"]).iter_rows():
        cs = net.trip_conns[net.trip_first[t]:net.trip_first[t + 1]]
        if len(cs) == 0 or net.c_arr[cs[-1]] < lo or net.c_dep[cs[0]] >= hi:
            continue
        out_trips.append({
            "r": r_index[route_id],
            "h": headsign.get(trip_id) or "",
            "s": [int(net.c_from[cs[0]])] + [int(net.c_to[c]) for c in cs],      # stations
            "g": [seg_of(int(net.c_from[c]), int(net.c_to[c])) for c in cs],      # segment geometry
            "d": net.c_dep[cs].astype(int).tolist(),                              # departs (s)
            "a": net.c_arr[cs].astype(int).tolist(),                              # arrives (s)
            "l": np.round(loads[cs]).astype(int).tolist(),                        # aboard
        })
    out_trips.sort(key=lambda d: d["d"][0])

    # Passengers aboard trains, system-wide, per minute of the view window.
    minutes = np.arange(lo, hi, 60)
    aboard = np.zeros(len(minutes))
    i0 = np.searchsorted(minutes, net.c_dep, side="left")
    i1 = np.searchsorted(minutes, net.c_arr, side="left")
    diff = np.zeros(len(minutes) + 1)
    np.add.at(diff, np.clip(i0, 0, len(minutes)), loads)
    np.add.at(diff, np.clip(i1, 0, len(minutes)), -loads)
    aboard = np.cumsum(diff)[:-1]

    # Stations: position, complex name, hourly entries over the view window.
    hourly = pl.read_parquet(C.PROCESSED / "hourly_entries.parquet")
    cx = net.complexes
    ent = np.zeros((cx.height, (hi - lo) // 3600))
    cx_idx = dict(zip(cx["complex_id"], cx["idx"]))
    for cid, t0, e in hourly.filter((pl.col("t0_s") >= lo) & (pl.col("t0_s") < hi)).select(
            ["complex_id", "t0_s", "entries"]).iter_rows():
        ent[cx_idx[cid], (int(t0) - lo) // 3600] += e
    stations = [
        {"n": n, "c": int(ci), "p": [round(lo_, 5), round(la_, 5)]}
        for n, ci, lo_, la_ in net.stops.select(["stop_name", "complex_idx", "lon", "lat"]).iter_rows()
    ]
    rmeta = {r["route_id"]: r for r in routes.iter_rows(named=True)}
    data = {
        "date": C.SERVICE_DATE.isoformat(),
        "t0": lo, "t1": hi,
        "routes": [{"id": r, "name": rmeta[r]["route_short_name"], "long": rmeta[r]["route_long_name"],
                    "color": "#" + (rmeta[r]["route_color"] or "808183")} for r in route_ids],
        "stations": stations,
        "complexes": [{"n": n, "e": np.round(ent[i]).astype(int).tolist()} for i, n in enumerate(cx["name"])],
        "aboard": np.round(aboard).astype(int).tolist(),
        "segs": segs,
        "trips": out_trips,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "day.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    peak = int(np.argmax(aboard))
    print(f"export: {len(out_trips):,} trips over {len(segs):,} track segments "
          f"({n_straight:,} straight-line, no shape); "
          f"peak aboard {aboard[peak]:,.0f} at {_clock(int(minutes[peak]))}; "
          f"{path.stat().st_size / 1e6:.1f} MB -> {path}")
    return path
