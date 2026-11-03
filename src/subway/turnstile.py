"""Clean MTA turnstile register reads into per-turnstile interval counts.

Raw data: for each turnstile (SCP) inside a control area (C/A) belonging to a remote
unit (UNIT), cumulative ENTRIES/EXITS registers read roughly every 4 hours. Each C/A
is read on a fixed "phase": an offset within the 4-hour cycle (mostly 03:00/07:00/...
or 00:00/04:00/..., but some C/As read at odd minutes like 03:22/07:22/...).

Cleaning, per turnstile and per register:
  1. keep subway divisions; parse timestamps; drop duplicate reads (prefer REGULAR);
  2. difference consecutive reads;
  3. flip registers that count backwards (most nonzero deltas negative);
  4. reject deltas that are negative or exceed a physical rate cap (counter resets,
     rollovers, garbage), and impute them from the same turnstile's median rate at
     the same hour of day on other weekdays of the week;
  5. integrate the clean deltas back into a cumulative curve and resample it
     (linearly in time) onto the turnstile's own 4-hour phase grid, so every
     turnstile in the system lands on one of four regular grids.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

SUBWAY_DIVISIONS = ("IRT", "IND", "BMT")
# Control areas that are not subway fare control: JFK AirTrain gates at Howard Beach
# and Jamaica are filed under division IND.
EXCLUDE_CA_PREFIXES = ("JFK",)
KEY = ["c_a", "unit", "scp"]
# Fastest credible sustained rate through one turnstile. The busiest observed 4h
# intervals are ~3,000 (~750/h); a counter reset produces millions.
MAX_RATE_PER_H = 1_500
GRID_H = 4


def load_raw(path: Path) -> pl.DataFrame:
    t = pl.read_csv(path, infer_schema_length=0)
    return (
        t.filter(
            pl.col("division").is_in(SUBWAY_DIVISIONS)
            & ~pl.any_horizontal(pl.col("c_a").str.starts_with(p) for p in EXCLUDE_CA_PREFIXES)
        )
        .with_columns(
            (pl.col("date").str.slice(0, 10) + " " + pl.col("time"))
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .alias("ts"),
            pl.col("entries").cast(pl.Int64),
            pl.col("exits").cast(pl.Int64),
        )
        # "REGULAR" sorts after "RECOVR AUD"; descending puts REGULAR first.
        .sort(KEY + ["ts", "description"], descending=[False] * 4 + [True])
        .unique(KEY + ["ts"], keep="first", maintain_order=True)
        .select(KEY + ["station", "line_name", "division", "ts", "description", "entries", "exits"])
    )


def intervals(raw: pl.DataFrame) -> pl.DataFrame:
    """One row per consecutive read pair, with cleaned & imputed deltas per register."""
    d = raw.with_columns(
        pl.col("ts").shift(1).over(KEY).alias("t0"),
        (pl.col("entries") - pl.col("entries").shift(1).over(KEY)).alias("d_entries"),
        (pl.col("exits") - pl.col("exits").shift(1).over(KEY)).alias("d_exits"),
    ).drop_nulls("t0").rename({"ts": "t1"})
    d = d.with_columns(((pl.col("t1") - pl.col("t0")).dt.total_seconds() / 3600).alias("dur_h"))
    for reg in ("entries", "exits"):
        dc = f"d_{reg}"
        # 3. backwards registers
        backwards = (
            d.filter(pl.col(dc) != 0)
            .group_by(KEY)
            .agg(((pl.col(dc) < 0).mean() > 0.5).alias("_bw"))
        )
        d = d.join(backwards, on=KEY, how="left").with_columns(
            pl.when(pl.col("_bw").fill_null(False)).then(-pl.col(dc)).otherwise(pl.col(dc)).alias(dc)
        ).drop("_bw")
        # 4. reject implausible deltas
        ok = (pl.col(dc) >= 0) & (pl.col(dc) <= MAX_RATE_PER_H * pl.col("dur_h"))
        d = d.with_columns(
            ok.alias(f"ok_{reg}"),
            pl.when(ok).then(pl.col(dc).cast(pl.Float64)).otherwise(None).alias(dc),
        )
        d = _impute(d, reg)
    return d


def _impute(d: pl.DataFrame, reg: str) -> pl.DataFrame:
    """Fill rejected deltas with dur x the turnstile's median weekday rate at that hour."""
    dc = f"d_{reg}"
    d = d.with_columns(
        pl.col("t0").dt.hour().alias("_h0"),
        (pl.col("t0").dt.weekday() <= 5).alias("_wkday"),
    )
    ref = (
        d.filter(pl.col(f"ok_{reg}") & pl.col("_wkday") & (pl.col("dur_h") >= 1))
        .group_by(KEY + ["_h0"])
        .agg((pl.col(dc) / pl.col("dur_h")).median().alias("_rate"))
    )
    d = d.join(ref, on=KEY + ["_h0"], how="left")
    return d.with_columns(
        pl.col(dc).fill_null(pl.col("_rate") * pl.col("dur_h")).fill_null(0.0)
    ).drop(["_h0", "_wkday", "_rate"])


def phases(raw: pl.DataFrame) -> pl.DataFrame:
    """Each turnstile's read phase: modal offset (seconds, minute-rounded) within the
    4-hour audit cycle. Most C/As read at :00 on hours = 3 or 0 (mod 4), but some read
    at odd minutes (03:22, 07:22, ...), so the phase is not just an hour."""
    cyc = GRID_H * 3600
    # dt.hour()/dt.minute() are Int8: widen before multiplying or 23*3600 overflows.
    sod = pl.col("ts").dt.hour().cast(pl.Int32) * 3600 + pl.col("ts").dt.minute().cast(pl.Int32) * 60
    return (
        raw.filter(pl.col("description") == "REGULAR")
        .group_by(KEY)
        .agg((sod % cyc).mode().min().alias("phase_s"))
    )


def resample(iv: pl.DataFrame, ph: pl.DataFrame, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
    """Resample each turnstile's cleaned cumulative counts onto its phase grid.

    Returns one row per (turnstile, grid interval) inside [start, end), with the
    fraction of the interval's count that came from imputed deltas.
    """
    iv = iv.join(ph, on=KEY, how="inner").sort(KEY + ["t1"])
    out: list[dict] = []
    epoch = np.datetime64(start, "s")
    for (c_a, unit, scp), g in iv.group_by(KEY, maintain_order=True):
        phase_s = int(g["phase_s"][0])
        # Cumulative curve at read times (seconds since `start`).
        t = np.concatenate([[g["t0"][0]], g["t1"].to_numpy()]).astype("datetime64[s]")
        ts = (t - epoch).astype(np.int64).astype(float)
        cols = {}
        for reg in ("entries", "exits"):
            dlt = g[f"d_{reg}"].to_numpy()
            bad = np.where(g[f"ok_{reg}"].to_numpy(), 0.0, dlt)
            cols[reg] = np.concatenate([[0.0], np.cumsum(dlt)])
            cols[f"imp_{reg}"] = np.concatenate([[0.0], np.cumsum(bad)])
        start_sod = start.hour * 3600 + start.minute * 60 + start.second
        first = (phase_s - start_sod) % (GRID_H * 3600)
        grid = np.arange(first, (end - start).total_seconds() + 1, GRID_H * 3600, dtype=float)
        grid = grid[(grid >= ts[0]) & (grid <= ts[-1])]
        if len(grid) < 2:
            continue
        vals = {k: np.diff(np.interp(grid, ts, v)) for k, v in cols.items()}
        for i in range(len(grid) - 1):
            out.append(
                {
                    "c_a": c_a, "unit": unit, "scp": scp, "phase_s": phase_s,
                    "t0_s": int(grid[i]), "t1_s": int(grid[i + 1]),
                    "entries": vals["entries"][i], "exits": vals["exits"][i],
                    "imp_entries": vals["imp_entries"][i], "imp_exits": vals["imp_exits"][i],
                }
            )
    return pl.DataFrame(out).with_columns(
        (pl.lit(start) + pl.duration(seconds=pl.col("t0_s"))).alias("t0"),
        (pl.lit(start) + pl.duration(seconds=pl.col("t1_s"))).alias("t1"),
    ).drop(["t0_s", "t1_s"])


def units(raw: pl.DataFrame) -> pl.DataFrame:
    """Descriptive attributes of each remote unit (for the station crosswalk)."""
    return (
        raw.group_by("unit")
        .agg(
            pl.col("station").mode().first(),
            pl.col("station").unique().alias("station_names"),
            pl.col("line_name").mode().first(),
            pl.col("division").mode().first(),
            pl.col("c_a").unique().alias("c_as"),
            pl.col("scp").n_unique().alias("n_scp"),
        )
        .sort("unit")
    )
