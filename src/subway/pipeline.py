"""Stage orchestration. Each stage reads data/raw or earlier stages, writes parquet."""

from __future__ import annotations

import datetime as dt

import polars as pl

from . import config as C
from . import crosswalk as X
from . import turnstile as T


def service_midnight() -> dt.datetime:
    return dt.datetime.combine(C.SERVICE_DATE, dt.time())


def load_hourly() -> pl.DataFrame:
    path = C.RAW / f"hourly_entries_{C.SERVICE_DATE:%Y%m%d}.csv"
    return pl.read_csv(path, infer_schema_length=0).with_columns(
        pl.col("transit_timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.3f"),
        pl.col("entries").cast(pl.Float64),
        pl.col("transfers").cast(pl.Float64),
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64),
    )


def load_stations() -> pl.DataFrame:
    return pl.read_csv(C.RAW / "stations.csv", infer_schema_length=0)


def clean() -> None:
    """Turnstiles -> regular grids; units -> complexes; validate; write observations."""
    C.ensure_dirs()
    raw_path = next(C.RAW.glob("turnstiles_*.csv"))
    raw = T.load_raw(raw_path)
    iv = T.intervals(raw)
    rejected = {r: int((~iv[f"ok_{r}"]).sum()) for r in ("entries", "exits")}
    print(f"turnstile intervals: {iv.height:,}; rejected+imputed: {rejected}")
    ph = T.phases(raw)
    lo = raw["ts"].min().replace(hour=0, minute=0, second=0)
    hi = raw["ts"].max()
    grid = T.resample(iv, ph, lo, hi)
    grid.write_parquet(C.INTERIM / "turnstile_grid.parquet")

    units = T.units(raw)
    hourly = load_hourly()
    xw = X.match(units, X.complexes(load_stations(), hourly))
    xw.write_parquet(C.INTERIM / "crosswalk.parquet")

    # Validation: each turnstile's 24h starting at its first read-phase boundary
    # after service-day midnight, vs the hourly dataset's calendar day.
    mid = service_midnight()
    off = pl.duration(seconds=pl.col("phase_s"))
    day = grid.filter((pl.col("t0") >= pl.lit(mid) + off) & (pl.col("t1") <= pl.lit(mid) + off + pl.duration(days=1)))
    hday = hourly.filter((pl.col("transit_timestamp") >= mid) & (pl.col("transit_timestamp") < mid + dt.timedelta(days=1)))
    v = X.validate(xw, day, hday)
    v.write_parquet(C.INTERIM / "crosswalk_validation.parquet")
    q = [round(v["ratio"].quantile(x), 3) for x in (0.05, 0.25, 0.5, 0.75, 0.95)]
    print(f"crosswalk: {xw.height} units -> {xw['complex_id'].n_unique()} complexes; "
          f"turnstile/hourly entry ratio quantiles (5,25,50,75,95%): {q}")
    bad = v.filter((pl.col("ratio") < 0.8) | (pl.col("ratio") > 1.25) | pl.col("ratio").is_null())
    if bad.height:
        print(f"{bad.height} complexes outside [0.8, 1.25]:")
        print(bad.select(["station_complex_id", "station_complex", "hr_entries", "ts_entries", "ratio", "units"]))

    # Observations for the inversion ------------------------------------------------
    # Exits: per (complex, read phase, grid interval), summed over that complex's
    # turnstiles on that phase. Seconds are relative to service-day midnight.
    secs = lambda c: (pl.col(c) - pl.lit(mid)).dt.total_seconds()  # noqa: E731
    exits = (
        grid.join(xw.select(["unit", "complex_id"]), on="unit")
        .group_by(["complex_id", "phase_s", "t0", "t1"])
        .agg(pl.col("exits").sum(), pl.col("entries").sum(), pl.col("imp_exits").sum(),
             pl.col("scp").n_unique().alias("n_turnstiles"))
        .with_columns(secs("t0").alias("t0_s"), secs("t1").alias("t1_s"))
        .sort(["complex_id", "phase_s", "t0"])
    )
    exits.write_parquet(C.PROCESSED / "turnstile_obs.parquet")
    entries = (
        hourly.with_columns(secs("transit_timestamp").alias("t0_s"))
        .select(["station_complex_id", "station_complex", "t0_s", "entries"])
        .rename({"station_complex_id": "complex_id"})
        .sort(["complex_id", "t0_s"])
    )
    entries.write_parquet(C.PROCESSED / "hourly_entries.parquet")
    print(f"wrote {exits.height:,} turnstile observations, {entries.height:,} hourly entry rows")


# --- Modeling stages ---------------------------------------------------------------

def network() -> None:
    from . import gtfs as G
    net = G.build(C.RAW / "gtfs_subway.zip", load_stations(), load_hourly(),
                  C.DEMAND_START_S, C.DEMAND_END_S + C.ROUTING_HORIZON_S)
    net.save(C.PROCESSED / "network")
    print(f"network: {net.n_stops} stations, {net.n_complexes} complexes, {net.trips.height:,} trips, "
          f"{len(net.c_dep):,} connections")


def route() -> None:
    import numpy as np

    from . import routing as R
    from .gtfs import Network
    net = Network.load(C.PROCESSED / "network")
    tt, legs = R.travel_times(net, C.DEMAND_START_S, C.DEMAND_END_S, horizon=C.ROUTING_HORIZON_S)
    unreachable = float(np.isnan(tt).mean() - 1 / tt.shape[0])  # minus the o == d diagonal
    print(f"travel times {tt.shape}: unreachable (o != d) {max(unreachable, 0):.3%}")
    np.savez_compressed(C.PROCESSED / "travel_times.npz", tt=tt, legs=legs)


def calibrate() -> None:
    from . import calibrate as K
    K.run()


def fit() -> None:
    import numpy as np

    from . import inversion as I
    from .gtfs import Network
    net = Network.load(C.PROCESSED / "network")
    z = np.load(C.PROCESSED / "travel_times.npz")
    obs = I.build_observations(net.complexes, pl.read_parquet(C.PROCESSED / "hourly_entries.parquet"),
                               pl.read_parquet(C.PROCESSED / "turnstile_obs.parquet"))
    print(f"observations: {int(obs.is_exit.sum())} exit intervals in {obs.n_groups} groups, "
          f"{int((~obs.is_exit).sum())} symmetry; entries {obs.E.sum():,.0f}")
    shape, scale = I.deterrence_params()
    print(f"deterrence: f(c) = c^{shape:.3f} exp(-c / {scale:.2f} min)"
          f"{' (calibrated)' if I.DETERRENCE_FILE.exists() else ' (a priori default)'}")
    res = I.fit(obs, z["tt"], I.deterrence(z["tt"], z["legs"], shape, scale))
    h = res.history[-1]
    print(f"fit: exits rel.err {h['rel_err_exit']:.3f}, symmetry rel.err {h['rel_err_sym']:.3f}; "
          f"kappa 5-95% {np.quantile(res.kappa, .05):.2f}-{np.quantile(res.kappa, .95):.2f}")
    np.savez_compressed(C.PROCESSED / "demand.npz", q=res.q.astype(np.float32), logA=res.logA,
                        kappa=res.kappa, m=res.m, mu=res.mu)
    pl.DataFrame(res.history).write_parquet(C.PROCESSED / "fit_history.parquet")


def assign() -> None:
    import numpy as np

    from . import routing as R
    from .gtfs import Network
    net = Network.load(C.PROCESSED / "network")
    q = np.load(C.PROCESSED / "demand.npz")["q"]
    loads, unrouted = R.assign_loads(net, q, C.DEMAND_START_S, horizon=C.ROUTING_HORIZON_S)
    np.save(C.PROCESSED / "loads.npy", loads)
    hours = float(np.sum(loads * (net.c_arr - net.c_dep)) / 3600)
    print(f"assigned {q.sum():,.0f} trips ({unrouted:,.0f} unrouted); {hours:,.0f} passenger-hours aboard")


def validate() -> None:
    from . import validate as V
    V.print_report(V.report())


def export() -> None:
    from . import export as X
    X.write(C.ROOT / "web" / "data")


STAGES = ["fetch", "clean", "network", "route", "calibrate", "fit", "assign", "validate", "export"]


def run(stage: str) -> None:
    if stage == "all":
        for s in STAGES:
            print(f"== {s}")
            run(s)
        return
    if stage == "fetch":
        from . import fetch
        fetch.all_()
    else:
        globals()[stage]()
