"""Compare inferred demand against MTA's own OD estimates (never used in fitting).

MTA publishes modeled OD only from 2021 on, so the reference is the average February
Wednesday of 2023: lower volume and less commute-heavy than February 2020, so we
compare *where each origin's riders go* (destination shares), not volumes.

If `subway calibrate` has run, the deterrence function was fit to this reference's
trip-length distribution (2 numbers), so the trip-minutes comparison is in-sample; the
pair-level destination shares are still an out-of-sample check.

Three candidates are scored, to show what each modeling step contributes:
  * null    - destination share proportional to destination size (daily entries),
              ignoring distance entirely;
  * prior   - the gravity prior (size x travel-cost deterrence), before seeing exits;
  * fitted  - the full model after fitting time-varying attractions to turnstile exits.

Metrics:
  * share overlap: per origin, sum_d min(p_d, m_d) between model and MTA destination
    shares (1 = identical, 0 = disjoint), averaged with origin-volume weights;
  * trip-time distribution: routed door-to-door minutes, weighted by each set of flows.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl

from . import config as C
from . import inversion as I
from .gtfs import Network


def od_matrix(path, cx_idx: dict[str, int], n: int) -> tuple[np.ndarray, float]:
    df = pl.read_csv(path, infer_schema_length=0).with_columns(pl.col("trips").cast(pl.Float64))
    M = np.zeros((n, n))
    dropped = 0.0
    for o, d, t in df.iter_rows():
        i, j = cx_idx.get(o), cx_idx.get(d)
        if i is None or j is None:
            dropped += t
            continue
        M[i, j] += t
    np.fill_diagonal(M, 0.0)
    return M, dropped / max(df["trips"].sum(), 1e-9)


def share_overlap(P: np.ndarray, M: np.ndarray) -> float:
    """Origin-weighted mean of sum_d min(p_od, m_od) over rows with flow in both."""
    rp, rm = P.sum(1, keepdims=True), M.sum(1, keepdims=True)
    ok = (rp[:, 0] > 0) & (rm[:, 0] > 0)
    p, m = P[ok] / rp[ok], M[ok] / rm[ok]
    per = np.minimum(p, m).sum(1)
    return float(np.average(per, weights=rm[ok, 0]))


def _tt_quantiles(W: np.ndarray, tt_min: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(tt_min) & (W > 0)
    w, x = W[ok], tt_min[ok]
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    q = {f"p{int(p * 100)}": float(x[order][np.searchsorted(cw, p)]) for p in (0.1, 0.5, 0.9)}
    q["mean"] = float(np.average(x, weights=w))
    return q


def report() -> dict:
    net = Network.load(C.PROCESSED / "network")
    n = net.n_complexes
    cx_idx = dict(zip(net.complexes["complex_id"], net.complexes["idx"]))
    q = np.load(C.PROCESSED / "demand.npz")["q"].astype(np.float64)       # [o, d, bin]
    z = np.load(C.PROCESSED / "travel_times.npz")
    tt, legs = z["tt"], z["legs"]
    obs = I.build_observations(net.complexes, pl.read_parquet(C.PROCESSED / "hourly_entries.parquet"),
                               pl.read_parquet(C.PROCESSED / "turnstile_obs.parquet"))
    lo = (C.VIEW_START_S - C.DEMAND_START_S) // I.BIN_S
    hi = (C.VIEW_END_S - C.DEMAND_START_S) // I.BIN_S
    day_entries = obs.E[:, lo:hi].sum(1)
    logA0 = np.repeat(np.log(np.maximum(day_entries, 1.0))[:, None], obs.E.shape[1], axis=1)
    shape, scale = I.deterrence_params()
    q0 = I._demand(obs.E, logA0, I.deterrence(tt, legs, shape, scale))
    null = np.outer(day_entries, day_entries)
    np.fill_diagonal(null, 0.0)

    ref_day, drop_day = od_matrix(C.RAW / "od_2023_day.csv", cx_idx, n)
    ref_am, drop_am = od_matrix(C.RAW / "od_2023_am.csv", cx_idx, n)
    am = slice(7 - C.DEMAND_START_S // 3600, 10 - C.DEMAND_START_S // 3600)   # entry hours 7-10
    am_entries = obs.E[:, am].sum(1)
    null_am = np.outer(am_entries, day_entries)
    np.fill_diagonal(null_am, 0.0)

    with warnings.catch_warnings():  # o == d is NaN for every hour
        warnings.simplefilter("ignore", RuntimeWarning)
        tt_day = np.nanmean(tt[:, lo:hi, :], axis=1) / 60.0               # [o, d] minutes
    out = {
        "reference": f"MTA OD estimate, {C.OD_REF_DOW}s in month {C.OD_REF_MONTH} of 2023",
        "reference_trips_per_day": float(ref_day.sum()),
        "model_trips_per_day": float(q[:, :, lo:hi].sum()),
        "reference_unmatched_fraction": {"day": drop_day, "am": drop_am},
        "share_overlap_day": {
            "null": share_overlap(null, ref_day),
            "prior": share_overlap(q0[:, :, lo:hi].sum(2), ref_day),
            "fitted": share_overlap(q[:, :, lo:hi].sum(2), ref_day),
        },
        "share_overlap_am_peak": {
            "null": share_overlap(null_am, ref_am),
            "prior": share_overlap(q0[:, :, am].sum(2), ref_am),
            "fitted": share_overlap(q[:, :, am].sum(2), ref_am),
        },
        "deterrence": {"shape": shape, "scale_min": scale, "calibrated": I.DETERRENCE_FILE.exists()},
        "trip_minutes": {
            "reference": _tt_quantiles(ref_day, tt_day),
            "prior": _tt_quantiles(q0[:, :, lo:hi].sum(2), tt_day),
            "fitted": _tt_quantiles(q[:, :, lo:hi].sum(2), tt_day),
        },
    }
    (C.PROCESSED / "validation.json").write_text(json.dumps(out, indent=2))
    return out


def print_report(r: dict) -> None:
    print(f"reference: {r['reference']}: {r['reference_trips_per_day']:,.0f} trips/day "
          f"(model, Feb 2020: {r['model_trips_per_day']:,.0f})")
    for k, label in (("share_overlap_day", "whole day"), ("share_overlap_am_peak", "AM peak 7-10")):
        v = r[k]
        print(f"destination-share overlap, {label:12s}: null {v['null']:.3f}  prior {v['prior']:.3f}  "
              f"fitted {v['fitted']:.3f}")
    det = r["deterrence"]
    if det["calibrated"]:
        print(f"note: deterrence (shape {det['shape']:.2f}, scale {det['scale_min']:.1f} min) was calibrated to the "
              f"reference trip-time distribution below, so that comparison is in-sample; the share overlaps are not")
    for k, v in r["trip_minutes"].items():
        print(f"trip minutes {k:9s}: p10 {v['p10']:5.1f}  median {v['p50']:5.1f}  p90 {v['p90']:5.1f}  "
              f"mean {v['mean']:5.1f}")
