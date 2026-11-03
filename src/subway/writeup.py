"""Figure data and key numbers for the "How this was made" page (web/about.html).

Everything here is computed from pipeline outputs, so the article's figures and the
numbers quoted in its text track the model. Runs as part of `subway export`.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import polars as pl

from . import config as C

HERO_TIME_S = int(8.75 * 3600)          # 08:45, the morning peak
SHOWCASE_TURNSTILE = ("R143", "R032", "02-03-03")   # a busy, cleanly read Times Sq turnstile
MAREY_ROUTES = ("7", "7X")
MAREY_WINDOW_S = (int(7.75 * 3600), int(8.75 * 3600))   # one hour of the morning rush
# The first structured fit (before the time-of-day recording factor existed) sent
# riders to 14 St-Union Sq by entry hour like this. Recorded from that run's output;
# kept here because it is the article's cautionary tale, not something we recompute.
FIRST_ATTEMPT_UNION_SQ = [1724, 695, 446, 533, 1709, 8616, 41304, 292443, 384108, 105146, 43945, 42105,
                          52319, 77160, 95442, 94811, 145603, 262851, 230428, 127789, 74451, 44803, 20334,
                          6536, 393, 62, 24, 19, 41, 142]   # bins 00:00..30:00 of service day


def _km(lon1, lat1, lon2, lat2) -> float:
    k = math.cos(math.radians(40.7))
    return math.hypot((lon2 - lon1) * k, lat2 - lat1) * 111.32


def _seg_point(flat: list[float], f: float) -> tuple[float, float]:
    """Point at fraction f of a flat [lon, lat, ...] polyline's length."""
    pts = np.asarray(flat).reshape(-1, 2)
    k = math.cos(math.radians(40.7))
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]) * k, np.diff(pts[:, 1])))])
    if d[-1] == 0:
        return float(pts[0, 0]), float(pts[0, 1])
    x = f * d[-1]
    i = min(int(np.searchsorted(d, x, side="right")) - 1, len(pts) - 2)
    u = (x - d[i]) / (d[i + 1] - d[i]) if d[i + 1] > d[i] else 0.0
    p = pts[i] + u * (pts[i + 1] - pts[i])
    return round(float(p[0]), 5), round(float(p[1]), 5)


def hero(day: dict, t: int) -> dict:
    """Every train's position and load at time t, plus track geometry with line colors."""
    trains = []
    seg_route = {}
    for tr in day["trips"]:
        for g in tr["g"]:
            seg_route.setdefault(g, tr["r"])
        if not (tr["d"][0] <= t <= tr["a"][-1]):
            continue
        k = max(i for i, dep in enumerate(tr["d"]) if dep <= t)
        if t < tr["a"][k]:
            f = (t - tr["d"][k]) / max(tr["a"][k] - tr["d"][k], 1)
            lon, lat = _seg_point(day["segs"][tr["g"][k]], f)
        else:  # dwelling at the next station
            lon, lat = day["stations"][tr["s"][k + 1]]["p"]
        trains.append([lon, lat, tr["r"], tr["l"][k]])
    trains.sort(key=lambda x: -x[3])
    tracks = [[seg_route[i], day["segs"][i]] for i in range(len(day["segs"])) if i in seg_route]
    return {"t": t, "trains": trains, "tracks": tracks, "aboard": int(sum(x[3] for x in trains))}


def turnstile_example() -> dict:
    ca, unit, scp = SHOWCASE_TURNSTILE
    raw = pl.read_csv(next(C.RAW.glob("turnstiles_*.csv")), infer_schema_length=0)
    mid = dt.datetime.combine(C.SERVICE_DATE, dt.time())
    r = (
        raw.filter((pl.col("c_a") == ca) & (pl.col("unit") == unit) & (pl.col("scp") == scp))
        .with_columns((pl.col("date").str.slice(0, 10) + " " + pl.col("time")).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts"))
        .filter((pl.col("ts") >= mid + dt.timedelta(seconds=C.VIEW_START_S))
                & (pl.col("ts") <= mid + dt.timedelta(seconds=C.VIEW_END_S)))
        .sort("ts")
    )
    return {
        "station": r["station"][0], "id": f"{ca} / {unit} / {scp}",
        "reads": [[int((t - mid).total_seconds()), int(e), int(x)]
                  for t, e, x in r.select(["ts", "entries", "exits"]).iter_rows()],
    }


def exit_ratios() -> list:
    """Per complex: recorded exits / entries over the 24h service day."""
    obs = pl.read_parquet(C.PROCESSED / "turnstile_obs.parquet")
    w = (obs.with_columns(((pl.col("phase_s") - 3600) % 14400 + 3600).alias("s0"))
         .filter((pl.col("t0_s") >= pl.col("s0")) & (pl.col("t1_s") <= pl.col("s0") + 86400)))
    x = w.group_by("complex_id").agg(pl.col("exits").sum().alias("x"))
    e = (pl.read_parquet(C.PROCESSED / "hourly_entries.parquet")
         .filter((pl.col("t0_s") >= C.VIEW_START_S) & (pl.col("t0_s") < C.VIEW_END_S))
         .group_by("complex_id").agg(pl.col("entries").sum().alias("e"), pl.col("station_complex").first().alias("name")))
    j = x.join(e, on="complex_id").filter(pl.col("e") > 0).with_columns((pl.col("x") / pl.col("e")).alias("ratio"))
    return [[n, int(ee), round(float(rr), 3)] for n, ee, rr in j.select(["name", "e", "ratio"]).sort("ratio").iter_rows()]


def union_square(net_complexes: pl.DataFrame, q: np.ndarray, m: np.ndarray) -> dict:
    us = int(net_complexes.filter(pl.col("name").str.starts_with("14 St-Union Sq"))["idx"][0])
    lo, hi = C.VIEW_START_S // 3600, C.VIEW_END_S // 3600
    final = q[:, us, :].sum(axis=0)
    return {
        "hours": list(range(lo, hi)),
        "first_attempt": FIRST_ATTEMPT_UNION_SQ[lo:hi],
        "final": [int(round(v)) for v in final[lo:hi]],
        "system_entries": [int(round(v)) for v in q.sum(axis=(0, 1))[lo:hi]],
        "first_attempt_day": int(sum(FIRST_ATTEMPT_UNION_SQ[lo:hi])),
        "final_day": int(round(final[lo:hi].sum())),
        "recording_factor": [round(float(v), 3) for v in m[lo:hi]],
    }


def marey(day: dict) -> dict:
    """Time-distance diagram for the 7 line: every 7/7X trip in the window, with loads."""
    rid = {r["id"]: i for i, r in enumerate(day["routes"])}
    want = {rid[r] for r in MAREY_ROUTES if r in rid}
    local = [tr for tr in day["trips"] if tr["r"] == rid["7"]]
    spine = max(local, key=lambda tr: len(tr["s"]))["s"]
    st = day["stations"]
    y, acc = {spine[0]: 0.0}, 0.0
    for a, b in zip(spine, spine[1:]):
        acc += _km(*st[a]["p"], *st[b]["p"])
        y[b] = round(acc, 3)
    lo, hi = MAREY_WINDOW_S
    segs = []
    for tr in day["trips"]:
        if tr["r"] not in want or tr["a"][-1] < lo or tr["d"][0] > hi:
            continue
        if not all(s in y for s in tr["s"]):
            continue
        express = int(day["routes"][tr["r"]]["id"].endswith("X"))
        to_manhattan = int(y[tr["s"][-1]] < y[tr["s"][0]])
        for k in range(len(tr["d"])):
            if k and tr["d"][k] > tr["a"][k - 1]:   # waiting at a station: draw it, or the line looks broken
                ys_ = y[tr["s"][k]]
                segs.append([tr["a"][k - 1], ys_, tr["d"][k], ys_, tr["l"][k], express, to_manhattan])
            segs.append([tr["d"][k], y[tr["s"][k]], tr["a"][k], y[tr["s"][k + 1]], tr["l"][k], express, to_manhattan])
    stations = [[st[s]["n"], y[s]] for s in spine]
    # Line load profile: riders carried past each stretch between adjacent stations, per
    # direction, by trains departing within the window (an express hop counts for every
    # stretch it covers). This is the classic "riders per hour past a point" curve.
    ys = sorted(y.values())
    flow = {"toward_manhattan": [0.0] * (len(ys) - 1), "toward_flushing": [0.0] * (len(ys) - 1)}
    in_window = [g for g in segs if lo <= g[0] < hi and g[1] != g[3]]   # moving hops only
    for t0, y0, t1, y1, load, _, toward_m in in_window:
        key = "toward_manhattan" if toward_m else "toward_flushing"
        a, b = sorted((y0, y1))
        for i in range(len(ys) - 1):
            if ys[i] >= a - 1e-9 and ys[i + 1] <= b + 1e-9:
                flow[key][i] += load
    inbound = [g for g in in_window if g[6]]
    outbound = [g for g in in_window if not g[6]]
    top = max(in_window, key=lambda g: g[4])
    name_of = {y[s]: st[s]["n"] for s in spine}
    return {"window": [lo, hi], "stations": stations, "segs": segs, "color": day["routes"][rid["7"]]["color"],
            "profile": {"y": ys, **{k: [round(v) for v in vs] for k, vs in flow.items()}},
            "mean_inbound": round(float(np.mean([g[4] for g in inbound]))),
            "mean_outbound": round(float(np.mean([g[4] for g in outbound]))),
            "fullest": {"t": top[0], "load": top[4], "express": top[5],
                        "from": name_of[top[1]], "to": name_of[top[3]]}}


def facts(day: dict, demand: dict, hist: pl.DataFrame, validation: dict, deterrence: dict) -> dict:
    aboard = np.array(day["aboard"])
    peak = int(np.argmax(aboard))
    xw_val = pl.read_parquet(C.INTERIM / "crosswalk_validation.parquet")
    xw = pl.read_parquet(C.INTERIM / "crosswalk.parquet")
    entries_day = sum(sum(c["e"]) for c in day["complexes"])
    k = demand["kappa"]
    t_peak = day["t0"] + 60 * peak
    r = xw_val["ratio"].to_numpy()
    return {
        "date": "Wednesday, February 5, 2020",
        "n_hops": int(sum(len(tr["g"]) for tr in day["trips"])),
        "xw_max_dev": round(float(np.abs(r - 1).max()), 3),
        "xw_within5": int((np.abs(r - 1) <= 0.05).sum()),
        "kappa_min": round(float(k.min()), 2), "kappa_max": round(float(k.max()), 2),
        "kappa_over1": int((k > 1).sum()), "n_groups": int(len(k)),
        "entries_day": int(entries_day),
        "peak_aboard": int(aboard[peak]),
        "peak_time": f"{t_peak // 3600 % 24}:{t_peak % 3600 // 60:02d}",
        "n_trips": len(day["trips"]),
        "n_units": xw.height, "n_complexes": int(xw["complex_id"].n_unique()),
        "xw_ratio_p5": round(float(xw_val["ratio"].quantile(0.05)), 3),
        "xw_ratio_p95": round(float(xw_val["ratio"].quantile(0.95)), 3),
        "kappa_p5": round(float(np.quantile(k, 0.05)), 2), "kappa_p95": round(float(np.quantile(k, 0.95)), 2),
        "exit_rel_err": round(float(hist["rel_err_exit"][-1]), 3),
        "overlap_day": validation["share_overlap_day"], "overlap_am": validation["share_overlap_am_peak"],
        "trip_minutes": validation["trip_minutes"],
        "deterrence_scale_min": round(deterrence["scale_min"], 1) if deterrence else None,
        "deterrence_shape": round(deterrence["shape"], 2) if deterrence else None,
        "calibration": {"labels": deterrence["statistics"], "target": deterrence["target"],
                        "before": deterrence["rounds"][0]["fitted"], "after": deterrence["fitted"]} if deterrence else None,
    }


def write(out_dir) -> None:
    from .gtfs import Network
    from .inversion import DETERRENCE_FILE

    day = json.loads((out_dir / "day.json").read_text())
    net = Network.load(C.PROCESSED / "network")
    dm = np.load(C.PROCESSED / "demand.npz")
    q, m = dm["q"].astype(np.float64), dm["m"]
    validation = json.loads((C.PROCESSED / "validation.json").read_text())
    deterrence = json.loads(DETERRENCE_FILE.read_text()) if DETERRENCE_FILE.exists() else None
    hist = pl.read_parquet(C.PROCESSED / "fit_history.parquet")
    from . import inversion as I
    obs = I.build_observations(net.complexes, pl.read_parquet(C.PROCESSED / "hourly_entries.parquet"),
                               pl.read_parquet(C.PROCESSED / "turnstile_obs.parquet"))
    f = facts(day, {"kappa": dm["kappa"]}, hist, validation, deterrence)
    f.update({"n_exit_obs": int(obs.is_exit.sum()), "n_sym_obs": int((~obs.is_exit).sum()),
              "n_pulls": int(obs.E.size), "n_bins": int(obs.E.shape[1]),
              "n_unknowns": int(obs.E.shape[0] * (obs.E.shape[0] - 1) * obs.E.shape[1])})
    ratios = exit_ratios()
    f["n_exits_gt_entries"] = sum(1 for x in ratios if x[2] > 1)
    data = {
        "facts": f,
        "routes": [{"id": r["id"], "name": r["name"], "color": r["color"]} for r in day["routes"]],
        "aboard": day["aboard"], "t0": day["t0"],
        "hero": hero(day, HERO_TIME_S),
        "turnstile": turnstile_example(),
        "exit_ratios": ratios,
        "union_square": union_square(net.complexes, q, m),
        "marey": marey(day),
    }
    path = out_dir / "writeup.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"writeup figures: {path.stat().st_size / 1e6:.2f} MB -> {path}")
