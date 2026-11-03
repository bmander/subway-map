"""Calibrate the gravity model's deterrence function to MTA's trip-length distribution.

Turnstile counts pin down *when* riders enter and leave each station, but hardly *how
far* they travel: with free per-station recording rates, almost any trip-length
distribution fits the exits about equally well. So the deterrence function
f(c) = c^shape exp(-c / scale) is calibrated externally, against the distribution of
routed door-to-door times implied by MTA's own OD estimates (the average February
Wednesday of 2023; the nearest available, see validate.py). Only these two numbers
are taken from that reference; which pairs the riders travel between still comes from
the model, so the destination-share comparison in `validate` stays informative.

The statistic matched is the fitted model's (not the prior's) p10/p25/p50/p75/p90 and
mean trip minutes. A full fit is slow and the fit shifts trip lengths relative to the
prior (exits pull riders toward the core), so we iterate:
    1. full fit at (shape, scale); measure shift = fitted stats - prior stats;
    2. choose (shape, scale) whose cheap *prior* stats hit target - shift;
    3. repeat (warm-started) until the fitted stats match.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
from scipy.optimize import minimize

from . import config as C
from . import inversion as I
from .gtfs import Network
from .validate import od_matrix

PROBS = (0.10, 0.25, 0.50, 0.75, 0.90)
LABELS = [f"p{int(p * 100)}" for p in PROBS] + ["mean"]
SHAPE_BOUNDS = (0.0, 3.0)
TOL_MIN = 0.3            # stop when RMS error over the six statistics is below this


class TripTimes:
    """Weighted trip-time statistics over complex pairs, with the pair order cached."""

    def __init__(self, tt_day_min: np.ndarray):
        x = tt_day_min.ravel()
        self.ok = np.isfinite(x)
        self.order = np.argsort(x[self.ok])
        self.x = x[self.ok][self.order]

    def stats(self, W: np.ndarray) -> np.ndarray:
        w = W.ravel()[self.ok][self.order]
        cw = np.cumsum(w) / w.sum()
        qs = [self.x[min(np.searchsorted(cw, p), len(self.x) - 1)] for p in PROBS]
        return np.array(qs + [float(np.dot(w, self.x) / w.sum())])


def _fmt(v: np.ndarray) -> str:
    return "  ".join(f"{k} {x:5.1f}" for k, x in zip(LABELS, v))


def run(max_rounds: int = 4, verbose: bool = True) -> dict:
    net = Network.load(C.PROCESSED / "network")
    z = np.load(C.PROCESSED / "travel_times.npz")
    tt, legs = z["tt"], z["legs"]
    obs = I.build_observations(net.complexes, pl.read_parquet(C.PROCESSED / "hourly_entries.parquet"),
                               pl.read_parquet(C.PROCESSED / "turnstile_obs.parquet"))
    lo = (C.VIEW_START_S - C.DEMAND_START_S) // I.BIN_S
    hi = (C.VIEW_END_S - C.DEMAND_START_S) // I.BIN_S
    with warnings.catch_warnings():  # o == d is NaN at every hour
        warnings.simplefilter("ignore", RuntimeWarning)
        tt_day = np.nanmean(tt[:, lo:hi, :], axis=1) / 60.0
    T = TripTimes(tt_day)
    cx_idx = dict(zip(net.complexes["complex_id"], net.complexes["idx"]))
    ref, _ = od_matrix(C.RAW / "od_2023_day.csv", cx_idx, net.n_complexes)
    target = T.stats(ref)

    day = obs.E[:, lo:hi].sum(1)
    logA0 = np.repeat(np.log(np.maximum(day, 1.0))[:, None], obs.E.shape[1], axis=1)

    def prior_stats(shape: float, scale: float) -> np.ndarray:
        q0 = I._demand(obs.E, logA0, I.deterrence(tt, legs, shape, scale))
        return T.stats(q0[:, :, lo:hi].sum(2))

    if verbose:
        print(f"target (MTA, routed minutes): {_fmt(target)}")
    shape, scale = I.GAMMA_SHAPE, I.GAMMA_SCALE_MIN
    x, best, rounds = None, None, []
    for r in range(max_rounds):
        res = I.fit(obs, tt, I.deterrence(tt, legs, shape, scale), x0=x,
                    max_iter=400 if x is None else 250, verbose=False)
        x = res.x
        fitted = T.stats(res.q[:, :, lo:hi].sum(2))
        rms = float(np.sqrt(np.mean((fitted - target) ** 2)))
        h = res.history[-1]
        rounds.append({"shape": shape, "scale_min": scale, "fitted": fitted.tolist(), "rms_min": rms,
                       "rel_err_exit": h["rel_err_exit"], "rel_err_sym": h["rel_err_sym"]})
        if verbose:
            print(f"round {r}: shape {shape:.3f} scale {scale:5.2f} -> fitted {_fmt(fitted)}  "
                  f"(RMS {rms:.2f} min; exits rel.err {h['rel_err_exit']:.3f})")
        if best is None or rms < best["rms_min"]:
            best = rounds[-1]
        if rms < TOL_MIN:
            break
        goal = target - (fitted - prior_stats(shape, scale))

        def loss(v):
            a = float(np.clip(v[0], *SHAPE_BOUNDS))
            return float(np.sum((prior_stats(a, float(np.exp(v[1]))) - goal) ** 2))

        sol = minimize(loss, [shape, np.log(scale)], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-3, "maxfev": 120})
        shape, scale = float(np.clip(sol.x[0], *SHAPE_BOUNDS)), float(np.exp(sol.x[1]))

    out = {
        "shape": best["shape"], "scale_min": best["scale_min"],
        "reference": f"MTA OD estimate, {C.OD_REF_DOW}s in month {C.OD_REF_MONTH} of 2023",
        "statistics": LABELS, "target": target.tolist(), "fitted": best["fitted"], "rms_min": best["rms_min"],
        "rounds": rounds,
    }
    I.DETERRENCE_FILE.write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"calibrated: f(c) = c^{out['shape']:.3f} exp(-c / {out['scale_min']:.2f} min); "
              f"RMS {out['rms_min']:.2f} min -> {I.DETERRENCE_FILE}")
    return out
