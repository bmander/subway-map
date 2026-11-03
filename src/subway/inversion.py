"""Infer time-dependent origin-destination demand q[o, d, b] from counts.

q[o, d, b] = passengers entering complex o during hour bin b, bound for complex d.

Model (a production-constrained gravity model with time-varying attractions):

    q[o, d, b] = E[o, b] * A[d, b] f(c[o, b, d]) / sum_d' A[d', b] f(c[o, b, d'])

  * E[o, b]: hourly entries (fare-system counts). Matched exactly by construction.
  * c: generalized cost of the routed journey (minutes, incl. transfer penalties);
    f(c) = c^shape exp(-c / scale) is a gamma deterrence function. Turnstile counts
    barely constrain (shape, scale); `subway calibrate` fits them to MTA's published
    trip-length distribution (see calibrate.py).
  * A[d, b]: attraction of destination d for trips starting in hour b -- the free
    parameters (one per complex-hour), smooth in b.

Observations:
  * Turnstile exits y_j, j = (complex d, read group g, 4h interval):
        mu_j = kappa_g * mbar_j * sum_{o,b} q[o, d, b] F_j(o, b, d)
    F_j = fraction of bin b's departures (uniform over the hour, shifted by routed
    travel time tt[o, b, d]) that exit inside the interval. kappa_g is a free
    per-group recording rate: exit registers miss emergency gates and exit-only
    stairs by station-specific amounts (5-95% range of exits/entries: 0.27-1.12), so
    exits inform *when* people leave each station more than how many. mbar_j is the
    interval mean of m(t), a smooth system-wide time-of-day recording factor:
    system-wide, recorded exits per arrival dip at the peaks (~0.72 AM vs ~0.82
    evening), and without m(t) the fit invents a "sink" destination to absorb it.
  * Return-trip symmetry: per complex, exits over the 24h service day equal its
    entries over that day (no kappa). Pins the daily destination totals.
  Both use a log-normal likelihood, (log mu - log y)^2 / (2 (cv^2 + 1/y)): the
  model's own relative error cv, not counting noise, sets how hard a big count is fit.

Penalties: smoothness on log A[d, .] and log m(.) across hours; small ridges on log A
toward log(daily entries of d) (fixes the per-hour gauge) and on log m (fixes its
scale against kappa).

Fit: MAP by L-BFGS over (log A, log kappa, log m) with analytic gradients (numba kernels).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit, prange
from scipy.optimize import minimize

from . import config as C
from .routing import TRANSFER_PENALTY_S

# Deterrence defaults (a priori). `subway calibrate` writes calibrated values to
# data/processed/deterrence.json, which `deterrence_params()` prefers when present.
GAMMA_SHAPE = 1.0
GAMMA_SCALE_MIN = 16.0
DETERRENCE_FILE = C.PROCESSED / "deterrence.json"
BIN_S = 3600
# Exits in the first hours after DEMAND_START include riders who entered before it;
# only fit intervals starting this long after the demand window opens.
EXIT_FIT_LEAD_S = 2 * 3600
CV_EXIT = 0.10       # relative model error for exit counts
CV_SYM = 0.15        # ... and for daily return-trip symmetry
SMOOTH = 5.0         # penalty on hour-to-hour changes in log attraction
SMOOTH_M = 50.0      # ... and in log recording factor m(t)
RIDGE = 1e-3
RIDGE_M = 1.0


@dataclass
class Observations:
    E: np.ndarray            # [n_cx, n_bins] entries
    obs_ptr: np.ndarray      # CSR over destination complexes
    obs_t0: np.ndarray       # seconds after service-day midnight
    obs_t1: np.ndarray
    obs_y: np.ndarray        # observed count
    obs_group: np.ndarray    # kappa group id; -1 = symmetry pseudo-observation (kappa = 1)
    n_groups: int
    groups: pl.DataFrame     # g -> (complex_idx, phase_s)

    @property
    def is_exit(self) -> np.ndarray:
        return self.obs_group >= 0


def build_observations(complexes: pl.DataFrame, hourly: pl.DataFrame, turn_obs: pl.DataFrame) -> Observations:
    n_cx = complexes.height
    n_bins = (C.DEMAND_END_S - C.DEMAND_START_S) // BIN_S
    cx_idx = dict(zip(complexes["complex_id"], complexes["idx"]))

    E = np.zeros((n_cx, n_bins))
    h = hourly.filter((pl.col("t0_s") >= C.DEMAND_START_S) & (pl.col("t0_s") < C.DEMAND_END_S))
    for cid, t0, e in h.select(["complex_id", "t0_s", "entries"]).iter_rows():
        E[cx_idx[cid], (int(t0) - C.DEMAND_START_S) // BIN_S] += e

    ex = turn_obs.filter(
        (pl.col("t0_s") >= C.DEMAND_START_S + EXIT_FIT_LEAD_S) & (pl.col("t1_s") <= C.DEMAND_END_S)
    ).with_columns(pl.col("complex_id").replace_strict(cx_idx, return_dtype=pl.Int32).alias("d"))
    groups = ex.select(["d", "phase_s"]).unique().sort(["d", "phase_s"]).with_row_index("g")
    ex = ex.join(groups, on=["d", "phase_s"])
    lo, hi = C.VIEW_START_S, C.VIEW_END_S
    sym_y = E[:, (lo - C.DEMAND_START_S) // BIN_S:(hi - C.DEMAND_START_S) // BIN_S].sum(axis=1)

    d_ = ex["d"].to_list() + list(range(n_cx))
    t0_ = ex["t0_s"].to_list() + [lo] * n_cx
    t1_ = ex["t1_s"].to_list() + [hi] * n_cx
    y_ = ex["exits"].to_list() + sym_y.tolist()
    g_ = ex["g"].to_list() + [-1] * n_cx
    order = np.argsort(np.array(d_), kind="stable")
    ptr = np.zeros(n_cx + 1, dtype=np.int64)
    np.add.at(ptr, np.array(d_) + 1, 1)
    return Observations(
        E=E, obs_ptr=np.cumsum(ptr),
        obs_t0=np.array(t0_, dtype=np.float64)[order], obs_t1=np.array(t1_, dtype=np.float64)[order],
        obs_y=np.array(y_, dtype=np.float64)[order], obs_group=np.array(g_, dtype=np.int64)[order],
        n_groups=groups.height,
        groups=groups.select(["g", pl.col("d").alias("complex_idx"), "phase_s"]),
    )


def generalized_cost_min(tt: np.ndarray, legs: np.ndarray) -> np.ndarray:
    return (tt + TRANSFER_PENALTY_S * (legs - 1)) / 60.0


def deterrence_params() -> tuple[float, float]:
    """(shape, scale_min): calibrated values if `subway calibrate` has run, else defaults."""
    if DETERRENCE_FILE.exists():
        d = json.loads(DETERRENCE_FILE.read_text())
        return float(d["shape"]), float(d["scale_min"])
    return GAMMA_SHAPE, GAMMA_SCALE_MIN


def deterrence(tt: np.ndarray, legs: np.ndarray, shape: float = GAMMA_SHAPE,
               scale_min: float = GAMMA_SCALE_MIN) -> np.ndarray:
    """f[o, b, d]; zero on the diagonal and where unreachable."""
    c = generalized_cost_min(tt, legs)
    with np.errstate(invalid="ignore"):
        f = np.where(np.isfinite(c), c**shape * np.exp(-c / scale_min), 0.0)
    n = f.shape[0]
    f[np.arange(n), :, np.arange(n)] = 0.0
    return np.ascontiguousarray(f)


@njit(parallel=True, cache=True)
def _demand(E, logA, f):
    """q[o, d, b] from attractions; production-constrained."""
    n_cx, n_bins = E.shape
    q = np.zeros((n_cx, n_cx, n_bins))
    A = np.exp(logA)
    for o in prange(n_cx):
        for b in range(n_bins):
            z = 0.0
            for d in range(n_cx):
                z += A[d, b] * f[o, b, d]
            if z <= 0.0:
                continue
            s = E[o, b] / z
            for d in range(n_cx):
                q[o, d, b] = s * A[d, b] * f[o, b, d]
    return q


@njit(cache=True)
def _overlap(a0, a1, b0, b1):
    lo = max(a0, b0)
    hi = min(a1, b1)
    return hi - lo if hi > lo else 0.0


@njit(parallel=True, cache=True)
def _forward(q, tt, t_start, bin_s, obs_ptr, obs_t0, obs_t1):
    """Unscaled expected count per observation: sum_{o,b} q[o,d,b] F_j(o,b,d)."""
    n_cx, _, n_bins = q.shape
    mu = np.zeros(len(obs_t0))
    for d in prange(n_cx):
        for o in range(n_cx):
            for b in range(n_bins):
                x = q[o, d, b]
                s = tt[o, b, d]
                if x <= 0.0 or not np.isfinite(s):
                    continue
                a0 = t_start + b * bin_s + s
                for j in range(obs_ptr[d], obs_ptr[d + 1]):
                    mu[j] += x * _overlap(a0, a0 + bin_s, obs_t0[j], obs_t1[j]) / bin_s
    return mu


@njit(parallel=True, cache=True)
def _grad_logA(q, E, tt, t_start, bin_s, obs_ptr, obs_t0, obs_t1, w):
    """d loglik / d logA[d, b], with w_j = d loglik/d mu_j * scale_j.

    g[o,d,b] = sum_j w_j F_j(o,b,d) is d loglik / d q[o,d,b]; then
    d/d logA[d,b] = sum_o q[o,d,b] (g[o,d,b] - gbar[o,b]),  gbar = sum_d q g / E.
    """
    n_cx, _, n_bins = q.shape
    g = np.zeros((n_cx, n_cx, n_bins))
    for d in prange(n_cx):
        for o in range(n_cx):
            for b in range(n_bins):
                s = tt[o, b, d]
                if q[o, d, b] <= 0.0 or not np.isfinite(s):
                    continue
                a0 = t_start + b * bin_s + s
                acc = 0.0
                for j in range(obs_ptr[d], obs_ptr[d + 1]):
                    acc += w[j] * _overlap(a0, a0 + bin_s, obs_t0[j], obs_t1[j]) / bin_s
                g[o, d, b] = acc
    gbar = np.zeros((n_cx, n_bins))
    for o in prange(n_cx):
        for b in range(n_bins):
            if E[o, b] <= 0.0:
                continue
            acc = 0.0
            for d in range(n_cx):
                acc += q[o, d, b] * g[o, d, b]
            gbar[o, b] = acc / E[o, b]
    out = np.zeros((n_cx, n_bins))
    for d in prange(n_cx):
        for b in range(n_bins):
            acc = 0.0
            for o in range(n_cx):
                acc += q[o, d, b] * (g[o, d, b] - gbar[o, b])
            out[d, b] = acc
    return out


def _lognormal(y, mu, cv):
    """Log-normal log-likelihood in mu (up to constants) and its derivative."""
    y = np.maximum(y, 0.5)
    mu = np.maximum(mu, 1e-9)
    var = cv**2 + 1.0 / y
    r = np.log(mu) - np.log(y)
    return -0.5 * r**2 / var, -r / (var * mu)


def _hour_overlap_matrix(t0, t1, n_bins, t_start, bin_s):
    """W[j, h] = fraction of observation j's interval lying in hour bin h."""
    W = np.zeros((len(t0), n_bins))
    for h in range(n_bins):
        lo, hi = t_start + h * bin_s, t_start + (h + 1) * bin_s
        W[:, h] = np.clip(np.minimum(t1, hi) - np.maximum(t0, lo), 0, None)
    return W / (t1 - t0)[:, None]


@dataclass
class FitResult:
    q: np.ndarray
    logA: np.ndarray
    kappa: np.ndarray
    m: np.ndarray            # system-wide recording factor per hour bin
    mu: np.ndarray           # fitted expected value per observation (kappa, m applied)
    history: list[dict]
    x: np.ndarray            # optimizer vector [log A, log kappa, log m] (for warm starts)


@dataclass
class Problem:
    """The MAP objective over x = [log A (n_cx*n_bins), log kappa (n_groups), log m (n_bins)]."""
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]]   # x -> (value, gradient)
    model: Callable[[np.ndarray], tuple]                          # x -> (q, mu_raw, scale, mbar)
    unpack: Callable[[np.ndarray], tuple]                         # x -> (logA, logk, logm)
    x0: np.ndarray
    history: list[dict]


def make_problem(obs: Observations, tt: np.ndarray, f: np.ndarray, smooth: float = SMOOTH,
                 t_start: float = float(C.DEMAND_START_S), sym_bins: tuple[int, int] | None = None,
                 verbose: bool = True) -> Problem:
    E = obs.E
    n_cx, n_bins = E.shape
    tt64 = np.ascontiguousarray(tt, dtype=np.float64)
    ex = obs.is_exit
    gid = obs.obs_group[ex]
    cv = np.where(ex, CV_EXIT, CV_SYM)
    W = _hour_overlap_matrix(obs.obs_t0[ex], obs.obs_t1[ex], n_bins, t_start, BIN_S)
    lo, hi = sym_bins or ((C.VIEW_START_S - C.DEMAND_START_S) // BIN_S, (C.VIEW_END_S - C.DEMAND_START_S) // BIN_S)
    day = E[:, lo:hi].sum(axis=1)
    logA0 = np.repeat(np.log(np.maximum(day, 1.0))[:, None], n_bins, axis=1)
    n_a, n_g = n_cx * n_bins, obs.n_groups
    hist: list[dict] = []

    def unpack(x):
        return x[:n_a].reshape(n_cx, n_bins), x[n_a:n_a + n_g], x[n_a + n_g:]

    def model(x):
        logA, logk, logm = unpack(x)
        q = _demand(E, logA, f)
        mu_raw = _forward(q, tt64, t_start, BIN_S, obs.obs_ptr, obs.obs_t0, obs.obs_t1)
        mbar = W @ np.exp(logm)
        scale = np.ones_like(mu_raw)
        scale[ex] = np.exp(logk[gid]) * mbar
        return q, mu_raw, scale, mbar

    def smooth_pen(v, w):
        dv = np.diff(v, axis=-1)
        g = np.zeros_like(v)
        g[..., 1:] += 2 * w * dv
        g[..., :-1] -= 2 * w * dv
        return w * np.sum(dv**2), g

    def objective(x):
        logA, logk, logm = unpack(x)
        q, mu_raw, scale, mbar = model(x)
        mu = scale * mu_raw
        ll, dll = _lognormal(obs.obs_y, mu, cv)
        gA = _grad_logA(q, E, tt64, t_start, BIN_S, obs.obs_ptr, obs.obs_t0, obs.obs_t1, dll * scale)
        gk = np.bincount(gid, weights=(dll * mu)[ex], minlength=n_g)
        # d mu_j / d logm[h] = mu_j * W[j, h] m[h] / mbar_j
        gm = ((dll * mu)[ex] / mbar) @ W * np.exp(logm)
        pA, gpA = smooth_pen(logA, smooth)
        pm, gpm = smooth_pen(logm, SMOOTH_M)
        pA += RIDGE * np.sum((logA - logA0) ** 2)
        gpA += 2 * RIDGE * (logA - logA0)
        pm += RIDGE_M * np.sum(logm**2)
        gpm += 2 * RIDGE_M * logm
        val = -ll.sum() + pA + pm
        grad = np.concatenate([(-gA + gpA).ravel(), -gk, -gm + gpm])
        rel = lambda m_: float(np.sqrt(np.sum((mu[m_] - obs.obs_y[m_]) ** 2) / np.sum(obs.obs_y[m_] ** 2)))  # noqa: E731
        hist.append({"obj": float(val), "nll_exit": float(-ll[ex].sum()), "nll_sym": float(-ll[~ex].sum()),
                     "penalty": float(pA + pm), "rel_err_exit": rel(ex), "rel_err_sym": rel(~ex)})
        if verbose and len(hist) % 25 == 1:
            h = hist[-1]
            print(f"  eval {len(hist):4d}: obj {h['obj']:10.1f}  nll exit {h['nll_exit']:9.1f} sym {h['nll_sym']:8.1f}  "
                  f"rel.err exit {h['rel_err_exit']:.4f} sym {h['rel_err_sym']:.4f}")
        return val, grad

    # kappa start: each group's exits over the day / its complex's daily entries
    gy = np.bincount(gid, weights=obs.obs_y[ex], minlength=n_g)
    gd = obs.groups["complex_idx"].to_numpy()
    k0 = np.clip(gy / np.maximum(day[gd], 1.0), 0.02, 3.0)
    x0 = np.concatenate([logA0.ravel(), np.log(k0), np.zeros(n_bins)])
    return Problem(objective=objective, model=model, unpack=unpack, x0=x0, history=hist)


def fit(obs: Observations, tt: np.ndarray, f: np.ndarray, smooth: float = SMOOTH,
        max_iter: int = 400, x0: np.ndarray | None = None, verbose: bool = True) -> FitResult:
    """MAP fit. `x0` (e.g. a previous FitResult.x) warm-starts the optimizer."""
    prob = make_problem(obs, tt, f, smooth=smooth, verbose=verbose)
    res = minimize(prob.objective, prob.x0 if x0 is None else x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxcor": 30, "ftol": 1e-12, "gtol": 1e-6})
    if verbose:
        print(f"  L-BFGS: {res.message} after {res.nit} iterations / {res.nfev} evaluations")
    logA, logk, logm = prob.unpack(res.x)
    q, mu_raw, scale, _ = prob.model(res.x)
    return FitResult(q=q, logA=logA, kappa=np.exp(logk), m=np.exp(logm), mu=scale * mu_raw,
                     history=prob.history, x=res.x)
