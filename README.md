# subway-map

Riders aboard every NYC subway train over one day, inferred from turnstile counts and the
scheduled timetable, shown as an animated, zoomable map.

The modeled day is **Wednesday, February 5, 2020**: a high-ridership, non-holiday weekday
a month before COVID, covered by an archived copy of MTA's Fall 2019 static GTFS.

## Run it

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/subway all            # fetch -> clean -> network -> route -> calibrate -> fit -> assign -> validate -> export
python3 -m http.server 8791 --bind 127.0.0.1 -d web    # then open http://127.0.0.1:8791
```

Add `?perf` to the URL for a frame-rate readout. `window.__subway.bench()` times
synchronous frames, but only in a visible tab, because Chrome throttles GPU work in
background tabs.

`subway <stage>` runs one stage. Downloads are cached in `data/raw/`. A full run from
cached data takes about 15 minutes on a 6-core laptop, most of it in `calibrate`
(several full fits). Routing and assignment are numba kernels running in parallel.

On Intel Macs, newer `llvmlite` releases ship no wheels. numba 0.62 / llvmlite 0.45 install
from binaries.

## Pipeline

| stage | what it does | output |
|---|---|---|
| `fetch` | turnstile register reads (week around the date), hourly entries by station complex, MTA stations table, archived GTFS, MTA 2023 OD estimates (validation only) | `data/raw/` |
| `clean` | turnstile counters -> per-turnstile 4h counts (backwards counters flipped, resets rejected and imputed, odd read phases resampled); remote units matched to station complexes, checked against hourly entries | `data/interim/`, `data/processed/turnstile_obs.parquet` |
| `network` | GTFS -> parent-station connections for the day (previous and next service days included), complexes, transfers | `data/processed/network/` |
| `route` | round-based Connection Scan: travel time and legs for every origin x hour x destination | `travel_times.npz` |
| `calibrate` | fits the deterrence function's shape and scale so the fitted model's trip-length distribution matches MTA's OD estimates (see below) | `deterrence.json` |
| `fit` | infers origin-destination demand per hour (see below) | `demand.npz` |
| `assign` | re-routes the demand at 2-minute departure resolution and adds it onto each train's stop-to-stop segments | `loads.npy` |
| `validate` | compares destination shares with MTA's own OD estimates | `validation.json` |
| `export` | shared track geometry, per-trip timings and loads, station entries; figure data and quoted numbers for the "How this was made" page (`web/about.html`) | `web/data/day.json`, `web/data/writeup.json` |

## The model

**Unknown:** `q[o, d, b]`, the riders entering complex `o` during hour `b` bound for complex `d`.

```
q[o,d,b] = E[o,b] · A[d,b] f(c[o,b,d]) / Σ_d' A[d',b] f(c[o,b,d'])
```

- **Entries `E[o,b]`.** Hourly fare-system entries, matched exactly. Turnstile entries
  agree with them within about 1% per complex, which also validates the station matching.
- **Travel cost `c` and deterrence `f`.** `c` is the routed journey's generalized cost:
  door-to-door time plus 5 minutes per transfer. `f(c) = c^shape · exp(−c/scale)`.
  Turnstile counts barely constrain how far people travel, so `calibrate` fits shape
  and scale externally to MTA's trip-length distribution. The result is
  `exp(−c / 17.2 min)`: the shape lands on its lower bound of 0, so it's a pure
  exponential.
- **Attractions `A[d,b]`.** One per destination complex and hour; these are the free
  parameters, smoothed across hours.
- **Turnstile exits.** Fit as observations of the riders whose routed exit time falls in
  each 4h read interval, scaled by a per-turnstile-group recording rate `κ`. Exit
  registers miss emergency gates and exit-only stairs: recorded exits range from 0.27
  to 1.12 of entries across stations (5th to 95th percentile). So the exits mostly
  inform *when* people leave each station.
- **Time-of-day recording factor `m(t)`.** Smooth and system-wide. Fewer exits are
  recorded per rider at the peaks, when crowds push through gates (about 0.85 at
  8–10 AM against about 1.04 at midday).
- **Return-trip symmetry.** A soft constraint that each station's daily exits are
  close to its daily entries pins the totals.
- **Fitting.** MAP estimate with a log-normal likelihood (10% model error), solved with
  L-BFGS using analytic gradients.

Route choice: from each origin at each departure time, riders take the journey that
minimizes arrival time plus the transfer penalty, using up to 4 trains. Earliest
arrival alone produced three-seat rides to save a minute.

## Validation

MTA's own modeled OD estimates start in 2021. `validate` compares against February
Wednesdays of 2023 (3.7M trips/day, against 5.7M in 2020). Two numbers are taken from
that reference: `calibrate` fits the deterrence shape and scale to its trip-length
distribution, so the trip-minutes comparison is in-sample. Which pairs riders travel
between is never fit to it, so destination *shares* remain an out-of-sample check.

| destination-share overlap (1 = identical) | null (size only) | gravity prior | fitted |
|---|---|---|---|
| whole day | 0.603 | 0.684 | 0.743 |
| AM peak, entries 7–10 | 0.514 | 0.664 | 0.687 |

| routed door-to-door minutes | p10 | median | p90 | mean |
|---|---|---|---|---|
| MTA 2023 flows (calibration target) | 12.6 | 28.2 | 53.3 | 31.0 |
| fitted model | 11.5 | 28.7 | 52.9 | 30.9 |

Before calibration, with the a priori `c · exp(−c/16 min)`, trips ran about 15% long
(median 34.0 minutes) and the overlaps were 0.689 / 0.634. Fitting to turnstile exits
adds about 6 points of overlap over the prior. The rest of the gap mixes model error with
real change between 2020 and 2023.

## Known limitations

- **These are estimates, not counts.** Entries are measured. Destinations, timing
  within the hour, and route choice are modeled.
- **Route choice is all-or-nothing, with no crowding.** Everyone between a pair of
  stations at a given minute takes the same train, and trains have no capacity. Fast
  routes (for example A express, Q) likely carry too much and parallel ones (for
  example the Lexington Av express) too little.
- **The schedule is the plan, not what ran.** No GTFS-realtime: delays, reroutes and
  planned work are not reflected.
- **Trip lengths are calibrated to 2023 behavior.** That's the nearest MTA reference,
  but post-pandemic travel is less commute-heavy, so it may not match 2020 exactly.
  Delete `data/processed/deterrence.json` (and skip `calibrate`) to use the a priori
  default instead.
