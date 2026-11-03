"""Crosswalk turnstile remote units -> station complexes (-> GTFS parent stops).

Turnstile files identify places only by remote unit (e.g. R051), a 15-character
truncated station name ("JKSN HT-ROOSVLT") and a string of route letters ("EFMR7").
The MTA Subway Stations table maps GTFS stop ids to complex ids, and the hourly
ridership data is keyed by complex id. So we match each unit to a complex by
(fuzzy name similarity) + (route overlap), then verify the match independently:
summed turnstile entries per complex must agree with the hourly dataset's entries.
Residual mistakes are fixed by hand in OVERRIDES.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import polars as pl

# Hand fixes, unit -> complex_id, for matches the scorer gets wrong. Each entry was
# checked against the hourly-ridership entry totals (see `validate`).
OVERRIDES: dict[str, str] = {
    # "59 ST" NRW is 59 St on the 4th Av line in Brooklyn (complex 35), not 5 Av/59 St:
    # its daily entries (~11.6k) match complex 35's hourly total (11.3k), and without
    # it complex 35 has no turnstiles while complex 8 is ~1.8x over.
    "R212": "35",
}

_ABBREV = [
    (r"\bSTREETS?\b", "ST"), (r"\bSTS\b", "ST"), (r"\bAVENUE\b", "AV"), (r"\bAVE\b", "AV"),
    (r"\bBOULEVARD\b", "BLVD"), (r"\bPARKWAY\b", "PKWY"), (r"\bPARK\b", "PK"),
    (r"\bSQUARE\b", "SQ"), (r"\bPLAZA\b", "PLZ"), (r"\bHEIGHTS\b", "HTS"), (r"\bHT\b", "HTS"),
    (r"\bROAD\b", "RD"), (r"\bCENTRE\b", "CTR"), (r"\bCENTER\b", "CTR"), (r"\bHIGHWAY\b", "HWY"),
    (r"\bGRD\b", "GRAND"), (r"\bCNTRL\b", "CENTRAL"), (r"\bSTATION\b", "STA"),
    (r"\bEAST\b", "E"), (r"\bWEST\b", "W"), (r"\bNORTH\b", "N"), (r"\bSOUTH\b", "S"),
    (r"\bJKSN\b", "JACKSON"), (r"\bROOSVLT\b", "ROOSEVELT"), (r"\bISLND\b", "ISLAND"),
    (r"\bQNSBRIDGE\b", "QUEENSBRIDGE"), (r"\bBKLYN\b", "BROOKLYN"), (r"\bBK\b", "BROOKLYN"),
    (r"\bLEXINGTON\b", "LEX"), (r"\bMT\b", "MOUNT"), (r"\bFT\b", "FORT"),
]
_ROUTE_RE = re.compile(r"\(([^)]*)\)")


def norm(name: str) -> str:
    s = name.upper().replace("'", "").replace(".", " ")
    s = re.sub(r"[-/&,()]", " ", s)
    for pat, rep in _ABBREV:
        s = re.sub(pat, rep, s)
    s = re.sub(r"(\d+)(ST|ND|RD|TH)\b", r"\1", s)  # 42ND -> 42
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in s.split() if t]


def _token_match(t: str, u: str) -> float:
    if t == u:
        return 1.0
    if t.isdigit() or u.isdigit():
        return 0.0
    if len(t) >= 3 and len(u) >= 3 and (u.startswith(t) or t.startswith(u)):
        return 0.9  # truncation: "RACETR" ~ "RACETRACK"
    if min(len(t), len(u)) >= 4:
        r = SequenceMatcher(None, t, u).ratio()
        return r if r >= 0.7 else 0.0
    return 0.0


def _coverage(a: list[str], b: list[str]) -> float:
    return sum(max(_token_match(t, u) for u in b) for t in a) / len(a)


def name_similarity(turnstile_name: str, station_name: str) -> float:
    """Token-overlap similarity, weighted toward covering the turnstile's tokens.

    Tokens match on equality, prefix (the turnstile name is truncated to 15 chars) or
    high character similarity. Numbers must match exactly, and a turnstile number
    absent from the station name is disqualifying (59 St is not 86 St). The minor
    weight on covering the *station's* tokens breaks ties toward the tighter name
    ("23 ST" -> "23 St", not "Court Sq-23 St").
    """
    a, b = _tokens(norm(turnstile_name)), _tokens(norm(station_name))
    if not a or not b:
        return 0.0
    nums_a = {t for t in a if t.isdigit()}
    nums_b = {t for t in b if t.isdigit()}
    if nums_a and not (nums_a & nums_b):
        return 0.0
    score = 0.8 * _coverage(a, b) + 0.2 * _coverage(b, a)
    if nums_b and not nums_a:
        score *= 0.8
    return score


def complexes(stations: pl.DataFrame, hourly: pl.DataFrame) -> pl.DataFrame:
    """One row per complex: candidate names and the union of routes serving it."""
    names = hourly.group_by("station_complex_id").agg(pl.col("station_complex").first())
    st = stations.group_by("complex_id").agg(
        pl.col("stop_name").unique().alias("stop_names"),
        pl.col("daytime_routes").str.split(" ").list.explode(keep_nulls=False, empty_as_null=False).unique().alias("routes_now"),
        pl.col("division").unique().alias("divisions"),
    )
    cx = st.join(names, left_on="complex_id", right_on="station_complex_id", how="inner")
    rows = []
    for r in cx.iter_rows(named=True):
        hourly_name = r["station_complex"]
        bare = _ROUTE_RE.sub("", hourly_name)
        routes = set(r["routes_now"])
        for grp in _ROUTE_RE.findall(hourly_name):
            routes |= {x.strip() for x in grp.split(",")}
        names_ = set(r["stop_names"]) | {p.strip() for p in bare.split("/") if p.strip()}
        rows.append({
            "complex_id": r["complex_id"],
            "complex_name": hourly_name,
            "names": sorted(names_),
            "routes": sorted(x for x in routes if x),
            "divisions": r["divisions"],
        })
    return pl.DataFrame(rows)


def _route_set(line_name: str) -> set[str]:
    return set(line_name.replace(" ", ""))


def match(units: pl.DataFrame, cx: pl.DataFrame) -> pl.DataFrame:
    """Best complex per unit, with its score and the runner-up for review."""
    cxs = cx.to_dicts()
    out = []
    for u in units.iter_rows(named=True):
        routes_u = _route_set(u["line_name"])
        scored = []
        for c in cxs:
            ns = max(
                name_similarity(tn, cn) for tn in u["station_names"] for cn in c["names"]
            )
            routes_c = set(c["routes"])
            inter = len(routes_u & routes_c)
            # Mostly containment of the unit's routes; a little Jaccard so that a "5"
            # unit prefers Gun Hill Rd (5) over Gun Hill Rd (2,5).
            rs = 0.7 * inter / max(len(routes_u), 1) + 0.3 * inter / max(len(routes_u | routes_c), 1)
            div = 0.1 if u["division"] in c["divisions"] else 0.0
            scored.append((ns + rs + div, ns, rs, c["complex_id"], c["complex_name"]))
        scored.sort(reverse=True)
        best, second = scored[0], scored[1]
        cid = OVERRIDES.get(u["unit"], best[3])
        out.append({
            "unit": u["unit"], "station": u["station"], "line_name": u["line_name"],
            "complex_id": cid,
            "complex_name": next(c["complex_name"] for c in cxs if c["complex_id"] == cid),
            "score": best[0], "name_score": best[1], "route_score": best[2],
            "margin": best[0] - second[0], "runner_up": second[4],
            "override": u["unit"] in OVERRIDES,
        })
    return pl.DataFrame(out)


def validate(xw: pl.DataFrame, turnstile_day: pl.DataFrame, hourly_day: pl.DataFrame) -> pl.DataFrame:
    """Compare per-complex daily entries: turnstiles (via crosswalk) vs hourly dataset."""
    ts = (
        turnstile_day.join(xw.select(["unit", "complex_id"]), on="unit")
        .group_by("complex_id").agg(pl.col("entries").sum().alias("ts_entries"),
                                    pl.col("unit").unique().alias("units"))
    )
    hr = hourly_day.group_by("station_complex_id").agg(
        pl.col("entries").sum().alias("hr_entries"), pl.col("station_complex").first()
    )
    return (
        hr.join(ts, left_on="station_complex_id", right_on="complex_id", how="full", coalesce=True)
        .with_columns((pl.col("ts_entries") / pl.col("hr_entries")).alias("ratio"))
        .sort("station_complex_id")
    )
