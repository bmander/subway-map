import datetime as dt

import polars as pl
import pytest

from subway import turnstile as T


def _raw(rows):
    """rows: (scp, datetime, entries, exits) for one control area/unit."""
    return pl.DataFrame(
        {
            "c_a": ["A001"] * len(rows), "unit": ["R001"] * len(rows), "scp": [r[0] for r in rows],
            "station": ["TEST ST"] * len(rows), "line_name": ["1"] * len(rows), "division": ["IRT"] * len(rows),
            "ts": [r[1] for r in rows], "description": ["REGULAR"] * len(rows),
            "entries": [r[2] for r in rows], "exits": [r[3] for r in rows],
        }
    )


def _reads(scp, start, n, step_h, e0, de, x0=0, dx=0):
    return [(scp, start + dt.timedelta(hours=step_h * i), e0 + de * i, x0 + dx * i) for i in range(n)]


def test_backwards_counter_is_flipped():
    iv = T.intervals(_raw(_reads("00-00-00", dt.datetime(2020, 2, 3, 3), 6, 4, 1_000_000, -10, 500, 5)))
    assert iv["d_entries"].to_list() == [10.0] * 5
    assert iv["d_exits"].to_list() == [5.0] * 5
    assert iv["ok_entries"].all()


def test_counter_reset_is_rejected_and_imputed_from_same_hour_other_days():
    # Three weekdays of steady 100 entries per 4h, then a reset to ~0 on the last day.
    rows = _reads("00-00-00", dt.datetime(2020, 2, 3, 3), 18, 4, 5_000, 100)
    scp, ts, e, x = rows[16]
    rows[16] = (scp, ts, 7, x)                      # reset at the 7th read of day 3
    rows[17] = (rows[17][0], rows[17][1], 107, rows[17][3])
    iv = T.intervals(_raw(rows))
    bad = iv.filter(~pl.col("ok_entries"))
    assert bad.height == 1
    assert bad["d_entries"][0] == pytest.approx(100.0)   # median rate at that hour x 4h
    assert iv.filter(pl.col("ok_entries"))["d_entries"].max() == 100.0


def test_phase_at_late_hours_does_not_overflow():
    # Regression: dt.hour() is Int8; 23 * 3600 used to overflow, garbling read phases.
    rows = _reads("00-00-00", dt.datetime(2020, 2, 5, 3, 22), 7, 4, 0, 50)   # 03:22, 07:22, ..., 03:22
    ph = T.phases(_raw(rows))
    assert ph["phase_s"].to_list() == [3 * 3600 + 22 * 60]


def test_resample_on_grid_reads_is_exact_and_conserves_counts():
    rows = _reads("00-00-00", dt.datetime(2020, 2, 5, 3), 7, 4, 0, 0)
    vals = [0, 10, 210, 400, 520, 800, 830]
    rows = [(s, t, v, 2 * v) for (s, t, _, _), v in zip(rows, vals)]
    raw = _raw(rows)
    g = T.resample(T.intervals(raw), T.phases(raw), dt.datetime(2020, 2, 5), dt.datetime(2020, 2, 6, 4))
    assert g["entries"].to_list() == pytest.approx([10, 200, 190, 120, 280, 30])
    assert g["exits"].sum() == pytest.approx(2 * 830)
    assert g["t0"][0] == dt.datetime(2020, 2, 5, 3)
