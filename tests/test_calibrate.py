import numpy as np
import pytest

from subway.calibrate import LABELS, TripTimes


def test_weighted_trip_time_stats():
    tt = np.array([[np.nan, 10.0], [30.0, np.nan], [20.0, 40.0]])  # NaNs (o == d) are ignored
    T = TripTimes(tt)
    s = dict(zip(LABELS, T.stats(np.array([[5.0, 1.0], [1.0, 9.0], [1.0, 1.0]]))))
    assert s["p50"] == 20.0
    assert s["mean"] == pytest.approx(25.0)
    s = dict(zip(LABELS, T.stats(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]]))))
    assert s["p10"] == 30.0 and s["p90"] == 40.0
    assert s["mean"] == pytest.approx(37.5)
