import numpy as np
import polars as pl
import pytest

from subway import routing as R
from subway.gtfs import ACCESS_S, EGRESS_S, Network


def toy_network(trips, change_s=60):
    """trips: list of [(stop, time), ...]; each stop is its own complex."""
    n_stops = 1 + max(s for t in trips for s, _ in t)
    rows = []
    for ti, t in enumerate(trips):
        for k in range(len(t) - 1):
            rows.append((t[k][1], t[k + 1][1], t[k][0], t[k + 1][0], ti, k))
    rows.sort()
    dep, arr, frm, to, trip, seq = (np.array(c, dtype=np.int32) for c in zip(*rows))
    order = np.lexsort((seq, trip))
    counts = np.bincount(trip, minlength=len(trips))
    stops = pl.DataFrame({"idx": np.arange(n_stops, dtype=np.int32), "stop_id": [str(i) for i in range(n_stops)],
                          "stop_name": [f"S{i}" for i in range(n_stops)], "lat": 0.0, "lon": 0.0,
                          "complex_idx": np.arange(n_stops, dtype=np.int32)})
    return Network(
        stops=stops,
        complexes=pl.DataFrame({"idx": np.arange(n_stops, dtype=np.int32), "complex_id": [str(i) for i in range(n_stops)],
                                "name": [f"S{i}" for i in range(n_stops)], "lat": 0.0, "lon": 0.0}),
        trips=pl.DataFrame({"idx": np.arange(len(trips), dtype=np.int32), "trip_id": [f"t{i}" for i in range(len(trips))],
                            "route_id": "X", "direction_id": "0", "shape_id": None, "day_offset": 0}),
        c_dep=dep, c_arr=arr, c_from=frm, c_to=to, c_trip=trip, c_seq=seq,
        trip_conns=order.astype(np.int64), trip_first=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        change_s=np.full(n_stops, change_s, dtype=np.int32),
        xfer_ptr=np.zeros(n_stops + 1, dtype=np.int64), xfer_to=np.zeros(0, np.int32), xfer_s=np.zeros(0, np.int32),
        cx_ptr=np.arange(n_stops + 1, dtype=np.int64), cx_stops=np.arange(n_stops, dtype=np.int32),
    )


def _choose(net, o, tau, d):
    st = R._state(net.n_stops, net.trips.height)
    R._scan(o, tau, tau + 10_000, net.c_dep, net.c_arr, net.c_from, net.c_to, net.c_trip, net.change_s,
            net.xfer_ptr, net.xfer_to, net.xfer_s, net.cx_ptr, net.cx_stops, *st)
    p, k = R._choose(d, net.cx_ptr, net.cx_stops, st[2])
    return st[2][k, p], k


@pytest.mark.parametrize("direct_arr, expect_legs", [(700, 1), (1000, 2)])
def test_transfer_penalty_trades_off_against_time(direct_arr, expect_legs):
    # 0 -> 1 -> 2 with a transfer at 1 (arrives 600) vs. a one-seat ride 0 -> 2.
    t0 = 0
    trips = [[(0, t0 + 100), (1, 200)], [(1, 500), (2, 600)], [(0, 150), (2, direct_arr)]]
    net = toy_network(trips)
    arr, legs = _choose(net, 0, t0, 2)
    assert legs == expect_legs
    # one-seat at 700 beats 600 + 300 s penalty; at 1000 it doesn't
    assert arr == (direct_arr if expect_legs == 1 else 600)


def test_access_time_is_respected():
    net = toy_network([[(0, ACCESS_S - 1), (1, 500)], [(0, ACCESS_S + 10), (1, 900)]])
    arr, legs = _choose(net, 0, 0, 1)
    assert (arr, legs) == (900, 1)


def test_assign_loads_puts_demand_on_the_ridden_segments():
    # One trip 0 -> 1 -> 2 departing every 10 minutes; demand 0 -> 2 only.
    trips = [[(0, 600 * i + 300), (1, 600 * i + 400), (2, 600 * i + 500)] for i in range(12)]
    net = toy_network(trips)
    q = np.zeros((3, 3, 1))
    q[0, 2, 0] = 120.0
    loads, unrouted = R.assign_loads(net, q, t_start=0, bin_s=3600, sub_per_bin=60, horizon=7200)
    assert unrouted == 0
    by_seg = {}
    for c in range(len(loads)):
        by_seg[(int(net.c_from[c]), int(net.c_to[c]))] = by_seg.get((int(net.c_from[c]), int(net.c_to[c])), 0) + loads[c]
    assert by_seg[(0, 1)] == pytest.approx(120.0)
    assert by_seg[(1, 2)] == pytest.approx(120.0)


def test_travel_times_include_access_and_egress():
    net = toy_network([[(0, 600 * i + 300), (1, 600 * i + 900)] for i in range(10)])
    tt, legs = R.travel_times(net, 0, 3600, bin_s=3600, sub_per_bin=6, horizon=7200)
    # Departures every 600 s from t=300; passenger at tau waits for next >= tau + ACCESS.
    taus = np.arange(0, 3600, 600) + 300
    expected = []
    for tau in taus:
        dep = 300 + 600 * int(np.ceil((tau + ACCESS_S - 300) / 600))
        expected.append(dep + 600 + EGRESS_S - tau)
    assert tt[0, 0, 1] == pytest.approx(np.mean(expected))
    assert legs[0, 0, 1] == 1
