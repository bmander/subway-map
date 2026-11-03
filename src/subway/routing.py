"""Route choice on the scheduled network: a round-based Connection Scan Algorithm.

A passenger who passes fare control at complex `o` at time `tau` can board at any of
o's stations after ACCESS_S. Changing trains at a station costs that station's change
time; walking to another station costs the transfers.txt time.

Pure earliest-arrival routing picks absurd three-seat rides to save a minute, so the
scan keeps labels per *number of legs* k = 1..MAX_LEGS (as RAPTOR rounds do): for
every stop, the earliest arrival using exactly k trains. At each destination the
passenger picks the k minimizing  arrival + TRANSFER_PENALTY_S * (k - 1), i.e. a
generalized cost with a per-transfer penalty. Within a round, a trip is boarded at the
first stop where it is reachable; with exact rounds this cannot create backriding
(a backriding journey has more legs than the direct one at the same arrival time).

Two passes reuse the same scan:
  * `travel_times`: per (origin, hour bin, destination), mean door-to-door time and
    mean legs of the chosen journeys (gravity prior and exit-timing operator);
  * `assign_loads`: spread each hour's demand q[o, d, bin] evenly over sub-departure
    times within the hour and add it onto every connection of every chosen leg ->
    passengers on board every trip segment.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from .gtfs import ACCESS_S, EGRESS_S, Network

INF = np.int32(2**31 - 1)
MAX_LEGS = 4                 # at most 3 transfers
TRANSFER_PENALTY_S = 300     # perceived cost of one transfer, on top of its real time


@njit(cache=True)
def _scan(o, tau, t_end, c_dep, c_arr, c_from, c_to, c_trip, change_s, xfer_ptr, xfer_to, xfer_s,
          cx_ptr, cx_stops, ready, ready_from, arr_t, arr_conn, arr_board, board):
    """One scan from complex `o` at `tau`. State arrays are [MAX_LEGS + 1, n] (row k =
    exactly k legs ridden; row 0 = at the origin). Fills them in place."""
    K = MAX_LEGS
    ready[:, :] = INF
    arr_t[:, :] = INF
    board[:, :] = -1
    t0 = tau + ACCESS_S
    for k0 in range(cx_ptr[o], cx_ptr[o + 1]):
        p = cx_stops[k0]
        ready[0, p] = t0
        ready_from[0, p] = -1
    for k0 in range(cx_ptr[o], cx_ptr[o + 1]):
        p = cx_stops[k0]
        for j in range(xfer_ptr[p], xfer_ptr[p + 1]):
            w = xfer_to[j]
            if t0 + xfer_s[j] < ready[0, w]:
                ready[0, w] = t0 + xfer_s[j]
                ready_from[0, w] = -1
    c = np.searchsorted(c_dep, t0)
    n = len(c_dep)
    while c < n:
        d = c_dep[c]
        if d > t_end:
            break
        t = c_trip[c]
        f = c_from[c]
        s = c_to[c]
        a = c_arr[c]
        for k in range(1, K + 1):
            if board[k, t] < 0:
                if ready[k - 1, f] <= d:
                    board[k, t] = c
                else:
                    continue
            if a < arr_t[k, s]:
                arr_t[k, s] = a
                arr_conn[k, s] = c
                arr_board[k, s] = board[k, t]
                r = a + change_s[s]
                if r < ready[k, s]:
                    ready[k, s] = r
                    ready_from[k, s] = s
                for j in range(xfer_ptr[s], xfer_ptr[s + 1]):
                    w = xfer_to[j]
                    r = a + xfer_s[j]
                    if r < ready[k, w]:
                        ready[k, w] = r
                        ready_from[k, w] = s
        c += 1


@njit(cache=True)
def _choose(dcx, cx_ptr, cx_stops, arr_t):
    """Best (stop, legs) at destination complex by generalized cost; (-1, 0) if none."""
    best_cost = np.int64(INF) * 2
    bs, bk = -1, 0
    for k in range(1, MAX_LEGS + 1):
        for j in range(cx_ptr[dcx], cx_ptr[dcx + 1]):
            p = cx_stops[j]
            a = arr_t[k, p]
            if a == INF:
                continue
            cost = np.int64(a) + TRANSFER_PENALTY_S * (k - 1)
            if cost < best_cost:
                best_cost, bs, bk = cost, p, k
    return bs, bk


@njit(cache=True)
def _state(n_stops, n_trips):
    K1 = MAX_LEGS + 1
    return (np.empty((K1, n_stops), np.int32), np.empty((K1, n_stops), np.int32),
            np.empty((K1, n_stops), np.int32), np.empty((K1, n_stops), np.int32),
            np.empty((K1, n_stops), np.int32), np.empty((K1, n_trips), np.int32))


@njit(parallel=True, cache=True)
def _travel_times(taus, bin_of, n_bins, horizon, c_dep, c_arr, c_from, c_to, c_trip, change_s,
                  xfer_ptr, xfer_to, xfer_s, cx_ptr, cx_stops, n_trips):
    n_cx = len(cx_ptr) - 1
    n_stops = len(change_s)
    tt_sum = np.zeros((n_cx, n_bins, n_cx), dtype=np.float64)
    tt_cnt = np.zeros((n_cx, n_bins, n_cx), dtype=np.int32)
    legs_sum = np.zeros((n_cx, n_bins, n_cx), dtype=np.float32)
    for o in prange(n_cx):
        ready, ready_from, arr_t, arr_conn, arr_board, board = _state(n_stops, n_trips)
        for j in range(len(taus)):
            tau = taus[j]
            b = bin_of[j]
            _scan(o, tau, tau + horizon, c_dep, c_arr, c_from, c_to, c_trip, change_s, xfer_ptr,
                  xfer_to, xfer_s, cx_ptr, cx_stops, ready, ready_from, arr_t, arr_conn, arr_board, board)
            for dcx in range(n_cx):
                if dcx == o:
                    continue
                p, k = _choose(dcx, cx_ptr, cx_stops, arr_t)
                if p < 0:
                    continue
                tt_sum[o, b, dcx] += arr_t[k, p] + EGRESS_S - tau
                tt_cnt[o, b, dcx] += 1
                legs_sum[o, b, dcx] += k
    return tt_sum, tt_cnt, legs_sum


@njit(parallel=True, cache=True)
def _assign(q, taus, bin_of, sub_per_bin, horizon, c_dep, c_arr, c_from, c_to, c_trip, c_seq,
            trip_conns, trip_first, change_s, xfer_ptr, xfer_to, xfer_s, cx_ptr, cx_stops, n_chunks):
    """Add demand onto connections. q[o, d, b] = trips entering o in hour bin b, bound for d."""
    n_cx = len(cx_ptr) - 1
    n_stops = len(change_s)
    n_trips = len(trip_first) - 1
    loads = np.zeros((n_chunks, len(c_dep)), dtype=np.float64)
    unrouted = np.zeros(n_chunks, dtype=np.float64)
    for ch in prange(n_chunks):
        ready, ready_from, arr_t, arr_conn, arr_board, board = _state(n_stops, n_trips)
        for o in range(ch, n_cx, n_chunks):
            for j in range(len(taus)):
                b = bin_of[j]
                tot = 0.0
                for dcx in range(n_cx):
                    tot += q[o, dcx, b]
                if tot <= 0.0:
                    continue
                tau = taus[j]
                _scan(o, tau, tau + horizon, c_dep, c_arr, c_from, c_to, c_trip, change_s, xfer_ptr,
                      xfer_to, xfer_s, cx_ptr, cx_stops, ready, ready_from, arr_t, arr_conn, arr_board, board)
                for dcx in range(n_cx):
                    w = q[o, dcx, b] / sub_per_bin
                    if w <= 0.0 or dcx == o:
                        continue
                    p, k = _choose(dcx, cx_ptr, cx_stops, arr_t)
                    if p < 0:
                        unrouted[ch] += w
                        continue
                    while k > 0:
                        ca = arr_conn[k, p]
                        cb = arr_board[k, p]
                        base = trip_first[c_trip[ca]]
                        for i in range(c_seq[cb], c_seq[ca] + 1):
                            loads[ch, trip_conns[base + i]] += w
                        k -= 1
                        p = ready_from[k, c_from[cb]]
    return loads.sum(axis=0), unrouted.sum()


def taus_for(t_start: int, t_end: int, bin_s: int, sub_per_bin: int):
    """Sub-departure times (midpoints of equal slices of each bin) and their bin index."""
    step = bin_s // sub_per_bin
    taus = np.arange(t_start, t_end, step, dtype=np.int32) + step // 2
    bin_of = ((taus - t_start) // bin_s).astype(np.int32)
    return taus, bin_of


def _net(net: Network):
    return (net.c_dep, net.c_arr, net.c_from, net.c_to, net.c_trip)


def travel_times(net: Network, t_start: int, t_end: int, bin_s: int = 3600, sub_per_bin: int = 6,
                 horizon: int = 3 * 3600):
    """Mean door-to-door time [o, bin, d] in seconds (NaN where unreachable for any
    sub-departure of the bin) and mean number of legs of the chosen journeys."""
    taus, bin_of = taus_for(t_start, t_end, bin_s, sub_per_bin)
    n_bins = int(bin_of.max()) + 1
    tt_sum, tt_cnt, legs = _travel_times(
        taus, bin_of, n_bins, horizon, *_net(net), net.change_s, net.xfer_ptr, net.xfer_to,
        net.xfer_s, net.cx_ptr, net.cx_stops, net.trips.height)
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = (tt_sum / tt_cnt).astype(np.float32)
        legs = (legs / tt_cnt).astype(np.float32)
    tt[tt_cnt < sub_per_bin] = np.nan
    legs[tt_cnt < sub_per_bin] = np.nan
    return tt, legs


def assign_loads(net: Network, q: np.ndarray, t_start: int, bin_s: int = 3600, sub_per_bin: int = 30,
                 horizon: int = 3 * 3600):
    """Passengers on board each connection, given q[o, d, bin] (entries per bin)."""
    n_bins = q.shape[2]
    taus, bin_of = taus_for(t_start, t_start + n_bins * bin_s, bin_s, sub_per_bin)
    return _assign(
        np.ascontiguousarray(q, dtype=np.float64), taus, bin_of, float(sub_per_bin), horizon,
        *_net(net), net.c_seq, net.trip_conns, net.trip_first, net.change_s, net.xfer_ptr,
        net.xfer_to, net.xfer_s, net.cx_ptr, net.cx_stops, numba.get_num_threads())
