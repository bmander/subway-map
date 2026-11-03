import numpy as np
import polars as pl

from subway import inversion as I


def tiny_problem(seed=0):
    rng = np.random.default_rng(seed)
    n, nb = 4, 6
    E = rng.uniform(50, 500, size=(n, nb))
    tt = rng.uniform(600, 3000, size=(n, nb, n))
    legs = rng.integers(1, 3, size=(n, nb, n)).astype(float)
    f = I.deterrence(tt, legs)
    # exit observations: two 3h intervals per complex, one group each, plus symmetry
    t0, t1, y, g, d_ = [], [], [], [], []
    for d in range(n):
        for a in (0, 3 * 3600):
            t0.append(a); t1.append(a + 3 * 3600); y.append(rng.uniform(100, 900)); g.append(d); d_.append(d)
        t0.append(0); t1.append(nb * 3600); y.append(E[d].sum()); g.append(-1); d_.append(d)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, np.array(d_) + 1, 1)
    obs = I.Observations(E=E, obs_ptr=np.cumsum(ptr), obs_t0=np.array(t0, float), obs_t1=np.array(t1, float),
                         obs_y=np.array(y, float), obs_group=np.array(g, dtype=np.int64), n_groups=n,
                         groups=pl.DataFrame({"g": np.arange(n), "complex_idx": np.arange(n), "phase_s": 0}))
    return obs, tt, f


def test_demand_is_production_constrained():
    obs, tt, f = tiny_problem()
    logA = np.random.default_rng(1).normal(size=obs.E.shape)
    q = I._demand(obs.E, logA, f)
    np.testing.assert_allclose(q.sum(axis=1), obs.E, rtol=1e-12)
    assert np.all(q[np.arange(4), np.arange(4), :] == 0)


def test_gradient_matches_finite_differences():
    obs, tt, f = tiny_problem()
    prob = I.make_problem(obs, tt, f, t_start=0.0, sym_bins=(0, obs.E.shape[1]), verbose=False)
    x = prob.x0 + np.random.default_rng(2).normal(0, 0.3, size=prob.x0.shape)
    _, g = prob.objective(x)
    h = 1e-5
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        fd = (prob.objective(x + e)[0] - prob.objective(x - e)[0]) / (2 * h)
        assert abs(g[i] - fd) <= 1e-5 * max(1.0, abs(fd)), (i, g[i], fd)
