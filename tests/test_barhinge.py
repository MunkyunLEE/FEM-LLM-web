import numpy as np
from helpers import load_example
from femweb.solvers import barhinge
from femweb.solvers.barhinge import Truss, run_arclength, run_static


def test_gradient_matches_fd():
    rng = np.random.default_rng(1)
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.3], [0, 1, 0], [0.5, -0.8, 0.4]], dtype=float)
    bars = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (0, 4), (1, 4)]
    tr = Truss(X, bars, np.full(len(bars), 100.0), [(3, 0, 2, 1), (2, 0, 1, 4)], [2.0, 3.0], [0, 1])
    x = X + rng.normal(scale=0.05, size=X.shape)
    g = tr.gradient(x)
    fd = np.zeros_like(x)
    e = 1e-6
    for i in range(x.size):
        xp = x.copy().ravel()
        xm = x.copy().ravel()
        xp[i] += e
        xm[i] -= e
        fd.ravel()[i] = (tr.energy(xp.reshape(-1, 3)) - tr.energy(xm.reshape(-1, 3))) / (2 * e)
    assert np.abs(g - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())


def test_snap_through_arclength():
    b, h, EA = 100.0, 10.0, 1000.0
    X = np.array([[-b, 0, 0], [0, 0, h], [b, 0, 0]])
    tr = Truss(X, [(0, 1), (1, 2)], [EA, EA])
    free = np.array([5])
    F0 = np.zeros((3, 3))
    F0[1, 2] = -1.0
    frames, msgs, _ = run_arclength(tr, free, F0, 60, 0.05)
    L0 = np.hypot(b, h)
    lam = np.array([f[0] for f in frames])
    z = np.array([f[1][1, 2] for f in frames])
    exact = -2 * EA * (np.hypot(b, z) - L0) / L0 * z / np.hypot(b, z)
    assert np.abs(lam - exact).max() < 1e-6 * np.abs(exact).max() + 1e-9
    assert z.min() < -h * 0.5
    assert (np.diff(lam) < 0).any()


def test_miura_folds_nearly_rigidly():
    base, spec = load_example("miura")
    res = barhinge.solve(base, spec)
    s = res.summary
    assert s["프레임 수"] == 30
    ratio = s["에너지 비율 바/굽힘/접힘"]
    assert ratio[2] > 0.9
    assert s["최대 바 변형률"] < 1e-3
    assert abs(res.history["u.max"][-1]) > 20
