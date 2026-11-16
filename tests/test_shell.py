import numpy as np
import pytest

torch = pytest.importorskip("torch")
from femweb.solvers import shell as S


def _cantilever(k, alt):
    E, L, b, t = 210000.0, 100.0, 10.0, 1.0
    X, T = S.grid(L, b, 40, 4, alt)
    cl = np.where(X[:, 0] < 1e-9)[0]
    sh = S.Shell(X, T, E, 0.0, t, clamped=cl)
    fixed = np.concatenate([cl * 3, cl * 3 + 1, cl * 3 + 2])
    tip = np.where(X[:, 0] > L - 1e-9)[0]
    P = k * E * b * t**3 / 12 / L**2
    f = np.zeros(sh.ndof)
    f[tip * 3 + 2] = 1.0
    f[tip[[0, -1]] * 3 + 2] = 0.5
    f *= P / f.sum()
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 8)
    x = h[-1]["x"][: sh.nx].reshape(-1, 3)
    return x[tip, 2].mean() / L, -(x[tip, 0] - L).mean() / L


@pytest.mark.parametrize("alt", [False, True])
def test_cantilever_elastica(alt):
    w, u = _cantilever(1.0, alt)
    assert abs(w - 0.3017) < 0.002
    assert abs(u - 0.0564) < 0.001


@pytest.mark.parametrize("alt,jit", [(False, 0.0), (True, 0.25)])
def test_plate_simply_supported(alt, jit):
    E, nu, t, a, q, n = 210000.0, 0.3, 1.0, 100.0, 1e-6, 16
    X, T = S.grid(a, a, n, n, alt)
    bd = np.where((X[:, 0] < 1e-9) | (X[:, 0] > a - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > a - 1e-9))[0]
    inner = np.setdiff1d(np.arange(len(X)), bd)
    X[inner, :2] += np.random.default_rng(1).uniform(-1, 1, (len(inner), 2)) * jit * a / n
    sh = S.Shell(X, T, E, nu, t)
    fixed = np.concatenate([bd * 3, bd * 3 + 1, bd * 3 + 2])
    f = np.zeros(sh.ndof)
    f[2 : sh.nx : 3] = q * sh.lumped_area()
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 1)
    w = h[-1]["x"][: sh.nx].reshape(-1, 3)[:, 2].max()
    D = E * t**3 / 12 / (1 - nu**2)
    assert abs(w * D / (q * a**4) - 0.00406) < 0.00406 * 0.03


def test_tube_axial_stiffness():
    E, a, L, t, d = 70000.0, 40.0, 80.0, 1.0, 0.01
    X, T = S.tube(a, L, 6, 12)
    sh = S.Shell(X, T, E, 0.0, t)
    bot = np.where(X[:, 2] < 1e-9)[0]
    top = np.where(X[:, 2] > L - 1e-9)[0]
    fixed = bot * 3 + 2
    fixed = np.concatenate([fixed, [bot[0] * 3, bot[0] * 3 + 1, bot[1] * 3 + 1]])
    h = S.run_static(sh, X, fixed, top * 3 + 2, np.full(len(top), -d), np.zeros(sh.ndof), 1)
    R = h[-1]["reaction"][top * 3 + 2].sum()
    assert abs(-R - E * 4 * a * t * d / L) < 0.01 * E * 4 * a * t * d / L


def test_self_contact_strip_on_plate():
    E, L, b, t = 210000.0, 100.0, 10.0, 1.0
    Xa, Ta = S.grid(L + 20, b + 10, 24, 4)
    Xa[:, 0] -= 10
    Xa[:, 1] -= 5
    Xb, Tb = S.grid(L, b, 20, 2)
    Xb[:, 2] = 1.5
    X = np.vstack([Xa, Xb])
    T = np.vstack([Ta, Tb + len(Xa)])
    na = len(Xa)
    cl = na + np.where(Xb[:, 0] < 1e-9)[0]
    sh = S.Shell(X, T, E, 0.0, t, clamped=cl)
    fa = np.arange(na)
    fixed = np.concatenate([fa * 3, fa * 3 + 1, fa * 3 + 2, cl * 3, cl * 3 + 1, cl * 3 + 2])
    tip = na + np.where(Xb[:, 0] > L - 1e-9)[0]
    P = E * b * t**3 / 12 / L**2
    f = np.zeros(sh.ndof)
    f[tip * 3 + 2] = -P / len(tip)
    ct = S.SelfContact(sh)
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 10, extra=ct)
    assert not h[-1].get("failed")
    x = h[-1]["x"]
    z = x[: sh.nx].reshape(-1, 3)[na:, 2]
    F, nact = ct.force(x)
    Rc = h[-1]["reaction"][cl * 3 + 2].sum()
    assert nact > 0
    assert z.min() > 1.0 - 0.02
    assert abs(Rc + F - P) < 0.02 * P
