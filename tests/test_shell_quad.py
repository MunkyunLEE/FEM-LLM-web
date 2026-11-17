import numpy as np
import pytest

torch = pytest.importorskip("torch")
from femweb.solvers import shell as S
from femweb.solvers import shell_quad as Q


def test_rigid_rotation_has_no_energy():
    from scipy.spatial.transform import Rotation

    X, F = Q.qgrid(100.0, 10.0, 6, 2)
    sh = Q.QuadShell(X, F, 210000.0, 0.3, 1.0)
    th = np.array([0.3, -0.2, 0.5])
    x = sh.x0()
    x[: sh.nx] = (X @ Rotation.from_rotvec(th).as_matrix().T).ravel()
    x[sh.nx :] = np.tile(th, sh.n)
    assert sh.energy(x) < 1e-12


def test_tangent_matches_finite_difference():
    X, F = Q.qgrid(100.0, 10.0, 6, 2)
    sh = Q.QuadShell(X, F, 210000.0, 0.3, 1.0)
    rng = np.random.default_rng(0)
    x = sh.x0() + 1e-2 * rng.normal(size=sh.ndof)
    d = rng.normal(size=sh.ndof)
    h = 1e-6
    fd = (sh.grad(x + h * d) - sh.grad(x - h * d)) / (2 * h)
    assert np.linalg.norm(sh.hess(x) @ d - fd) < 1e-6 * np.linalg.norm(fd)


@pytest.mark.parametrize("k,w,u", [(1.0, 0.3017, 0.0564), (2.0, 0.4935, 0.1606)])
def test_cantilever_elastica(k, w, u):
    E, L, b, t = 210000.0, 100.0, 10.0, 1.0
    X, F = Q.qgrid(L, b, 20, 2)
    cl = np.where(X[:, 0] < 1e-9)[0]
    tip = np.where(X[:, 0] > L - 1e-9)[0]
    sh = Q.QuadShell(X, F, E, 0.0, t, rot_fixed=[(i, c) for i in cl for c in range(3)])
    fixed = np.concatenate([cl * 3, cl * 3 + 1, cl * 3 + 2])
    f = np.zeros(sh.ndof)
    f[tip * 3 + 2] = 1.0
    f[tip[[0, -1]] * 3 + 2] = 0.5
    f *= k * E * b * t**3 / 12 / L**2 / f.sum()
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 8)
    x = h[-1]["x"][: sh.nx].reshape(-1, 3)
    assert abs(x[tip, 2].mean() / L - w) < 0.002
    assert abs(-(x[tip, 0] - L).mean() / L - u) < 0.001


@pytest.mark.parametrize("bc,ref,jit", [("ss", 0.00406, 0.0), ("cl", 0.00126, 0.25)])
def test_plate_bending(bc, ref, jit):
    E, nu, t, a, q, n = 210000.0, 0.3, 1.0, 100.0, 1e-6, 16
    X, F = Q.qgrid(a, a, n, n)
    bd = np.where((X[:, 0] < 1e-9) | (X[:, 0] > a - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > a - 1e-9))[0]
    inner = np.setdiff1d(np.arange(len(X)), bd)
    X[inner, :2] += np.random.default_rng(1).uniform(-1, 1, (len(inner), 2)) * jit * a / n
    sh = Q.QuadShell(X, F, E, nu, t, rot_fixed=[(i, c) for i in bd for c in range(3)] if bc == "cl" else None)
    fixed = np.concatenate([bd * 3, bd * 3 + 1, bd * 3 + 2])
    f = np.zeros(sh.ndof)
    f[2 : sh.nx : 3] = q * sh.lumped_area()
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 1)
    w = h[-1]["x"][: sh.nx].reshape(-1, 3)[:, 2].max() * E * t**3 / 12 / (1 - nu**2) / (q * a**4)
    assert abs(w - ref) < 0.02 * ref


def test_in_plane_bending_does_not_lock():
    E, L, b, t, P = 210000.0, 100.0, 10.0, 1.0, 1e-3
    X, F = Q.qgrid(L, b, 20, 2)
    cl = np.where(X[:, 0] < 1e-9)[0]
    tip = np.where(X[:, 0] > L - 1e-9)[0]
    sh = Q.QuadShell(X, F, E, 0.0, t)
    fixed = np.concatenate([cl * 3, cl * 3 + 1, np.arange(len(X)) * 3 + 2])
    f = np.zeros(sh.ndof)
    f[tip * 3 + 1] = P / len(tip)
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 1)
    v = h[-1]["x"][: sh.nx].reshape(-1, 3)[tip, 1].mean() - b / 2
    ref = P * L**3 / (3 * E * t * b**3 / 12) + P * L / (5 / 6 * E / 2 * b * t)
    assert abs(v / ref - 1) < 0.01


def test_tube_buckling_load():
    E, nu, a, L, t = 70000.0, 0.3, 40.0, 120.0, 1.0
    X, F = Q.qtube(a, L, 8, 24)
    sh = Q.QuadShell(X, F, E, nu, t)
    bot = np.where(X[:, 2] < 1e-9)[0]
    top = np.where(X[:, 2] > L - 1e-9)[0]
    ends = np.concatenate([bot, top])
    fixed = np.concatenate([bot * 3 + 2, ends * 3, ends * 3 + 1])
    fac, _ = S.linear_buckling(sh, fixed, top * 3 + 2, np.full(len(top), -1.0), np.zeros(sh.ndof), k=1)
    assert abs(fac[0] * E * 4 * a * t / L / 1000 - 24.43) < 0.04 * 24.43


def test_plastic_uniaxial():
    E, nu, sy, Hm, L, b, t, d = 70000.0, 0.3, 200.0, 1000.0, 100.0, 20.0, 1.0, 2.0
    X, F = Q.qgrid(L, b, 10, 2)
    left = np.where(X[:, 0] < 1e-9)[0]
    right = np.where(X[:, 0] > L - 1e-9)[0]
    fixed = np.concatenate([left * 3, [left[0] * 3 + 1], np.arange(len(X)) * 3 + 2])
    sh = Q.QuadShell(X, F, E, nu, t, plastic=[(0, sy), (1, sy + Hm)])
    h = S.run_static(sh, X, fixed, right * 3, np.full(len(right), d), np.zeros(sh.ndof), 10)
    e = d / L
    eG = e + 0.5 * e * e
    ref = (sy + E * Hm / (E + Hm) * (eG - sy / E)) * (1 + e)
    assert abs(h[-1]["reaction"][right * 3].sum() / (b * t) - ref) < 1e-3 * ref
    assert abs(sh.peeq_max() - (eG - sy / E) * E / (E + Hm)) < 1e-6


def test_adapter_picks_quads_and_moment_load():
    from femweb.io_obj import parse_obj
    from femweb.spec import apply_ops, default_spec

    L, b, nx, ny, t, E = 100.0, 10.0, 20, 2, 1.0, 210000.0
    out = [f"v {L * i / nx} {b * j / ny} 0" for j in range(ny + 1) for i in range(nx + 1)]
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i + 1
            out.append(f"f {a} {a + 1} {a + nx + 2} {a + nx + 1}")
    m = parse_obj("\n".join(out) + "\n", "strip.obj")
    Mt = 0.05 * E * b * t**3 / 12 / L
    ops = [
        {"op": "set_solver", "solver": "shell"},
        {"op": "set_material", "name": "default", "E": E, "nu": 0.0},
        {"op": "set_section", "thickness": t},
        {"op": "add_bc", "type": "fixed", "select": {"plane": {"axis": "x", "at": "min"}}},
        {"op": "add_load", "type": "moment", "vector": [0, -Mt, 0], "select": {"plane": {"axis": "x", "at": "max"}}},
        {"op": "set_step", "type": "static", "increments": 2},
    ]
    spec, _ = apply_ops(default_spec(m), ops, m)
    r = S.solve(m, spec)
    assert "사각형 요소 수" in r.summary
    tip = np.where(m.nodes[:, 0] > L - 1e-9)[0]
    w = r.frames[-1]["U"][tip, 2].mean()
    assert abs(w - 0.05 * L / 2) < 0.03 * 0.05 * L / 2


def test_element_rule_needs_the_word_element():
    from femweb import rules
    from femweb.io_obj import parse_obj
    from femweb.spec import default_spec

    m = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", "q.obj")
    sp = default_spec(m)
    assert rules.parse("쿼드 요소로 풀어줘", m, sp)[0] == [{"op": "set_step", "element": "quad"}]
    assert not any(o.get("element") for o in rules.parse("사각형 관을 눌러줘", m, sp)[0])
