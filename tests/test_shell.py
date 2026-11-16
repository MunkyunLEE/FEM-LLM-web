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


def _strip_on_plate(mu):
    E, L, b, t = 210000.0, 100.0, 10.0, 1.0
    Xa, Ta = S.grid(L + 20, b + 10, 24, 4)
    Xa[:, 0] -= 10
    Xa[:, 1] -= 5
    Xb, Tb = S.grid(L, b, 20, 2)
    Xb[:, 2] = 1.6
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
    ct = S.IPCContact(sh, mu=mu)
    h = S.run_static(sh, X, fixed, np.array([], int), [], f, 10, extra=ct)
    assert not h[-1].get("failed")
    x = h[-1]["x"]
    z = x[: sh.nx].reshape(-1, 3)[na:, 2]
    _, nact, gap = ct.force(x)
    R = h[-1]["reaction"]
    assert nact > 0
    assert gap > t
    assert z.min() > t
    assert abs(R[cl * 3 + 2].sum() + R[fa * 3 + 2].sum() - P) < 0.01 * P


@pytest.mark.parametrize("mu", [0.0, 0.3])
def test_ipc_strip_on_plate(mu):
    pytest.importorskip("ipctk")
    _strip_on_plate(mu)


def test_ipc_rigid_plane_on_arch():
    pytest.importorskip("ipctk")
    E, L, b, t = 210000.0, 100.0, 10.0, 1.0
    X, T = S.grid(L, b, 20, 2)
    X[:, 2] = 5 * np.sin(np.pi * X[:, 0] / L)
    en = np.where((X[:, 0] < 1e-9) | (X[:, 0] > L - 1e-9))[0]
    fx = np.concatenate([en * 3, en * 3 + 1, en * 3 + 2])
    sh = S.Shell(X, T, E, 0.0, t)
    ct = S.IPCContact(sh, planes=[{"point": [0, 0, 6.6], "normal": [0, 0, -1], "motion": [0, 0, -2.5]}])
    h = S.run_static(sh, X, fx, np.array([], int), [], np.zeros(sh.ndof), 10, extra=ct)
    assert not h[-1].get("failed")
    x = h[-1]["x"]
    crown = x[: sh.nx].reshape(-1, 3)[:, 2].max()
    Fp = ct.plane_force(x)[2]
    assert 4.1 - crown > t
    assert Fp > 100
    assert abs(abs(h[-1]["reaction"][en * 3 + 2].sum()) - Fp) < 0.01 * Fp


def test_rayleigh_coefficients():
    al, be = S.rayleigh(0.05, 10.0, 30.0)
    for w in (10.0, 30.0):
        assert abs(al / (2 * w) + be * w / 2 - 0.05) < 1e-12


def _tube_model():
    from femweb.io_obj import parse_obj

    a, L, npa, nz = 40.0, 120.0, 4, 12
    ring = []
    for side in range(4):
        for i in range(npa):
            q = a * i / npa
            ring.append([(q, 0.0), (a, q), (a - q, a), (0.0, a - q)][side])
    m = len(ring)
    out = [f"v {x} {y} {L * k / nz}" for k in range(nz + 1) for x, y in ring]
    for k in range(nz):
        for i in range(m):
            p, q = k * m + i, k * m + (i + 1) % m
            out.append(f"f {p + 1} {q + 1} {q + m + 1} {p + m + 1}")
    return parse_obj("\n".join(out) + "\n", "tube.obj")


def test_adapter_tube_with_contact_and_damping():
    pytest.importorskip("ipctk")
    from femweb.spec import apply_ops, default_spec, warnings_for

    base = _tube_model()
    ops = [
        {"op": "set_solver", "solver": "shell"},
        {"op": "set_material", "name": "default", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9},
        {"op": "set_section", "thickness": 1.0},
        {"op": "add_bc", "dofs": [1, 2, 3], "select": {"plane": {"axis": "z", "at": "min"}}},
        {"op": "add_bc", "dofs": [1, 2], "select": {"plane": {"axis": "z", "at": "max"}}},
        {"op": "add_load", "type": "force", "vector": [0, 0, -1000.0], "select": {"plane": {"axis": "z", "at": "max"}}},
        {"op": "set_step", "type": "static", "increments": 4, "imperfection": 0.2, "control": "displacement", "control_disp": 0.5, "control_kind": "rigid", "damping": {"kind": "rayleigh", "zeta": 2}},
        {"op": "add_contact", "type": "self", "friction": 0.2},
    ]
    spec, _ = apply_ops(default_spec(base), ops, base)
    assert warnings_for(spec, base) == []
    r = S.solve(base, spec)
    k = r.history["control.force"][0] / r.history["control.disp"][0]
    assert abs(k - 70000.0 * 160.0 / 120.0) < 0.1 * 70000.0 * 160.0 / 120.0
    assert r.summary["감쇠 소산 / 변형 에너지"] < 0.01
    assert len(r.frames) >= 4


def test_rules_shell_sentences():
    from femweb import rules
    from femweb.spec import default_spec

    m = _tube_model()
    sp = default_spec(m)
    assert rules.parse("자체 쉘 솔버로 풀어줘", m, sp)[0][0] == {"op": "set_solver", "solver": "shell"}
    assert rules.parse("자기 접촉 추가 마찰 0.2", m, sp)[0] == [{"op": "add_contact", "type": "self", "friction": 0.2}]
    assert rules.parse("레일리 감쇠 5%", m, sp)[0] == [{"op": "set_step", "damping": {"kind": "rayleigh", "zeta": 5.0}}]


def test_bc_conflict_warning():
    from femweb.spec import apply_ops, default_spec, warnings_for

    m = _tube_model()
    top = {"plane": {"axis": "z", "at": "max"}}
    spec, _ = apply_ops(default_spec(m), [{"op": "add_bc", "dofs": [3], "select": top}, {"op": "add_bc", "dofs": [3], "value": -1.0, "select": top}], m)
    assert any("서로 다른 값" in w for w in warnings_for(spec, m))
