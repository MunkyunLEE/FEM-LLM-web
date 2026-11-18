import numpy as np
import pytest
from fastapi.testclient import TestClient

from bhweb import BHModel, Problem, solve_displacement
from bhweb.analysis import dof_analysis, modal, rigid_fold_path
from bhweb.core import Assembler, State, evaluate
from bhweb.geom import dihedral, dihedral_grad, dihedral_hess
from bhweb.model import FOLD
from bhweb.shapes import miura, strip
from femweb import server

c = TestClient(server.app)
server.S["llm"]["mode"] = "rule"


def test_dihedral_grad_hess_match_finite_difference():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(5, 4, 3))
    xs = [P[:, q] for q in range(4)]
    g = dihedral_grad(*xs)
    H = dihedral_hess(*xs)
    h = 1e-6
    for k in range(12):
        Q = P.reshape(5, 12).copy()
        Q[:, k] += h
        a = dihedral(*[Q.reshape(5, 4, 3)[:, q] for q in range(4)])
        Q[:, k] -= 2 * h
        b = dihedral(*[Q.reshape(5, 4, 3)[:, q] for q in range(4)])
        assert np.allclose((a - b) / (2 * h), g[:, k], atol=1e-7)
        Q = P.reshape(5, 12).copy()
        Q[:, k] += h
        ga = dihedral_grad(*[Q.reshape(5, 4, 3)[:, q] for q in range(4)])
        Q[:, k] -= 2 * h
        gb = dihedral_grad(*[Q.reshape(5, 4, 3)[:, q] for q in range(4)])
        assert np.allclose((ga - gb) / (2 * h), H[:, :, k], atol=1e-6)


def test_tangent_matches_finite_difference():
    X, F, cx, cy = miura(nx=2, ny=2, theta=50)
    m = BHModel(X, F, crease_edges=cx + cy, params={"bar_yield": 1e9})
    asm = Assembler(m)
    st = State(m)
    x = m.X + np.random.default_rng(1).normal(scale=0.5, size=m.X.shape)
    T, K, _ = evaluate(m, asm, x, st, True)
    K = K.toarray()
    h = 1e-6
    for k in range(0, m.ndof, 5):
        xp = x.ravel().copy()
        xp[k] += h
        Tp, _, _ = evaluate(m, asm, xp.reshape(-1, 3), st, False)
        xp[k] -= 2 * h
        Tm, _, _ = evaluate(m, asm, xp.reshape(-1, 3), st, False)
        fd = (Tp.ravel() - Tm.ravel()) / (2 * h)
        assert np.abs(fd - K[:, k]).max() < 1e-6 * max(1.0, np.abs(K).max())


def _square(W=10.0, H=10.0):
    X = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], float)
    return X, [[0, 1, 2, 3]]


@pytest.mark.parametrize("mode", ["tension", "shear"])
def test_n5b8_in_plane_stiffness_matches_filipov(mode):
    E, nu, t, W, H = 1000.0, 1.0 / 3.0, 0.1, 10.0, 10.0
    X, F = _square(W, H)
    m = BHModel(X, F, params={"E": E, "nu": nu, "t": t})
    d = 1e-6 * H
    fixed = [2 + 3 * i for i in range(m.nn)]
    fixed += [0, 1, 4]
    disp = {}
    if mode == "tension":
        fixed += [3 * 3]
        disp = {3 * 2 + 1: d, 3 * 3 + 1: d}
        ref = E * W * t / H
    else:
        fixed += [3]
        fixed += [3 * 2 + 1, 3 * 3 + 1]
        disp = {3 * 2: d, 3 * 3: d}
        ref = E / (2 * (1 + nu)) * W * t / H
    pb = Problem(m, fixed=fixed, disp=disp)
    rec = solve_displacement(m, pb, steps=1)
    R = rec.react[-1]
    f = R[3 * 2 + 1] + R[3 * 3 + 1] if mode == "tension" else R[3 * 2] + R[3 * 3]
    assert abs(f / d / ref - 1) < 1e-4


def test_simple_fold_elastoplastic_matches_bilinear():
    X = np.array([[0, 0, 0], [10, 0, 0], [5, 8, 0], [5, -8, 0]], float)
    F = [[0, 1, 2], [1, 0, 3]]
    k_len, Y, Hh = 2.0, 0.5, 0.4
    m = BHModel(X, F, crease_edges=[(0, 1)], params={"fold_mode": "value", "fold_k": k_len, "fold_yield": Y, "fold_H": Hh, "E": 1e6})
    i = m.fold_index((0, 1))
    L = 10.0
    fixed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    disp = {3 * 3 + 2: 6.0}
    rec = solve_displacement(m, Problem(m, fixed=fixed, disp=disp), steps=40)
    th = np.array([t[i] for t in rec.th])
    M = np.array([q[i] for q in rec.M])
    rot = np.abs(th - np.pi)
    k = k_len * L
    my = Y * L
    hh = Hh * L
    ref = np.where(k * rot <= my, k * rot, my + k * hh / (k + hh) * (rot - my / k))
    assert np.abs(M).max() > 1.3 * my
    assert np.allclose(np.abs(M), ref, rtol=1e-6, atol=1e-9)


def test_miura_rigid_fold_relation_cos_gamma():
    X, F, cx, cy = miura(nx=2, ny=2, gamma=60, theta=40)
    m = BHModel(X, F, crease_edges=cx + cy)
    da = dof_analysis(m)
    assert da["mechanisms"] == 1
    ia = m.fold_index(cy[0])
    ib = m.fold_index(cx[0])
    p = rigid_fold_path(m, [ia], [m.th_init[ia] + 0.6], steps=20)
    for k in (5, 20):
        ra = np.pi - p["angles"][k][ia]
        rb = np.pi - p["angles"][k][ib]
        assert abs(np.tan(ra / 2) / np.tan(rb / 2) - 0.5) < 1e-8
    assert p["err"].max() < 1e-6


def test_folded_miura_modal_has_one_soft_folding_mode():
    X, F, cx, cy = miura(nx=2, ny=2, theta=50)
    m = BHModel(X, F, crease_edges=cx + cy, params={"Lstar": 1e4})
    r = modal(m, n=10, use_mass=False)
    ev = np.sort(np.abs(r["eig"]))
    assert r["rigid"] == 6
    assert ev[6] < 1e-2 * ev[7]


def test_contact_stops_fold_through():
    pytest.importorskip("ipctk")
    from bhweb.contact import GeometricContact

    X = np.array([[0, 0, 0], [10, 0, 0], [5, 8, 0], [5, -8, 0]], float)
    F = [[0, 1, 2], [1, 0, 3]]
    m = BHModel(X, F, crease_edges=[(0, 1)], params={"fold_mode": "value", "fold_k": 1.0, "E": 1e6})
    fixed = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    i = m.fold_index((0, 1))
    rest = {i: -np.radians(200.0)}
    free = solve_displacement(m, Problem(m, fixed=fixed, rest=rest), steps=40)
    assert free.th[-1][i] < 0 or free.th[-1][i] > np.pi
    ct = GeometricContact(m)
    rec = solve_displacement(m, Problem(m, fixed=fixed, contact=ct, rest=rest), steps=40)
    x = (m.X.ravel() + rec.U[-1]).reshape(-1, 3)
    assert not ct.intersects(x)
    assert rec.th[-1][i] > 0 and rec.th[-1][i] < 2 * np.pi
    assert min(rec.th[-1][i], 2 * np.pi - rec.th[-1][i]) < np.radians(5)


def test_strip_tension_and_bending_scale():
    X, F = strip(100.0, 10.0, 10)
    m = BHModel(X, F, params={"E": 1000.0, "nu": 0.3, "t": 0.5})
    assert m.summary()["folds"] == 9
    m = BHModel(X, F, crease_edges=[(4, 15)], params={"E": 1000.0, "nu": 0.3, "t": 0.5})
    assert m.summary()["folds"] == 1 and m.summary()["edge_bends"] == 8


def _bh_api(step, extra=None):
    assert c.post("/api/examples/miura").status_code == 200
    ops = [{"op": "set_solver", "solver": "bh"}, {"op": "set_step", "type": step}] + (extra or [])
    assert c.post("/api/ops", json={"ops": ops}).status_code == 200
    r = c.post("/api/solve")
    assert r.status_code == 200, r.text
    return c.get("/api/state").json()["result"]


def test_api_bh_static_energy_balance():
    r = _bh_api("static")
    s = r["summary"]
    assert any("에너지" in k for k in s)


def test_api_bh_modal_and_kinematic():
    r = _bh_api("modal")
    assert "강체 접힘 기구 수 (패널 강체 가정)" in r["summary"] and r["summary"]["고유진동수 (Hz)"]
    cr = c.get("/api/creases").json()["edges"]
    folds = ["%d-%d" % tuple(e["edge"]) for e in cr if e["type"] == "fold"][:3]
    r = _bh_api("kinematic", [{"op": "set_bh", "plot3": folds}])
    assert r["summary"]["최대 구속 오차"] < 1e-5
    k = c.get("/api/result/kin3d").json()
    assert k["dof"] == 1 and len(k["edges"]) == 3


def test_set_crease_ops_and_validation():
    assert c.post("/api/examples/miura").status_code == 200
    c.post("/api/ops", json={"ops": [{"op": "set_solver", "solver": "bh"}]})
    cr = c.get("/api/creases").json()["edges"]
    e = next(x["edge"] for x in cr if x["type"] == "fold")
    assert c.post("/api/ops", json={"ops": [{"op": "set_crease", "edges": [e], "type": "bend"}]}).status_code == 200
    cr2 = {tuple(x["edge"]): x["type"] for x in c.get("/api/creases").json()["edges"]}
    assert cr2[tuple(e)] == "bend"
    assert c.post("/api/ops", json={"ops": [{"op": "set_crease", "edges": [[0, 999]], "type": "bend"}]}).status_code != 200
    c.post("/api/ops", json={"ops": [{"op": "set_crease", "edges": [e], "type": "auto"}]})
    cr3 = {tuple(x["edge"]): x["type"] for x in c.get("/api/creases").json()["edges"]}
    assert cr3[tuple(e)] == "fold"
