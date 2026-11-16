import importlib.util

import numpy as np
import pytest
from helpers import active, load_example
from femweb import rules
from femweb.contact import bodies, icosphere, place
from femweb.solvers import calculix
from femweb.spec import apply_ops, warnings_for

ccx = pytest.mark.skipif(not calculix.find_ccx(), reason="ccx not installed")
tm = pytest.mark.skipif(importlib.util.find_spec("tensormesh") is None, reason="tensormesh 없음")


def test_spec_contact_ops_and_auto_place():
    base, spec = load_example("shell_panel")
    spec, _ = apply_ops(spec, [{"op": "set_section", "thickness": 2.0}, {"op": "add_contact", "type": "rigid", "shape": "sphere", "radius": 40, "motion": [0, 0, -3], "friction": 0.2}], base)
    c = spec["contacts"][0]
    assert c["id"] == "ct1" and np.allclose(c["center"], [100, 100, 41])
    assert any(s["quantity"] == "contact.force" for p in spec["plots"] for s in p["series"])
    spec, _ = apply_ops(spec, [{"op": "add_contact", "type": "rigid", "shape": "plane", "normal": [0, 0, 1]}], base)
    assert np.allclose(spec["contacts"][1]["point"][2], -1.0)
    spec, _ = apply_ops(spec, [{"op": "update", "id": "ct1", "friction": 0.5}, {"op": "remove", "id": "ct2"}], base)
    assert spec["contacts"][0]["friction"] == 0.5 and len(spec["contacts"]) == 1
    with pytest.raises(ValueError):
        apply_ops(spec, [{"op": "add_contact", "type": "rigid", "shape": "sphere", "radius": -1}], base)
    sp, _ = apply_ops(spec, [{"op": "set_solver", "solver": "barhinge"}], None)
    assert any("접촉을 지원하지" in w for w in warnings_for(sp, base))
    spec, _ = apply_ops(spec, [{"op": "clear", "what": "contacts"}], base)
    assert spec["contacts"] == []


def test_rules_contact_sentences():
    base, spec = load_example("shell_panel")
    ops, _, info = rules.parse("반지름 20 구를 아래로 3 mm 눌러, 마찰 0.2", base, spec)
    assert info["complete"] and ops == [{"op": "add_contact", "type": "rigid", "shape": "sphere", "radius": 20.0, "motion": [0.0, 0.0, -3.0], "friction": 0.2}]
    ops, _, info = rules.parse("바닥에 강체 평면 접촉", base, spec)
    assert info["complete"] and ops[0]["shape"] == "plane" and ops[0]["normal"] == [0.0, 0.0, 1.0]
    ops, _, info = rules.parse("sphere radius 15 push down 2 mm friction 0.2", base, spec)
    assert info["complete"] and ops[0]["radius"] == 15.0 and ops[0]["friction"] == 0.2
    assert not rules.parse("마찰 0.2", base, spec)[2]["complete"]
    sp, _ = apply_ops(spec, [{"op": "add_contact", "type": "rigid", "shape": "sphere", "radius": 20, "motion": [0, 0, -1]}], base)
    assert rules.parse("마찰계수 0.3", base, sp)[0] == [{"op": "update", "id": "ct1", "friction": 0.3}]
    assert rules.parse("ct1 삭제", base, sp)[0] == [{"op": "remove", "id": "ct1"}]
    assert rules.parse("접촉 삭제", base, sp)[0] == [{"op": "clear", "what": "contacts"}]
    assert rules.parse("x 최소 변 구속해줘", base, spec)[0][0]["op"] == "add_bc"


def test_icosphere_is_closed_and_outward():
    v, f = icosphere(2)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    assert (np.einsum("ij,ij->i", n, v[f].mean(axis=1)) > 0).all()


def test_two_blocks_are_tagged_as_bodies():
    base, spec = load_example("contact_blocks")
    assert len(bodies(base)) == 2
    lo = base.nodes[base.patches["body_1"]["nodes"]]
    hi = base.nodes[base.patches["body_2"]["nodes"]]
    assert lo[:, 2].max() <= 10 + 1e-6 and hi[:, 2].min() >= 10 - 1e-6
    tri = base.solid_boundary()
    p = base.nodes[tri]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    top = np.abs(p[:, :, 2] - 20).max(axis=1) < 1e-6
    assert top.any() and (n[top][:, 2] > 0).all()


@ccx
def test_ccx_rigid_sphere_on_plate_matches_plate_theory():
    base, spec = load_example("contact_sphere")
    spec, _ = apply_ops(spec, [{"op": "set_step", "nlgeom": False, "increments": 4}, {"op": "update", "id": "ct1", "friction": 0.0}], base)
    res = calculix.solve(active(base, spec), base, spec)
    D = 210000 * 2.0**3 / (12 * (1 - 0.09))
    P = 3.0 * D / (0.0056 * 200.0**2)
    f = res.history["contact.force"]
    assert len(f) == 4 and abs(f[-1] - P) < 0.08 * P
    assert abs(res.summary["반력 합 (N)"][2] - f[-1]) < 1e-2 * f[-1]
    assert res.rigid and res.rigid[0]["shape"] == "sphere"
    assert res.history["u.max"][-1] < 3.0


@ccx
@tm
def test_pair_contact_with_friction_agrees_between_solvers():
    from femweb.solvers import solid_tm

    base, spec = load_example("contact_blocks")
    spec, _ = apply_ops(spec, [{"op": "set_step", "increments": 2}], base)
    a = solid_tm.solve(base, spec)
    b = calculix.solve(base, base, dict(spec, solver="calculix"))
    fa = [a.history[f"contact.ct1.f{c}"][-1] for c in "xz"]
    fb = [b.history[f"contact.ct1.f{c}"][-1] for c in "xz"]
    assert abs(fa[1] - fb[1]) < 0.03 * abs(fb[1])
    for f in (fa, fb):
        assert 0.27 < abs(f[0] / f[1]) < 0.31


@tm
def test_tm_rigid_plane_presses_block():
    from femweb.solvers import solid_tm

    base, spec = load_example("contact_blocks")
    spec, _ = apply_ops(spec, [{"op": "clear", "what": "all"}, {"op": "add_bc", "type": "pinned", "select": {"plane": {"axis": "z", "at": "min"}}}, {"op": "add_bc", "dofs": [1, 2], "select": {"plane": {"axis": "z", "at": "max"}}}, {"op": "set_step", "increments": 2}, {"op": "add_contact", "type": "pair", "a": {"group": "body_2"}, "b": {"group": "body_1"}}, {"op": "add_contact", "type": "rigid", "shape": "plane", "motion": [0, 0, -0.02]}], base)
    assert np.allclose(spec["contacts"][1]["point"][2], 20.0) and spec["contacts"][1]["normal"] == [0.0, 0.0, -1.0]
    res = solid_tm.solve(base, spec)
    f1 = res.history["contact.ct1.fz"][-1]
    f2 = res.history["contact.ct2.fz"][-1]
    assert f2 < 0 and abs(abs(f1) - abs(f2)) < 0.02 * abs(f2)
    assert abs(res.summary["반력 합 (N)"][2] + f2) < 0.02 * abs(f2)
