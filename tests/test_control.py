import numpy as np
import pytest
from helpers import active, load_example
from femweb import rules
from femweb.hinges import with_auto
from femweb.io_obj import parse_obj, refine
from femweb.solvers import barhinge, calculix
from femweb.solvers.barhinge import Truss
from femweb.spec import apply_ops, default_spec

ccx = pytest.mark.skipif(not calculix.find_ccx(), reason="ccx not installed")


def strip(n=8, L=200.0, w=20.0):
    out = []
    for j in (0, 1):
        for i in range(n + 1):
            out.append(f"v {L * i / n} {w * j} 0")
    for i in range(n):
        out.append(f"f {i + 1} {i + 2} {n + i + 3} {n + i + 2}")
    return "\n".join(out) + "\n"


def column(solver, extra, load=-400.0):
    base = parse_obj(strip(), "strip.obj")
    ops = [
        {"op": "set_solver", "solver": solver},
        {"op": "set_section", "thickness": 2.0},
        {"op": "add_bc", "dofs": [1, 2, 3], "select": {"plane": {"axis": "x", "at": "min"}}},
        {"op": "add_bc", "dofs": [2, 3], "select": {"plane": {"axis": "x", "at": "max"}}},
        {"op": "add_load", "type": "force", "vector": [load, 0, 0], "select": {"plane": {"axis": "x", "at": "max"}}},
    ] + extra
    spec, _ = apply_ops(default_spec(base), ops, None)
    return with_auto(base, spec), spec


GABLE = "v 0 0 0\nv 100 0 6\nv 200 0 0\nv 0 20 0\nv 100 20 6\nv 200 20 0\ng left\nf 1 2 5 4\ng right\nf 2 3 6 5\n"


def gable(extra):
    base = parse_obj(GABLE, "gable.obj")
    ops = [
        {"op": "set_divisions", "divisions": 4},
        {"op": "set_section", "thickness": 2.0},
        {"op": "add_bc", "dofs": [1, 2, 3], "select": {"plane": {"axis": "x", "at": "min"}}},
        {"op": "add_bc", "dofs": [1, 2, 3], "select": {"plane": {"axis": "x", "at": "max"}}},
        {"op": "add_load", "type": "force", "vector": [0, 0, -100], "select": {"plane": {"axis": "x", "at": 100}}},
    ] + extra
    spec, _ = apply_ops(default_spec(base), ops, None)
    base = with_auto(base, spec)
    return refine(base, spec["divisions"]), base, spec


def test_legacy_control_becomes_forced_displacement():
    base, spec = load_example("shell_panel")
    nb, nl = len(spec["bcs"]), len(spec["loads"])
    sp, log = apply_ops(spec, [{"op": "set_step", "type": "static", "control_disp": 2.0}], base)
    assert "control" not in sp["step"] and len(sp["loads"]) == nl - 1 and len(sp["bcs"]) == nb + 1
    assert abs(sp["bcs"][-1]["value"]) == 2.0 and any("강제 변위" in x for x in log)
    sp2, _ = apply_ops(sp, [{"op": "set_step", "control_disp": 3.0}], base)
    assert abs(sp2["bcs"][-1]["value"]) == 3.0 and sp2["bcs"][-1]["value"] * sp["bcs"][-1]["value"] > 0
    sp3, log = apply_ops(spec, [{"op": "set_step", "control": "load"}], base)
    assert sp3["loads"] == spec["loads"] and any("없어졌습니다" in x for x in log)
    old = dict(spec, step=dict(spec["step"], control="displacement", control_disp=1.0, control_kind="average"))
    sp4, _ = apply_ops(old, [], base)
    assert "control" not in sp4["step"] and abs(sp4["bcs"][-1]["value"]) == 1.0


def test_rules_control_sentences():
    base, spec = load_example("shell_panel")
    ops, _, info = rules.parse("변위제어 5mm로 비선형 해석", base, spec)
    assert info["complete"] and ops == [{"op": "set_step", "nlgeom": True, "control": "displacement", "control_disp": 5.0}]
    ops, _, info = rules.parse("강체 압판으로 변위제어 2mm", base, spec)
    assert info["complete"] and ops[0]["control_disp"] == 2.0 and "control_kind" not in ops[0]
    ops, _, info = rules.parse("ld1 기준으로 변위제어 1cm", base, spec)
    assert info["complete"] and ops[0]["control_load"] == "ld1" and ops[0]["control_disp"] == 10.0
    assert rules.parse("하중제어로 바꿔", base, spec)[0] == [{"op": "set_step", "control": "load"}]
    assert rules.parse("변위제어 꺼줘", base, spec)[0] == [{"op": "set_step", "control": "load"}]
    assert rules.parse("분기 전환 꺼줘", base, spec)[0] == [{"op": "set_step", "branch": "primary"}]


def test_dihedral_angle_is_precise_when_flat():
    X = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]], dtype=float)
    tr = Truss(X, [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)], np.ones(5), [(0, 1, 2, 3)], [1.0], [0])
    x = X.copy()
    x[3, 2] = 1e-9
    assert abs(abs(tr.angles(x)[0] - np.pi) - 1e-9) < 1e-12


def test_barhinge_bifurcation_switch_arclength():
    base, spec = column("barhinge", [{"op": "set_step", "type": "arclength", "increments": 30}])
    res = barhinge.solve(base, spec)
    assert any("분기점 감지" in m for m in res.summary["참고"])
    assert res.history["u.max"][-1] > 5.0
    lam = np.array(res.history["load_factor"])
    assert abs(lam[-1] - lam[-10]) < 0.01 * lam[-1]
    base, spec = column("barhinge", [{"op": "set_step", "type": "arclength", "increments": 30, "branch": "primary"}])
    res = barhinge.solve(base, spec)
    assert res.history["u.max"][-1] < 0.5
    assert max(res.history["stability.negative"]) >= 1


def test_barhinge_imperfection_matches_bifurcation_load():
    base, spec = column("barhinge", [{"op": "set_step", "type": "arclength", "increments": 30}])
    crit = barhinge.solve(base, spec).history["load_factor"][-1]
    base, spec = column("barhinge", [{"op": "set_step", "type": "arclength", "increments": 30, "imperfection": 0.2}])
    res = barhinge.solve(base, spec)
    assert any("초기 결함" in m for m in res.summary["참고"])
    lam = res.history["load_factor"]
    assert 0.85 * crit < lam[-1] < 1.02 * crit
    assert res.history["u.max"][-1] > 2.0


def test_barhinge_forced_displacement_records_curve():
    base, spec = column("barhinge", [{"op": "set_step", "type": "static", "increments": 20, "control_disp": 2.0}])
    assert not spec["loads"] and spec["bcs"][-1]["value"] == -2.0
    res = barhinge.solve(base, spec)
    assert abs(res.history["control.disp"][-1] - 2.0) < 1e-6
    assert res.history["u.max"][-1] > 5.0
    assert 70 < res.history["control.force"][-1] < 90


@ccx
def test_ccx_displacement_control_passes_snap_through():
    m, base, spec = gable([{"op": "set_step", "type": "static", "nlgeom": True, "control": "displacement", "control_disp": 14.0}])
    res = calculix.solve(m, base, spec)
    f = np.array(res.history["control.force"])
    assert len(f) == 25
    assert f.min() < 0 < f[1]
    assert f[-1] > f[:10].max()
    assert "극한점" in res.summary
    m, base, spec = gable([{"op": "update", "id": "ld1", "vector": [0, 0, -500]}, {"op": "set_step", "type": "static", "nlgeom": True}])
    with pytest.raises(RuntimeError, match="강제 변위"):
        calculix.solve(m, base, spec)


@ccx
def test_ccx_forced_displacement_linear_stiffness():
    base, spec = load_example("shell_panel")
    spec, _ = apply_ops(spec, [{"op": "set_step", "type": "static", "nlgeom": False}], base)
    m = active(base, spec)
    ref = calculix.solve(m, base, spec)
    spec2, _ = apply_ops(spec, [{"op": "set_step", "control": "displacement", "control_disp": 0.01}], base)
    res = calculix.solve(m, base, spec2)
    k = res.history["control.force"][-1] / res.history["control.disp"][-1]
    assert abs(res.history["control.disp"][-1] - 0.01) < 2e-5
    assert abs(res.history["u.max"][-1] / res.history["control.force"][-1] - ref.history["u.max"][-1] / 1000.0) < 0.01 * ref.history["u.max"][-1] / 1000.0
    assert k > 0


@ccx
def test_ccx_rigid_control_and_arclength_note():
    base, spec = load_example("shell_panel")
    spec, _ = apply_ops(spec, [{"op": "set_step", "type": "static", "nlgeom": True, "increments": 5, "control": "displacement", "control_kind": "rigid", "control_disp": 0.05}], base)
    m = active(base, spec)
    res = calculix.solve(m, base, spec)
    assert len(res.history["control.force"]) == 5 and res.history["control.force"][-1] > 1000
    spec, _ = apply_ops(spec, [{"op": "set_step", "type": "arclength", "control": "load", "increments": 3}], base)
    res = calculix.solve(m, base, spec)
    assert any("호장법이 없어" in n for n in res.summary["참고"])
