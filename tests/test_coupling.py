import numpy as np
import pytest

from femweb import rules
from femweb.io_obj import parse_obj
from femweb.solvers import calculix as C
from femweb.solvers import shell as S
from femweb.solvers import shell_quad as Qd
from femweb.spec import apply_ops, default_spec

R, L, T, E, NU = 50.0, 200.0, 2.0, 210000.0, 0.3
I = np.pi * R**3 * T


def _cyl():
    return parse_obj(open("examples/cylinder_shell.obj").read(), "c.obj")


def _ops(solver, upd):
    return [
        {"op": "set_solver", "solver": solver},
        {"op": "set_section", "thickness": T},
        {"op": "add_bc", "type": "fixed", "select": {"plane": {"axis": "z", "at": "min"}}},
        {"op": "add_coupling", "select": {"plane": {"axis": "z", "at": "max"}}},
        {"op": "set_step", "type": "static", "increments": 2},
        {"op": "update", "id": "cp1", **upd},
    ]


def test_rules_create_and_drive_reference_point():
    m = _cyl()
    sp = default_spec(m)
    ops = rules.parse("위 끝면을 기준점에 묶어줘", m, sp)[0]
    assert ops == [{"op": "add_coupling", "select": {"plane": {"axis": "z", "at": "max"}}}]
    sp, _ = apply_ops(sp, ops, m)
    assert sp["couplings"][0]["ref"] == pytest.approx([0.0, 0.0, 200.0], abs=1e-6)
    assert rules.parse("기준점에 x축 모멘트 2 kN·m", m, sp)[0] == [{"op": "update", "id": "cp1", "moment": [2e6, 0.0, 0.0]}]
    assert rules.parse("cp1 z 방향 -5 mm", m, sp)[0] == [{"op": "update", "id": "cp1", "fix": {"3": -5.0}}]
    assert rules.parse("기준점 x축 0.1 rad 회전", m, sp)[0] == [{"op": "update", "id": "cp1", "fix": {"4": 0.1}}]
    assert rules.parse("기준점 회전 구속", m, sp)[0] == [{"op": "update", "id": "cp1", "fix": {"4": 0.0, "5": 0.0, "6": 0.0}}]
    sp, _ = apply_ops(sp, [{"op": "update", "id": "cp1", "fix": {"1": 0}}, {"op": "update", "id": "cp1", "fix": {"3": -1}}], m)
    assert sp["couplings"][0]["fix"] == {"1": 0.0, "3": -1.0}
    sp, _ = apply_ops(sp, [{"op": "remove", "id": "cp1"}], m)
    assert sp["couplings"] == []


def test_shell_coupling_end_moment_matches_beam_theory():
    m = _cyl()
    sp, _ = apply_ops(default_spec(m), _ops("shell", {"moment": [1e7, 0, 0]}), m)
    r = S.solve(m, sp)
    assert r.history["rp.cp1.rx"][-1] == pytest.approx(1e7 * L / (E * I), rel=0.01)
    assert r.history["rp.cp1.mx"][-1] == pytest.approx(1e7, rel=1e-6)
    assert abs(r.summary["에너지 균형 오차 (총 에너지 / 외부 일 - 1)"]) < 1e-3


def test_shell_coupling_prescribed_rotation_reaction():
    m = _cyl()
    sp, _ = apply_ops(default_spec(m), _ops("shell", {"fix": {"4": 0.01}}), m)
    r = S.solve(m, sp)
    assert r.history["rp.cp1.mx"][-1] == pytest.approx(E * I * 0.01 / L, rel=0.01)


def test_shell_coupling_finite_rigid_rotation():
    X, Q = Qd.qtube(40.0, 60.0, 4, 6)
    txt = "\n".join([f"v {x:g} {y:g} {z:g}" for x, y, z in X] + [f"f {' '.join(str(i + 1) for i in q)}" for q in Q]) + "\n"
    m = parse_obj(txt, "t.obj")
    ops = [
        {"op": "set_solver", "solver": "shell"},
        {"op": "set_section", "thickness": 1.0},
        {"op": "add_coupling", "select": {"plane": {"axis": "z", "at": "max"}}, "fix": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 1.0}},
        {"op": "set_step", "type": "static", "increments": 4},
    ]
    sp, _ = apply_ops(default_spec(m), ops, m)
    r = S.solve(m, sp)
    ref = np.array(sp["couplings"][0]["ref"])
    exp = ref + (X - ref) @ S.rotation([0, 0, 1.0]).T
    assert np.abs(X + r.frames[-1]["U"] - exp).max() < 1e-8
    assert r.history["energy.strain"][-1] < 1e-12


@pytest.mark.skipif(C.find_ccx() is None, reason="ccx 없음")
def test_calculix_rigid_body_coupling():
    m = _cyl()
    sp, _ = apply_ops(default_spec(m), _ops("calculix", {"moment": [1e7, 0, 0]}), m)
    r = C.solve(m, m, sp)
    assert r.history["rp.cp1.rx"][-1] == pytest.approx(1e7 * L / (E * I), rel=0.01)
    assert r.summary["기준점 cp1 모멘트 (N·mm)"][0] == pytest.approx(1e7, rel=1e-4)
    sp, _ = apply_ops(default_spec(m), _ops("calculix", {"fix": {"3": -0.1}}), m)
    r = C.solve(m, m, sp)
    assert r.history["rp.cp1.fz"][-1] == pytest.approx(-E * 2 * np.pi * R * T * 0.1 / L, rel=0.01)
