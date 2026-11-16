import importlib.util

import pytest
from helpers import load_example

tm = pytest.mark.skipif(importlib.util.find_spec("tensormesh") is None, reason="tensormesh 없음")


@tm
def test_solid_cantilever_matches_beam_theory():
    from femweb.solvers import solid_tm

    base, spec = load_example("cantilever_solid")
    res = solid_tm.solve(base, spec)
    assert abs(res.summary["최대 변위 (mm)"] - 0.381) < 0.01
    assert abs(res.summary["반력 합 (N)"][2] - 1000) < 1e-3
    assert abs(res.summary["변형에너지 (N·mm)"] - 0.5 * 1000 * 0.381) < 3


@tm
def test_solid_displacement_control_recovers_load():
    from femweb.solvers import solid_tm
    from femweb.spec import apply_ops

    base, spec = load_example("cantilever_solid")
    ref = solid_tm.solve(base, spec)
    for kind in ("average", "rigid"):
        sp, _ = apply_ops(spec, [{"op": "set_step", "control": "displacement", "control_kind": kind, "control_disp": 0.38}], base)
        res = solid_tm.solve(base, sp)
        assert abs(res.history["control.disp"][-1] - 0.38) < 1e-9
        assert abs(res.history["control.force"][-1] - 1000) < 20
        assert abs(res.history["u.max"][-1] - ref.history["u.max"][-1]) < 0.01


@tm
def test_solid_nonlinear_displacement_control_runs():
    from femweb.solvers import solid_tm
    from femweb.spec import apply_ops

    base, spec = load_example("cantilever_solid")
    sp, _ = apply_ops(spec, [{"op": "set_step", "nlgeom": True, "increments": 2, "control": "displacement", "control_disp": 0.38}], base)
    res = solid_tm.solve(base, sp)
    assert len(res.history["control.force"]) == 2
    assert abs(res.history["control.force"][-1] - 1000) < 30
