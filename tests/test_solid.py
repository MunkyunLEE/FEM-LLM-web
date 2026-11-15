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
