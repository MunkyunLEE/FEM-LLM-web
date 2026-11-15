import shutil

import numpy as np
import pytest
from helpers import active, load_example
from femweb.hinges import auto_edges, split_pins
from femweb.io_obj import parse_obj
from femweb.solvers import barhinge, calculix
from femweb.spec import apply_ops, default_spec

ccx = pytest.mark.skipif(calculix.find_ccx() is None, reason="ccx 없음")


def chain():
    return parse_obj(open("examples/panel_chain.obj").read(), "panel_chain.obj")


def test_auto_edges_modes():
    m = chain()
    assert len(auto_edges(m, "groups")) == 6
    assert len(auto_edges(m, "crease")) == 0
    assert len(auto_edges(m, "auto")) == 6
    inner = sum(1 for fl in m.shell_edges().values() if len(fl) == 2)
    assert len(auto_edges(m, "all")) == inner
    base, _ = load_example("miura")
    plain = parse_obj(open("examples/miura.obj").read(), "miura.obj")
    plain.hinges = {}
    assert len(auto_edges(plain, "crease", 5.0)) == sum(len(v) for v in base.hinges.values())


def test_split_pins_duplicates_nodes_along_the_joint():
    base, spec = load_example("panel_chain")
    m, ties, pairs, names = split_pins(base, spec)
    assert names == ["hinge_auto"] and len(ties) == 8 and len(m.nodes) == len(base.nodes) + 8
    assert all(np.allclose(m.nodes[a], m.nodes[b]) for a, b in ties)
    assert len(pairs) == 6
    spec["hinges"]["*"]["pin"] = False
    assert split_pins(base, spec)[1] == []


@ccx
def test_pin_joint_chain_matches_hinged_beam_theory():
    base, spec = load_example("panel_chain")
    res = calculix.solve(active(base, spec), base, spec)
    E, I, P, L = 210000.0, 60 * 2.0**3 / 12, 100.0, 100.0
    beam = (P / 2) * L**3 / (3 * E * I) + P * L**3 / (48 * E * I)
    assert beam * (1 - 0.3**2) * 0.97 < res.summary["최대 변위 (mm)"] < beam * 1.03
    assert abs(res.summary["반력 합 (N)"][2] - 100) < 0.1
    rigid, _ = apply_ops(spec, [{"op": "auto_hinge", "mode": "off"}])
    plain = chain()
    r2 = calculix.solve(active(plain, rigid), plain, rigid)
    assert r2.summary["최대 변위 (mm)"] < 0.85 * res.summary["최대 변위 (mm)"]


@ccx
def test_miura_pin_joint_folds_like_bar_hinge_mechanism():
    base, spec = load_example("miura")
    ops = [{"op": "set_hinge", "group": "*", "pin": True}, {"op": "set_divisions", "divisions": 2}, {"op": "update", "id": "bc4", "value": -5.0}, {"op": "set_step", "increments": 5}]
    s1, _ = apply_ops(spec, ops)
    r1 = barhinge.solve(base, s1)
    s2, _ = apply_ops(s1, [{"op": "set_solver", "solver": "calculix"}])
    r2 = calculix.solve(active(base, s2), base, s2)
    u1, u2 = r1.summary["최대 변위 (mm)"], r2.summary["최대 변위 (mm)"]
    assert abs(u1 - u2) / u1 < 0.02
    assert abs(r2.history["reaction.x"][-1]) < 1e-3
