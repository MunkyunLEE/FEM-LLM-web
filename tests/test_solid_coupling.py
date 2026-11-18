import numpy as np
import pytest

from femweb.io_stl import stl_to_solid
from femweb.loads import distributed_moment
from femweb.solvers import calculix as C
from femweb.solvers import solid_tm as TM
from femweb.spec import apply_ops, default_spec

E_, L, B, H = 210000.0, 1000.0, 50.0, 100.0
IY = B * H**3 / 12
END = {"plane": {"axis": "x", "at": "max"}}


@pytest.fixture(scope="module")
def beam():
    return stl_to_solid(open("examples/cantilever_solid.stl", "rb").read(), "c.stl", size=28.0)


def _solve(m, solver, ops):
    base = [{"op": "set_solver", "solver": solver}, {"op": "add_bc", "type": "fixed", "select": {"plane": {"axis": "x", "at": "min"}}}]
    sp, _ = apply_ops(default_spec(m), base + ops, m)
    return TM.solve(m, sp) if solver == "tensormesh" else C.solve(m, m, sp)


SOLVERS = ["tensormesh", pytest.param("calculix", marks=pytest.mark.skipif(C.find_ccx() is None, reason="ccx 없음"))]


def test_distributed_moment_has_zero_resultant():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(30, 3)) * 10
    M = np.array([1e5, -2e5, 3e5])
    f = distributed_moment(P, M)
    c = P.mean(0)
    assert np.abs(f.sum(0)).max() < 1e-6 * np.abs(M).max()
    assert np.cross(P - c, f).sum(0) == pytest.approx(M, rel=1e-9)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solid_reference_point_moment(beam, solver):
    r = _solve(beam, solver, [{"op": "add_coupling", "select": END, "moment": [0, 1e7, 0]}])
    assert r.history["rp.cp1.ry"][-1] / (1e7 * L / (E_ * IY)) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solid_face_moment_load(beam, solver):
    r = _solve(beam, solver, [{"op": "add_load", "type": "moment", "vector": [0, 1e7, 0], "select": END}])
    X = r.nodes
    U = r.frames[-1]["U"]
    e = np.abs(X[:, 0] - L) < 1e-6
    top = e & (np.abs(X[:, 2] - X[:, 2].max()) < 1e-6)
    bot = e & (np.abs(X[:, 2] - X[:, 2].min()) < 1e-6)
    rot = (U[bot, 0].mean() - U[top, 0].mean()) / H
    assert abs(rot) / (1e7 * L / (E_ * IY)) == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solid_face_rotation_bc(beam, solver):
    r = _solve(beam, solver, [{"op": "add_bc", "dofs": [5], "value": 0.01, "select": END}])
    my = r.summary["기준점 cp_bc2 모멘트 (N·mm)"][1]
    assert abs(my) / (E_ * IY * 0.01 / L) == pytest.approx(1.0, abs=0.02)
