import json
import os
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import ROOT

from femweb import abaqus, compare, rules, server
from femweb.io_obj import parse_obj
from femweb.solvers import calculix
from femweb.spec import apply_ops, default_spec, imperfections

c = TestClient(server.app)
server.S["llm"]["mode"] = "rule"
ccx = pytest.mark.skipif(calculix.find_ccx() is None, reason="ccx 없음")


def _check(text):
    blocks = re.split(r"^\*(?!\*)", text, flags=re.M)
    defined = set()
    eids = []
    for b in blocks:
        head = b.split("\n", 1)[0]
        for k in ("NSET", "ELSET", "NAME"):
            mm = re.search(k + r"=([A-Z0-9_]+)", head)
            if mm and (head.startswith(("NSET", "ELSET", "SURFACE", "ELEMENT", "NODE,")) or k == "NAME"):
                defined.add(mm.group(1))
        if head.startswith("ELEMENT,"):
            eids += [int(l.split(",")[0]) for l in b.splitlines()[1:] if l.strip()]
    assert len(eids) == len(set(eids))
    used = set(re.findall(r"(?:ELSET|NSET|MATERIAL|AMPLITUDE)=([A-Z0-9_]+)", text))
    for b in blocks:
        head = b.split("\n", 1)[0]
        if head.startswith(("BOUNDARY", "KINEMATIC COUPLING", "CONTACT PROPERTY ASSIGNMENT", "CONTACT INCLUSIONS", "SURFACE")):
            for l in b.splitlines()[1:]:
                for t in [x.strip() for x in l.split(",")]:
                    if re.fullmatch(r"[A-Z][A-Z0-9_]*", t) and t not in {"ENCASTRE", "PINNED", "XSYMM", "YSYMM", "ZSYMM", "XASYMM", "YASYMM", "ZASYMM", "SPOS", "SNEG"}:
                        used.add(t)
    mats = set(re.findall(r"\*MATERIAL, NAME=([A-Z0-9_]+)", text))
    missing = {u for u in used if u not in defined and u not in mats}
    assert not missing, missing
    assert text.count("*STEP") == 1 and text.count("*END STEP") == 1


def test_every_example_exports():
    for ex in c.get("/api/examples").json():
        assert c.post(f"/api/examples/{ex['name']}").status_code == 200
        r = c.get("/api/export/abaqus")
        assert r.status_code == 200, (ex["name"], r.text)
        _check(r.text)


def test_export_uses_solver_coordinates_and_damping():
    p = json.load(open(os.path.join(ROOT, "examples/presets.json")))["tube_platen"]
    m = parse_obj(open(os.path.join(ROOT, "examples", p["file"])).read(), "t.obj")
    sp, _ = apply_ops(default_spec(m), p["ops"], m)
    X = m.nodes + 0.1
    text, notes = abaqus.export(m, sp, info={"X": X, "alpha": 12.5, "beta": 3e-6, "time": 0.65, "f1": 300.0, "f2": 500.0})
    assert "*DAMPING, ALPHA=12.5, BETA=3e-06" in text and "*DYNAMIC, APPLICATION=QUASI-STATIC" in text
    line = next(l for l in text.splitlines() if l.startswith("1, "))
    assert np.allclose([float(v) for v in line.split(",")[1:]], X[0], atol=1e-6)
    assert "*RIGID BODY" in text and "RP_CT1" in text and "R3D4" in text
    text2, notes2 = abaqus.export(m, sp)
    assert "*STATIC" in text2 and any("감쇠 계수" in n for n in notes2)


def test_export_rigid_part_coupling_and_symmetry():
    m = parse_obj(open(os.path.join(ROOT, "examples/four_point_tube.obj")).read(), "f.obj")
    ops = [{"op": "set_solver", "solver": "shell"}, {"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": "load_l"}, "motion": [0, 0, -5]}, {"op": "add_bc", "type": "xsymm", "select": {"and": [{"group": "tube"}, {"plane": {"axis": "x", "at": 0}}]}}, {"op": "add_coupling", "select": {"group": "support_r"}, "rigid": True}]
    sp, _ = apply_ops(default_spec(m), ops, m)
    text, _ = abaqus.export(m, sp)
    _check(text)
    assert ", XSYMM" in text and "*KINEMATIC COUPLING" in text
    n_load = sum(1 for g in m.shell_group if g == "load_l")
    cnt = sum(len([l for l in part.split("*")[0].splitlines()[1:] if l.strip()]) for part in text.split("ELSET=ER_CT1")[1:] if part.startswith("\n"))
    assert cnt == n_load


def test_parse_csv_and_rpt():
    cs = compare.parse("disp,force\n0,0\n1,10\n2,15\n", "a")
    assert cs[0]["name"] == "force" and cs[0]["y"] == [0.0, 10.0, 15.0]
    rpt = "\n\n           X           RF3_RP  ALLSE\n\n            0.             0.   0.\n           0.5        -100.   1.\n            1.        -150.   2.\n"
    cs = compare.parse(rpt, "job")
    assert [x["name"] for x in cs] == ["RF3_RP", "ALLSE"] and cs[0]["y"][2] == -150.0
    with pytest.raises(ValueError):
        compare.parse("no numbers here", "x")


def test_metrics_and_sign():
    x = np.linspace(0, 10, 51)
    y = np.sin(x) + 2
    m = compare.metrics(x, y, x[::2], y[::2])
    assert m["peak_ratio"] == pytest.approx(1.0, abs=1e-3) and m["mean_ratio"] == pytest.approx(1.0, abs=2e-3) and m["rms"] < 5e-3
    m = compare.metrics(x, 1.1 * y, x, y)
    assert m["mean_ratio"] == pytest.approx(1.1)
    assert compare.auto_sign(y, -y) == -1.0


@ccx
def test_ref_overlay_api():
    assert c.post("/api/examples/cantilever_beam").status_code == 200
    s = c.post("/api/chat", json={"text": "증분 4, 비선형으로"}).json()
    c.post("/api/ops", json={"ops": [{"op": "add_plot", "title": "끝 처짐", "x": {"quantity": "time"}, "series": [{"quantity": "u.max"}]}]})
    assert c.post("/api/solve").status_code == 200
    pid = c.get("/api/state").json()["spec"]["plots"][-1]["id"]
    q = next(p for p in c.get("/api/plots").json() if p["id"] == pid)
    xs, ys = q["series"][0]["x"], q["series"][0]["y"]
    text = "LF, U\n" + "\n".join(f"{a}, {-1.02 * b}" for a, b in zip(xs, ys))
    r = c.post("/api/refs", json={"plot": pid, "text": text, "name": "abq"})
    assert r.status_code == 200 and r.json()[0]["flipped"]
    q = next(p for p in c.get("/api/plots").json() if p["id"] == pid)
    assert q["refs"][0]["metrics"]["peak_ratio"] == pytest.approx(1 / 1.02, rel=1e-6)
    rid = q["refs"][0]["id"]
    assert c.post("/api/refs/update", json={"id": rid, "remove": True}).status_code == 200
    assert next(p for p in c.get("/api/plots").json() if p["id"] == pid)["refs"] == []


def test_rules_and_spec_multi_mode_imperfection():
    m = parse_obj(open(os.path.join(ROOT, "examples/square_tube_fine.obj")).read(), "t.obj")
    sp = default_spec(m)
    ops = rules.parse("초기 결함 1차 0.2 mm, 2차 0.1 mm", m, sp)[0]
    assert ops == [{"op": "set_step", "imperfection_modes": [[1, 0.2], [2, 0.1]], "imperfection": 0.0}]
    assert rules.parse("초기 결함 모드 1~3 에 0.1 mm", m, sp)[0][0]["imperfection_modes"] == [[1, 0.1], [2, 0.1], [3, 0.1]]
    assert rules.parse("초기 결함 모드 1,2 에 각각 0.2, 0.05 mm", m, sp)[0][0]["imperfection_modes"] == [[1, 0.2], [2, 0.05]]
    sp2, _ = apply_ops(sp, ops, m)
    assert imperfections(sp2["step"]) == [[1, 0.2], [2, 0.1]]
    sp3, _ = apply_ops(sp2, [{"op": "set_step", "imperfection": 0.3}], m)
    assert imperfections(sp3["step"]) == [[1, 0.3]]


def test_shell_multi_mode_imperfection_is_sum_of_modes():
    from femweb.solvers import shell

    p = json.load(open(os.path.join(ROOT, "examples/presets.json")))["tube_crush"]
    m = parse_obj(open(os.path.join(ROOT, "examples", p["file"])).read(), "t.obj")
    base = [o for o in p["ops"] if o["op"] != "set_step"] + [{"op": "set_step", "increments": 1, "control_disp": 0.02}]
    d = {}
    for tag, extra in (("a", {"imperfection_modes": [[1, 0.2]]}), ("b", {"imperfection_modes": [[2, 0.1]]}), ("ab", {"imperfection_modes": [[1, 0.2], [2, 0.1]]})):
        sp, _ = apply_ops(default_spec(m), base + [dict(op="set_step", **extra)], m)
        r = shell.solve(m, sp)
        d[tag] = r.abaqus["X"] - m.nodes
    assert np.abs(d["a"]).max() == pytest.approx(0.2, rel=1e-6) and np.abs(d["b"]).max() == pytest.approx(0.1, rel=1e-6)
    assert np.abs(d["ab"] - (d["a"] + d["b"])).max() < 1e-6
    assert d["a"].ravel()[np.argmax(np.abs(d["a"]))] > 0


@ccx
def test_calculix_multi_mode_modes():
    p = json.load(open(os.path.join(ROOT, "examples/presets.json")))["shell_panel"]
    m = parse_obj(open(os.path.join(ROOT, "examples", p["file"])).read(), "p.obj")
    sp, _ = apply_ops(default_spec(m), p["ops"], m)
    modes = calculix.buckle_mode(m, m, sp, k=3)
    assert len(modes) == 3 and all(np.abs(q).max() == pytest.approx(1.0) for q in modes)
    assert abs(float((modes[0] * modes[1]).sum())) < 0.9 * float((modes[0] ** 2).sum())
