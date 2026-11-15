import json
import os
import pytest
from helpers import ROOT
from fastapi.testclient import TestClient
from femweb import server
from femweb.solvers import calculix

c = TestClient(server.app)
server.S["llm"]["mode"] = "rule"
ccx = pytest.mark.skipif(calculix.find_ccx() is None, reason="ccx 없음")


def post_ops(ops):
    r = c.post("/api/ops", json={"ops": ops})
    assert r.status_code == 200, r.text
    return r.json()


def test_examples_listed():
    names = [e["name"] for e in c.get("/api/examples").json()]
    assert names == ["cantilever_beam", "shell_panel", "cylinder_shell", "miura", "panel_chain", "cantilever_solid", "cylinder_solid"]


@ccx
def test_cantilever_by_chat_matches_beam_theory():
    assert c.post("/api/examples/cantilever_beam?preset=0").status_code == 200
    s = c.get("/api/state").json()
    assert s["spec"]["bcs"] == [] and s["spec"]["sections"]["*"]["beam"]["h"] == 100
    r = c.post("/api/chat", json={"text": "x 최소인 끝을 완전 고정해줘"}).json()
    assert r["chat"]["engine"] == "rule" and len(r["spec"]["bcs"]) == 1
    r = c.post("/api/chat", json={"text": "x 최대 끝단에 z 방향으로 -1000 N 하중 줘"}).json()
    assert r["preview"]["loads"][0]["total"] == [0.0, 0.0, -1000.0]
    assert r["warnings"] == []
    r = c.post("/api/solve")
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert abs(res["summary"]["최대 변위 (mm)"] - 0.381) < 0.01
    assert abs(res["summary"]["반력 합 (N)"][2] - 1000) < 1
    f = c.get("/api/result/frame?field=vonMises").json()
    assert 10 < f["max"] < 13
    mesh = c.get("/api/result/mesh").json()
    assert len(mesh["nodes"]) == len(f["values"]) and len(mesh["tris"]) > 0


@ccx
def test_divisions_keep_load_total_and_answer():
    c.post("/api/examples/cantilever_beam?preset=1")
    s = post_ops([{"op": "set_divisions", "divisions": 3}])
    assert s["model"]["counts"]["beams"] == 30
    assert s["preview"]["loads"][0]["total"] == [0.0, 0.0, -1000.0]
    res = c.post("/api/solve").json()["result"]
    assert abs(res["summary"]["최대 변위 (mm)"] - 0.381) < 0.01


@ccx
def test_panel_buckling_factor_near_theory():
    c.post("/api/examples/shell_panel?preset=1")
    res = c.post("/api/solve").json()["result"]
    assert res["kind"] == "buckle" and len(res["frames"]) == 4
    assert abs(res["summary"]["좌굴 계수"][0] - 30.37) / 30.37 < 0.03


def test_miura_plot_and_patch_flow():
    c.post("/api/examples/miura?preset=1")
    s = post_ops([{"op": "set_step", "increments": 6}])
    assert s["spec"]["solver"] == "barhinge"
    res = c.post("/api/solve").json()["result"]
    n = len(res["frames"])
    assert n >= 6 and abs(res["frames"][-1]["time"] - 1.0) < 1e-9
    r = c.post("/api/chat", json={"text": "접힘각 그래프 그려줘"}).json()
    assert r["stale"] is False
    p = c.get("/api/plots").json()
    assert len(p) == 2 and len(p[0]["series"]) == 4 and len(p[0]["series"][0]["x"]) == n
    assert p[1]["series"][0]["y"][-1] > p[1]["series"][0]["y"][0]
    s = post_ops([{"op": "update", "id": "bc4", "value": -10.0}])
    assert s["stale"] is True
    assert [b["value"] for b in s["spec"]["bcs"] if b["id"] == "bc4"] == [-10.0]


def test_bad_requests_are_reported():
    c.post("/api/examples/miura?preset=1")
    r = c.post("/api/ops", json={"ops": [{"op": "add_bc", "type": "fixed", "select": {"group": "nope"}}]})
    assert r.status_code == 400 and "nope" in r.json()["detail"]
    r = c.post("/api/chat", json={"text": "하중 줘"}).json()
    assert r["chat"]["ops"] == []
    r = c.post("/api/upload", files={"file": ("open.stl", b"solid a\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid a\n")})
    assert r.status_code == 400 and "닫힌" in r.json()["detail"]


def test_upload_obj_and_export():
    data = open(os.path.join(ROOT, "examples", "shell_panel.obj"), "rb").read()
    r = c.post("/api/upload", files={"file": ("my_panel.obj", data)})
    assert r.status_code == 200
    assert r.json()["model"]["counts"]["shells"] == 100
    post_ops([{"op": "add_bc", "type": "fixed", "select": {"plane": {"axis": "x", "at": "min"}}}, {"op": "add_load", "type": "pressure", "value": 0.01, "select": {"all": True}}])
    inp = c.get("/api/export/inp").text
    assert "*SHELL SECTION" in inp and "*CLOAD" in inp and "S8R" in inp


@ccx
def test_cantilever_modal_frequencies():
    c.post("/api/examples/cantilever_beam?preset=1")
    post_ops([{"op": "set_step", "type": "modal", "modes": 3}])
    res = c.post("/api/solve").json()["result"]
    f = res["summary"]["고유진동수 (Hz)"]
    assert len(res["frames"]) == 3
    assert abs(f[0] - 41.8) / 41.8 < 0.03 and abs(f[1] - 83.5) / 83.5 < 0.03


def test_default_increments_is_25():
    c.post("/api/examples/cantilever_beam?preset=0")
    assert c.get("/api/state").json()["spec"]["step"]["increments"] == 25


def test_cancel_endpoints_without_active_job():
    assert c.post("/api/cancel/solve").json() == {"cancelled": False}
    assert c.post("/api/cancel/llm").json() == {"cancelled": False}


def test_solve_can_be_cancelled():
    import threading
    import time

    c.post("/api/examples/cantilever_solid?preset=0")
    post_ops([{"op": "set_solver", "solver": "calculix"}, {"op": "set_divisions", "divisions": 1}])
    out = {}

    def run():
        out["r"] = c.post("/api/solve")

    th = threading.Thread(target=run)
    th.start()
    for _ in range(100):
        time.sleep(0.1)
        if c.post("/api/cancel/solve").json()["cancelled"]:
            break
    th.join(timeout=30)
    assert out["r"].status_code == 409 and "중단" in out["r"].json()["detail"]
    assert c.post("/api/solve").status_code in (200, 422)


@ccx
def test_buckling_reports_critical_load_and_requires_load():
    c.post("/api/examples/shell_panel?preset=1")
    res = c.post("/api/solve").json()["result"]
    s = res["summary"]
    assert s["가한 기준 하중 (N)"] == 1000.0
    assert abs(s["임계 하중 (N) = 계수 × 기준 하중"][0] - 30090.83) < 50
    c.post("/api/examples/shell_panel?preset=1")
    post_ops([{"op": "clear", "what": "loads"}])
    r = c.post("/api/solve")
    assert r.status_code == 422 and "압축 하중" in r.json()["detail"]


@ccx
def test_result_mesh_has_face_edges_without_quad_diagonals():
    c.post("/api/examples/shell_panel?preset=1")
    c.post("/api/solve")
    m = c.get("/api/result/mesh").json()
    assert len(m["fedges"]) > 0 and len(m["fedges"]) < len(m["tris"]) * 3
    assert len(m["fedges"]) == 480
