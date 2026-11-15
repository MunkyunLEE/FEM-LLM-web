import pytest
from helpers import load_example
from femweb import rules
from femweb.spec import apply_ops, default_spec

MODELS = {}


def model(name):
    if name not in MODELS:
        base, spec = load_example(name)
        MODELS[name] = (base, spec, default_spec(base))
    return MODELS[name]


def run(name, text, preset=False):
    base, spec, empty = model(name)
    use = spec if preset else empty
    ops, reply, info = rules.parse(text, base, use)
    apply_ops(use, ops, base)
    return ops, reply, info


XMIN = {"plane": {"axis": "x", "at": "min"}}
XMAX = {"plane": {"axis": "x", "at": "max"}}
ZMIN = {"plane": {"axis": "z", "at": "min"}}
ZMAX = {"plane": {"axis": "z", "at": "max"}}

BC_CASES = [
    ("cantilever_beam", "x 최소인 끝을 완전 고정해줘", {"type": "fixed", "select": XMIN}),
    ("cantilever_beam", "왼쪽 끝 고정", {"type": "fixed", "select": XMIN}),
    ("cantilever_beam", "clamp x min", {"type": "fixed", "select": XMIN}),
    ("cantilever_beam", "x 최소 면을 핀 지지", {"type": "pinned", "select": XMIN}),
    ("cantilever_beam", "양단 단순지지", {"type": "pinned", "select": {"or": [XMIN, XMAX]}}),
    ("cantilever_beam", "오른쪽 끝은 롤러", {"dofs": [3], "select": XMAX}),
    ("cantilever_beam", "x 최소 끝을 x, y 방향만 구속", {"dofs": [1, 2], "select": XMIN}),
    ("cantilever_beam", "왼쪽 끝 ux uy uz 구속", {"dofs": [1, 2, 3], "select": XMIN}),
    ("cantilever_beam", "왼쪽 끝을 x축 회전만 구속", {"dofs": [4], "select": XMIN}),
    ("cantilever_beam", "x 최소 끝은 x 방향만 자유롭게 하고 나머지 구속", {"dofs": [2, 3], "select": XMIN}),
    ("cantilever_beam", "x 최대 끝을 z 방향으로 -5 mm 강제변위", {"dofs": [3], "value": -5.0, "select": XMAX}),
    ("cantilever_beam", "x 최대 끝을 2 mm 압축", {"dofs": [1], "value": -2.0, "select": XMAX}),
    ("cantilever_beam", "x 최대 끝을 1 mm 당겨줘", {"dofs": [1], "value": 1.0, "select": XMAX}),
    ("cantilever_beam", "prescribed displacement of -2 mm in x at x max", {"dofs": [1], "value": -2.0, "select": XMAX}),
    ("shell_panel", "네 변 모두 z 방향만 구속해줘", {"dofs": [3], "select": {"boundary": True}}),
    ("shell_panel", "테두리 단순지지", {"type": "pinned", "select": {"boundary": True}}),
    ("shell_panel", "네 모서리 고정", {"type": "fixed", "select": {"corners": True}}),
    ("shell_panel", "y 최소 변에 y 대칭", {"type": "ysymm", "select": {"plane": {"axis": "y", "at": "min"}}}),
    ("shell_panel", "yz 평면 대칭", {"type": "xsymm"}),
    ("shell_panel", "-x 변 고정", {"type": "fixed", "select": XMIN}),
    ("cylinder_shell", "바닥 고정", {"type": "fixed", "select": ZMIN}),
    ("cylinder_shell", "위쪽 끝은 x, y 방향만 구속", {"dofs": [1, 2], "select": ZMAX}),
    ("cylinder_shell", "상단을 2mm 압축해줘", {"dofs": [3], "value": -2.0, "select": ZMAX}),
    ("cantilever_solid", "face_1 고정", {"type": "fixed", "select": {"group": "face_1"}}),
]


@pytest.mark.parametrize("name,text,want", BC_CASES)
def test_bc_sentences(name, text, want):
    ops, reply, info = run(name, text)
    assert info["complete"], reply
    assert len(ops) == 1 and ops[0]["op"] == "add_bc"
    for k, v in want.items():
        assert ops[0].get(k) == pytest.approx(v) if k == "value" else ops[0].get(k) == v, (k, ops[0])


LOAD_CASES = [
    ("cantilever_beam", "x 최대 끝단에 z 방향으로 -1000 N 하중 줘", "force", [0, 0, -1000], XMAX),
    ("cantilever_beam", "끝단에 아래로 500 N", "force", [0, 0, -500], XMAX),
    ("cantilever_beam", "자유단에 -z 방향으로 1.5kN 걸어줘", "force", [0, 0, -1500], XMAX),
    ("cantilever_beam", "오른쪽 끝에 -2000 N z", "force", [0, 0, -2000], XMAX),
    ("cantilever_beam", "오른쪽 끝에 수직하중 2kN", "force", [0, 0, -2000], XMAX),
    ("cantilever_beam", "끝단에 (0, 100, -200) N", "force", [0, 100, -200], XMAX),
    ("cantilever_beam", "apply -500 N in z direction at x max", "force", [0, 0, -500], XMAX),
    ("cantilever_beam", "x 최대 끝에 y축 모멘트 500 N·m", "moment", [0, 500000, 0], XMAX),
    ("cantilever_beam", "오른쪽 끝에 위로 300 kgf", "force", [0, 0, 300 * 9.80665], XMAX),
    ("shell_panel", "x 최대 변에 1000 N 압축", "force", [-1000, 0, 0], XMAX),
    ("shell_panel", "+x 면에 100 N 인장", "force", [100, 0, 0], XMAX),
    ("shell_panel", "x 최대 변에 선하중 10 N/mm 아래로", "force", [0, 0, -2000], XMAX),
    ("cylinder_shell", "위에서 1톤으로 눌러줘", "force", [0, 0, -9806.65], ZMAX),
    ("cylinder_shell", "위에 100 kg 올려줘", "force", [0, 0, -980.665], ZMAX),
    ("cylinder_shell", "윗면에 100 kN 압축", "force", [0, 0, -100000], ZMAX),
]


@pytest.mark.parametrize("name,text,typ,vec,sel", LOAD_CASES)
def test_load_sentences(name, text, typ, vec, sel):
    ops, reply, info = run(name, text)
    assert info["complete"], reply
    assert len(ops) == 1 and ops[0]["op"] == "add_load" and ops[0]["type"] == typ
    assert ops[0]["vector"] == pytest.approx(vec)
    assert ops[0]["select"] == sel


def test_pressure_and_gravity():
    ops, _, info = run("shell_panel", "전체에 등분포 하중 2 kPa 아래로")
    assert info["complete"] and ops[0]["type"] == "pressure" and ops[0]["value"] == pytest.approx(0.002) and ops[0]["direction"] == [0, 0, -1]
    ops, _, info = run("cylinder_shell", "전체에 압력 1 bar")
    assert ops[0]["value"] == pytest.approx(0.1)
    ops, _, info = run("cantilever_beam", "중력을 -y 방향으로")
    assert ops[0]["type"] == "gravity" and ops[0]["vector"] == [0, -9810, 0]


def test_multi_clause_sentences():
    ops, reply, info = run("cantilever_beam", "왼쪽 끝을 고정하고 오른쪽 끝에 아래로 2 kN")
    assert info["complete"] and [o["op"] for o in ops] == ["add_bc", "add_load"] and ops[1]["vector"] == [0, 0, -2000]
    ops, reply, info = run("cantilever_beam", "왼쪽 고정 오른쪽 아래로 1kN")
    assert info["complete"] and ops[0]["select"] == XMIN and ops[1]["select"] == XMAX
    ops, reply, info = run("cantilever_beam", "x 최소 고정, x 최대에 아래로 1 kN, 좌굴 해석 모드 3개")
    assert [o["op"] for o in ops] == ["add_bc", "add_load", "set_step"] and ops[2] == {"op": "set_step", "type": "buckle", "modes": 3}
    ops, reply, info = run("cantilever_beam", "fix the left end and apply 2 kN downward at the right end")
    assert info["complete"] and ops[1]["vector"] == [0, 0, -2000] and ops[1]["select"] == XMAX
    ops, reply, info = run("cantilever_beam", "한쪽 끝 고정하고 반대쪽 끝에 1000 N 압축")
    assert info["complete"] and ops[0]["select"] == XMIN and ops[1]["select"] == XMAX and ops[1]["vector"] == [-1000, 0, 0]


def test_settings_sentences():
    assert run("cantilever_beam", "재료를 알루미늄으로 바꿔줘")[0][0]["E"] == 70000
    assert run("cantilever_beam", "steel 말고 알루미늄")[0][0]["E"] == 70000
    assert run("cantilever_beam", "밀도 7850 kg/m3")[0][0]["rho"] == pytest.approx(7.85e-9)
    assert run("cantilever_beam", "항복 없애줘")[0][0]["yield"] is None
    assert run("cantilever_beam", "보 단면 20x40")[0][0]["beam"] == {"shape": "rect", "b": 20.0, "h": 40.0}
    assert run("cantilever_beam", "원형 단면 지름 30")[0][0]["beam"] == {"shape": "circ", "r": 15.0}
    assert run("shell_panel", "panel 두께를 1.5로")[0][0] == {"op": "set_section", "group": "panel", "thickness": 1.5}
    assert run("cantilever_beam", "비선형 켜고 증분 50")[0][0] == {"op": "set_step", "nlgeom": True, "increments": 50}
    assert run("cantilever_beam", "고유진동수 5개 구해줘")[0][0] == {"op": "set_step", "type": "modal", "modes": 5}
    assert run("cantilever_beam", "요소 분할 8")[0][0] == {"op": "set_divisions", "divisions": 8}
    assert run("miura", "바-힌지 솔버로 풀어줘")[2]["run"]


def test_hinge_sentences():
    ops, _, info = run("miura", "쉘 간 일치하는 엣지는 핀조인트 힌지로 처리")
    assert info["complete"] and ops == [{"op": "auto_hinge", "mode": "auto"}, {"op": "set_hinge", "group": "*", "pin": True}]
    ops, _, info = run("miura", "꺾인 엣지를 힌지로, 각도 10도 이상")
    assert ops == [{"op": "auto_hinge", "mode": "crease", "angle": 10.0}]
    assert run("miura", "힌지 해제")[0] == [{"op": "auto_hinge", "mode": "off"}]
    assert run("miura", "힌지 강성 0.05")[0] == [{"op": "set_hinge", "group": "*", "kappa": 0.05, "pin": False}]
    assert run("miura", "hinge_fold 강성 0.1")[0][0]["group"] == "hinge_fold"
    assert run("miura", "treat shared edges between shells as pin joints")[2]["complete"]


def test_edit_sentences():
    ops, _, info = run("cantilever_beam", "ld1 2배", preset=True)
    assert ops == [{"op": "update", "id": "ld1", "vector": [0.0, 0.0, -2000.0]}]
    ops, _, info = run("cantilever_beam", "하중을 2500 N으로 바꿔", preset=True)
    assert ops[0]["vector"] == pytest.approx([0, 0, -2500])
    assert run("cantilever_beam", "하중 방향 반대로", preset=True)[0][0]["vector"] == [0.0, 0.0, 1000.0]
    assert run("cantilever_beam", "구속 모두 삭제", preset=True)[0] == [{"op": "clear", "what": "bcs"}]
    assert run("cantilever_beam", "마지막 하중 삭제", preset=True)[0] == [{"op": "remove", "id": "ld1"}]
    assert run("cantilever_beam", "되돌려")[2]["undo"]
    assert run("cantilever_beam", "해석 실행해줘")[2]["run"]


def test_plots():
    assert run("miura", "시간-총에너지 그래프 그려줘")[0][0]["series"] == [{"quantity": "energy.total"}]
    p = run("cantilever_beam", "끝단 z 변위 그래프")[0][0]
    assert p["series"][0] == {"quantity": "uz", "select": XMAX, "reduce": "mean"}
    assert run("cantilever_beam", "하중-변위 곡선 그려줘")[0][0]["title"] == "하중-변위"


@pytest.mark.parametrize(
    "text",
    ["이 구조물 어디가 제일 위험할 것 같아?", "끝단에 적당히 힘 좀 줘봐", "오른쪽 끝에 1 kN", "위쪽 끝에 5000", "아무거나 해줘", "z=100 위치 절점들을 반경 방향으로 구속", "끝단을 살짝 비스듬하게 눌러줘 3 mm"],
)
def test_unclear_sentences_are_not_marked_complete(text):
    ops, reply, info = run("cantilever_beam", text)
    assert not info["complete"], (ops, reply)


def test_numeric_plane_snaps_to_nearest_nodes():
    ops, reply, info = run("cantilever_beam", "x=260 위치에 아래로 100 N")
    assert ops[0]["select"]["plane"]["at"] in (200.0, 300.0) and "맞췄습니다" in reply
