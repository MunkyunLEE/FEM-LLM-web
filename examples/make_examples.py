import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, text):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)


def fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


def cantilever_beam(L=1000.0, n=10):
    out = ["# cantilever beam L=1000 rect 50x100", "g beam"]
    for i in range(n + 1):
        out.append("v " + fmt((L * i / n, 0, 0)))
    for i in range(n):
        out.append(f"l {i + 1} {i + 2}")
    write("cantilever_beam.obj", "\n".join(out) + "\n")


def shell_panel(a=200.0, n=10):
    out = ["# square shell panel 200x200", "g panel"]
    for j in range(n + 1):
        for i in range(n + 1):
            out.append("v " + fmt((a * i / n, a * j / n, 0)))
    for j in range(n):
        for i in range(n):
            p = j * (n + 1) + i + 1
            out.append(f"f {p} {p + 1} {p + n + 2} {p + n + 1}")
    write("shell_panel.obj", "\n".join(out) + "\n")


def cylinder_shell(R=50.0, L=200.0, nc=48, na=40):
    out = ["# cylindrical shell R=50 L=200", "g cylinder"]
    for j in range(na + 1):
        for i in range(nc):
            t = 2 * math.pi * i / nc
            out.append("v " + fmt((R * math.cos(t), R * math.sin(t), L * j / na)))
    for j in range(na):
        for i in range(nc):
            a = j * nc + i + 1
            b = j * nc + (i + 1) % nc + 1
            out.append(f"f {a} {b} {b + nc} {a + nc}")
    write("cylinder_shell.obj", "\n".join(out) + "\n")


def miura(a=50.0, b=50.0, gamma=60.0, theta=60.0, nx=4, ny=4):
    g = math.radians(gamma)
    t = math.radians(theta)
    H = a * math.sin(t) * math.sin(g)
    q = math.sqrt(1 + math.cos(t) ** 2 * math.tan(g) ** 2)
    S = b * math.cos(t) * math.tan(g) / q
    Lp = a * math.sqrt(1 - math.sin(t) ** 2 * math.sin(g) ** 2)
    V = b / q
    out = [f"# miura-ori a={a} b={b} gamma={gamma} theta={theta}", "g panel"]
    idx = lambda i, j: j * (nx + 1) + i + 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            out.append("v " + fmt((i * S, j * Lp + (V if i % 2 else 0), H if j % 2 else 0)))
    for j in range(ny):
        for i in range(nx):
            out.append(f"f {idx(i, j)} {idx(i + 1, j)} {idx(i + 1, j + 1)} {idx(i, j + 1)}")
    out.append("g hinge_fold")
    for j in range(1, ny):
        for i in range(nx):
            out.append(f"l {idx(i, j)} {idx(i + 1, j)}")
    for i in range(1, nx):
        for j in range(ny):
            out.append(f"l {idx(i, j)} {idx(i, j + 1)}")
    write("miura.obj", "\n".join(out) + "\n")


def stl(name, tris):
    out = [f"solid {name}"]
    for p, q, r in tris:
        ux, uy, uz = (q[k] - p[k] for k in range(3))
        vx, vy, vz = (r[k] - p[k] for k in range(3))
        n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        ln = math.sqrt(sum(c * c for c in n)) or 1.0
        out.append("facet normal " + " ".join(f"{c / ln:.6e}" for c in n))
        out.append(" outer loop")
        for v in (p, q, r):
            out.append("  vertex " + " ".join(f"{c:.6e}" for c in v))
        out.append(" endloop")
        out.append("endfacet")
    out.append(f"endsolid {name}")
    write(name + ".stl", "\n".join(out) + "\n")


def cantilever_solid(L=1000.0, b=50.0, h=100.0):
    y0, y1, z0, z1 = -b / 2, b / 2, -h / 2, h / 2
    c = [(0, y0, z0), (L, y0, z0), (L, y1, z0), (0, y1, z0), (0, y0, z1), (L, y0, z1), (L, y1, z1), (0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for q in quads:
        tris.append((c[q[0]], c[q[1]], c[q[2]]))
        tris.append((c[q[0]], c[q[2]], c[q[3]]))
    stl("cantilever_solid", tris)


def cylinder_solid(Ro=50.0, Ri=40.0, L=200.0, n=64):
    tris = []
    for i in range(n):
        t0 = 2 * math.pi * i / n
        t1 = 2 * math.pi * (i + 1) / n
        o0 = (Ro * math.cos(t0), Ro * math.sin(t0))
        o1 = (Ro * math.cos(t1), Ro * math.sin(t1))
        i0 = (Ri * math.cos(t0), Ri * math.sin(t0))
        i1 = (Ri * math.cos(t1), Ri * math.sin(t1))
        tris.append(((*o0, 0), (*o1, 0), (*o1, L)))
        tris.append(((*o0, 0), (*o1, L), (*o0, L)))
        tris.append(((*i0, 0), (*i1, L), (*i1, 0)))
        tris.append(((*i0, 0), (*i0, L), (*i1, L)))
        tris.append(((*o0, L), (*o1, L), (*i1, L)))
        tris.append(((*o0, L), (*i1, L), (*i0, L)))
        tris.append(((*o0, 0), (*i1, 0), (*o1, 0)))
        tris.append(((*o0, 0), (*i0, 0), (*i1, 0)))
    stl("cylinder_solid", tris)


PRESETS = {
    "cantilever_beam": {
        "title": "1. 직사각 단면 캔틸레버 빔 (보 요소)",
        "file": "cantilever_beam.obj",
        "note": "L=1000, 단면 b50 x h100, 끝단 -1000 N. 이론 처짐 PL^3/3EI = 0.381 mm",
        "prompts": ["x 최소인 끝을 완전 고정해줘", "x 최대 끝단에 z 방향으로 -1000 N 하중 줘", "끝단 처짐을 하중계수에 대해 그래프로 그려줘"],
        "ops": [
            {"op": "set_solver", "solver": "calculix"},
            {"op": "set_section", "group": "*", "beam": {"shape": "rect", "b": 50, "h": 100}},
            {"op": "add_bc", "type": "fixed", "name": "고정단", "select": {"plane": {"axis": "x", "at": "min"}}},
            {"op": "add_load", "type": "force", "name": "끝단 하중", "vector": [0, 0, -1000], "select": {"plane": {"axis": "x", "at": "max"}}},
        ],
    },
    "shell_panel": {
        "title": "2. 정사각형 쉘 패널 엣지 압축 (좌굴)",
        "file": "shell_panel.obj",
        "note": "200x200, t=2, 네 변 단순지지, x 방향 압축 1000 N. 이론 좌굴하중 4*pi^2*D/b = 30.37 kN (계수 30.4)",
        "prompts": ["네 변 모두 z 방향만 구속해줘", "x 최소 변을 x 방향 구속해줘", "절점 (0, 0, 0) 을 y 방향 구속해줘", "x 최대 변에 x 방향으로 -1000 N 압축 하중", "좌굴 해석으로 바꾸고 모드 4개"],
        "ops": [
            {"op": "set_solver", "solver": "calculix"},
            {"op": "set_section", "group": "*", "thickness": 2.0},
            {"op": "add_bc", "name": "네 변 z 지지", "dofs": [3], "select": {"or": [{"plane": {"axis": "x", "at": "min"}}, {"plane": {"axis": "x", "at": "max"}}, {"plane": {"axis": "y", "at": "min"}}, {"plane": {"axis": "y", "at": "max"}}]}},
            {"op": "add_bc", "name": "x 최소 변 x 구속", "dofs": [1], "select": {"plane": {"axis": "x", "at": "min"}}},
            {"op": "add_bc", "name": "강체 이동 방지 y", "dofs": [2], "select": {"and": [{"plane": {"axis": "x", "at": "min"}}, {"plane": {"axis": "y", "at": "min"}}]}},
            {"op": "add_load", "type": "force", "name": "엣지 압축", "vector": [-1000, 0, 0], "select": {"plane": {"axis": "x", "at": "max"}}},
            {"op": "set_step", "type": "buckle", "modes": 4},
        ],
    },
    "cylinder_shell": {
        "title": "3. 원형 중공 실린더 압축 좌굴 (쉘 요소)",
        "file": "cylinder_shell.obj",
        "note": "R=50, L=200, t=2, 축 압축 100 kN. 고전 이론 좌굴하중 약 3194 kN (계수 약 31.9)",
        "prompts": ["아래쪽 끝(z 최소)을 고정해줘", "위쪽 끝은 x, y 방향만 구속", "위쪽 끝에 z 방향 -100000 N 압축", "좌굴 해석 모드 4개"],
        "ops": [
            {"op": "set_solver", "solver": "calculix"},
            {"op": "set_section", "group": "*", "thickness": 2.0},
            {"op": "add_bc", "name": "하단 지지", "dofs": [1, 2, 3], "select": {"plane": {"axis": "z", "at": "min"}}},
            {"op": "add_bc", "name": "상단 반경 구속", "dofs": [1, 2], "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "add_load", "type": "force", "name": "축 압축", "vector": [0, 0, -100000], "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "set_step", "type": "buckle", "modes": 4},
        ],
    },
    "miura": {
        "title": "4. 미우라 오리 1자유도 접힘 (바-힌지)",
        "file": "miura.obj",
        "note": "a=b=50, gamma=60도, 초기 접힘각 60도, 4x4 패널, t=0.5. 힌지 강성 kappa = 0.1 x D / L",
        "prompts": ["x 최소 쪽 절점들을 x 방향 구속해줘", "x 최대 쪽 절점들을 x 방향으로 -30 mm 밀어줘", "시간-총에너지 그래프 그려줘"],
        "ops": [
            {"op": "set_solver", "solver": "barhinge"},
            {"op": "set_section", "group": "*", "thickness": 0.5},
            {"op": "set_material", "name": "default", "E": 3000, "nu": 0.3, "rho": 1.2e-9},
            {"op": "set_hinge", "group": "*", "kappa": None, "ratio": 0.1},
            {"op": "add_bc", "name": "x 최소 변 x 구속", "dofs": [1], "select": {"plane": {"axis": "x", "at": "min"}}},
            {"op": "add_bc", "name": "기준점 y z 구속", "dofs": [2, 3], "select": {"nearest": [0, 0, 0]}},
            {"op": "add_bc", "name": "회전 방지 z", "dofs": [3], "select": {"and": [{"plane": {"axis": "x", "at": "min"}}, {"plane": {"axis": "z", "at": "min"}}]}},
            {"op": "add_bc", "name": "x 최대 변 강제 변위", "dofs": [1], "value": -30.0, "select": {"plane": {"axis": "x", "at": "max"}}},
            {"op": "set_step", "type": "static", "nlgeom": True, "increments": 30},
            {"op": "add_plot", "title": "에너지 이력", "x": {"quantity": "time"}, "series": [{"quantity": "energy.total"}, {"quantity": "energy.fold"}, {"quantity": "energy.bar"}, {"quantity": "energy.bend"}]},
        ],
    },
    "cantilever_solid": {
        "title": "5a. 캔틸레버 빔 (솔리드 요소)",
        "file": "cantilever_solid.stl",
        "size": 28.0,
        "note": "1000 x 50 x 100 솔리드, 끝단 면에 -1000 N. 보 이론 처짐 0.381 mm",
        "prompts": ["x 최소 면을 고정해줘", "x 최대 면에 z 방향 -1000 N"],
        "ops": [
            {"op": "set_solver", "solver": "tensormesh"},
            {"op": "add_bc", "type": "pinned", "name": "고정단", "select": {"plane": {"axis": "x", "at": "min"}}},
            {"op": "add_load", "type": "force", "name": "끝단 하중", "vector": [0, 0, -1000], "select": {"plane": {"axis": "x", "at": "max"}}},
        ],
    },
    "cylinder_solid": {
        "title": "5b. 원형 중공 실린더 압축 (솔리드 요소)",
        "file": "cylinder_solid.stl",
        "size": 9.0,
        "note": "외경 R50, 내경 R40, L=200, 축 압축 100 kN. 이론 축변형 PL/EA = 0.0337 mm",
        "prompts": ["z 최소 면을 고정해줘", "z 최대 면에 z 방향 -100000 N 압축"],
        "ops": [
            {"op": "set_solver", "solver": "tensormesh"},
            {"op": "add_bc", "type": "pinned", "name": "하단 고정", "select": {"plane": {"axis": "z", "at": "min"}}},
            {"op": "add_load", "type": "force", "name": "축 압축", "vector": [0, 0, -100000], "select": {"plane": {"axis": "z", "at": "max"}}},
        ],
    },
}

if __name__ == "__main__":
    cantilever_beam()
    shell_panel()
    cylinder_shell()
    miura()
    cantilever_solid()
    cylinder_solid()
    write("presets.json", json.dumps(PRESETS, ensure_ascii=False, indent=1) + "\n")
    print("ok")
