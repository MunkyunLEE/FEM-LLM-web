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
    a, L, npa, nz = 40.0, 120.0, 6, 18
    ring = []
    for side in range(4):
        for i in range(npa):
            q = a * i / npa
            ring.append([(q, 0.0), (a, q), (a - q, a), (0.0, a - q)][side])
    m = len(ring)
    out = ["# square tube 40x40x120"]
    for k in range(nz + 1):
        for x, y in ring:
            out.append(f"v {x:g} {y:g} {L * k / nz:g}")
    for k in range(nz):
        for i in range(m):
            p0, q0 = k * m + i, k * m + (i + 1) % m
            out.append(f"f {p0 + 1} {q0 + 1} {q0 + m + 1} {p0 + m + 1}")
    write("square_tube.obj", "\n".join(out) + "\n")


def square_tube_fine(a=40.0, L=120.0, npa=20):
    nz = int(round(npa * L / a))
    ring = []
    for side in range(4):
        for i in range(npa):
            q = a * i / npa
            ring.append([(q, 0.0), (a, q), (a - q, a), (0.0, a - q)][side])
    m = len(ring)
    out = [f"# square tube 40x40x120, {npa} quads per wall"]
    for k in range(nz + 1):
        for x, y in ring:
            out.append(f"v {x:g} {y:g} {L * k / nz:g}")
    for k in range(nz):
        for i in range(m):
            p0, q0 = k * m + i, k * m + (i + 1) % m
            out.append(f"f {p0 + 1} {q0 + 1} {q0 + m + 1} {p0 + m + 1}")
    write("square_tube_fine.obj", "\n".join(out) + "\n")


def four_point_tube(a=40.0, L=400.0, npa=8, t=1.0, R=10.0, w=60.0, sup=170.0, ld=60.0):
    nx = int(round(npa * L / a))
    h = 0.5 * a
    ring = []
    for side in range(4):
        for i in range(npa):
            q = -h + a * i / npa
            ring.append([(q, -h), (h, q), (-q, h), (-h, -q)][side])
    m = len(ring)
    V = []
    out = [f"# square tube {a:g}x{a:g}x{L:g} on two rollers, two loading rollers (four-point bending)", "g tube"]
    for k in range(nx + 1):
        for y, z in ring:
            V.append((-0.5 * L + L * k / nx, y, z))
    for k in range(nx):
        for i in range(m):
            p0, q0 = k * m + i, k * m + (i + 1) % m
            out.append(f"f {p0 + 1} {q0 + 1} {q0 + m + 1} {p0 + m + 1}")
    nc, ny = 32, 6
    for name, xc, zc in (("load_l", -ld, h + 0.5 * t + R), ("load_r", ld, h + 0.5 * t + R), ("support_l", -sup, -h - 0.5 * t - R), ("support_r", sup, -h - 0.5 * t - R)):
        out.append(f"g {name}")
        b = len(V)
        for j in range(ny + 1):
            y = -0.5 * w + w * j / ny
            for i in range(nc):
                th = 2 * math.pi * i / nc
                V.append((xc + R * math.cos(th), y, zc + R * math.sin(th)))
        c0, c1 = len(V), len(V) + 1
        V += [(xc, -0.5 * w, zc), (xc, 0.5 * w, zc)]
        for j in range(ny):
            for i in range(nc):
                p0, q0 = b + j * nc + i, b + j * nc + (i + 1) % nc
                out.append(f"f {p0 + 1} {q0 + 1} {q0 + nc + 1} {p0 + nc + 1}")
        for i in range(nc):
            out.append(f"f {c0 + 1} {b + (i + 1) % nc + 1} {b + i + 1}")
            out.append(f"f {c1 + 1} {b + ny * nc + i + 1} {b + ny * nc + (i + 1) % nc + 1}")
    write("four_point_tube.obj", "\n".join([out[0]] + [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in V] + out[1:]) + "\n")


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


def panel_chain(L=100.0, W=60.0, n=4, m=3):
    out = ["# three separate shell panels with coincident edges (welded on import)"]
    faces = []
    nv = 0
    for k, name in enumerate(("panel_A", "panel_B", "panel_C")):
        x0 = k * L
        for j in range(m + 1):
            for i in range(n + 1):
                out.append(f"v {x0 + L * i / n:.6f} {W * j / m:.6f} 0")
        faces.append(f"o {name}")
        for j in range(m):
            for i in range(n):
                a = nv + j * (n + 1) + i + 1
                faces.append(f"f {a} {a + 1} {a + n + 2} {a + n + 1}")
        nv += (n + 1) * (m + 1)
    write("panel_chain.obj", "\n".join(out + faces) + "\n")


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


def two_blocks():
    def box(lo, hi):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        out = []
        for q in quads:
            out.append((c[q[0]], c[q[1]], c[q[2]]))
            out.append((c[q[0]], c[q[2]], c[q[3]]))
        return out

    stl("two_blocks", box((0, 0, 0), (40, 40, 10)) + box((10, 10, 10), (30, 30, 20)))


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


def _bh_obj(name, title, X, F, creases=(), group="panel"):
    out = [f"# {title}", f"g {group}"]
    out += ["v " + fmt(p) for p in X]
    out += ["f " + " ".join(str(int(i) + 1) for i in f) for f in F]
    if creases:
        out.append("g hinge_fold")
        out += [f"l {int(a) + 1} {int(b) + 1}" for a, b in creases]
    write(name, "\n".join(out) + "\n")


def bh_examples():
    import sys

    sys.path.insert(0, os.path.dirname(HERE))
    from bhweb.shapes import kresling, strip

    X, F = strip(10.0, 1.0, 60, tri=True, alt=True)
    _bh_obj("bh_strip.obj", "Woodruff 2020 strip L=10 W=1 a=6", X, F)
    edges = sorted({(min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3])) for f in F for i in range(3)})
    inner = [e for e in edges if sum(1 for f in F if e[0] in f and e[1] in f) == 2]
    PRESETS["bh_strip"]["ops"].insert(1, {"op": "set_crease", "edges": [list(e) for e in inner], "type": "bend"})
    _bh_obj("bh_simple_fold.obj", "Tang 2026 simple fold", [(0, 0, 0), (10, 0, 0), (5, 5, 0), (5, -5, 0)], [[0, 1, 2], [1, 0, 3]], [(0, 1)])
    X, F, mo, va, b, t = kresling()
    _bh_obj("bh_kresling.obj", "Tang 2026 Kresling a=1 h0=1 phi=11pi/18", X, F, mo + va)
    al = (55, 55, 70, 70)
    nx, ny, a, bb = 4, 3, 0.8, 1.0
    V = []
    for j in range(ny + 1):
        d = a / math.tan(math.radians(al[j]))
        for i in range(nx + 1):
            V.append((i * a, j * bb + (d if i % 2 else 0.0), 0.0))
    idx = lambda i, j: j * (nx + 1) + i
    Fq = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(ny) for i in range(nx)]
    cr = [(idx(i, j), idx(i + 1, j)) for j in range(1, ny) for i in range(nx)] + [(idx(i, j), idx(i, j + 1)) for i in range(1, nx) for j in range(ny)]
    _bh_obj("bh_flat_miura.obj", "Filipov 2017 flat Miura 4x3, alpha 55/70", V, Fq, cr)
    X = [(0, 0, 0), (30, 0, 0), (60, 0, 0), (90, 0, 0), (0, 30, 0), (30, 30, 0), (60, 30, 0), (90, 30, 0)]
    Fq = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]]
    _bh_obj("bh_roll.obj", "three panels rolled up with contact", X, Fq, [(1, 5), (2, 6)])


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
    "panel_chain": {
        "title": "6. 패널 3장 이음 (공유 엣지 핀 조인트)",
        "file": "panel_chain.obj",
        "note": "100x60 패널 3장(t=2)이 엣지를 맞대고 있음. 양 끝 고정, 가운데 패널 중앙선에 -100 N. 핀 조인트면 이음선 처짐 이론값 PL^3/(6EI) = 1.98 mm (보 이론)",
        "prompts": ["쉘 사이 공유 엣지를 핀 조인트로 처리해줘", "양단 고정", "x=150 위치에 아래로 100 N", "해석 실행"],
        "ops": [
            {"op": "set_solver", "solver": "calculix"},
            {"op": "set_section", "group": "*", "thickness": 2.0},
            {"op": "set_divisions", "divisions": 2},
            {"op": "auto_hinge", "mode": "auto"},
            {"op": "set_hinge", "group": "*", "pin": True},
            {"op": "add_bc", "type": "fixed", "name": "양단 고정", "select": {"or": [{"plane": {"axis": "x", "at": "min"}}, {"plane": {"axis": "x", "at": "max"}}]}},
            {"op": "add_load", "type": "force", "name": "중앙선 하중", "vector": [0, 0, -100], "select": {"plane": {"axis": "x", "at": 150.0}}},
        ],
    },
    "contact_sphere": {
        "title": "7a. 패널에 강체 구 압입 (접촉, 마찰)",
        "file": "shell_panel.obj",
        "note": "200x200 패널(t=2) 네 변 고정, 반지름 40 강체 구를 중앙에서 3 mm 압입, 마찰 0.2. 선형 판 이론(중앙 집중하중, 고정 지지)으로는 3 mm 처짐에 약 2060 N 이고, 기하비선형을 켜면 막 효과로 더 커집니다",
        "prompts": ["테두리 고정", "반지름 40 구를 아래로 3 mm 눌러, 마찰 0.2", "해석 실행"],
        "ops": [
            {"op": "set_solver", "solver": "calculix"},
            {"op": "set_section", "group": "*", "thickness": 2.0},
            {"op": "set_divisions", "divisions": 2},
            {"op": "add_bc", "type": "fixed", "name": "네 변 고정", "select": {"boundary": True}},
            {"op": "set_step", "type": "static", "nlgeom": True, "increments": 10},
            {"op": "add_contact", "type": "rigid", "shape": "sphere", "radius": 40.0, "center": [100.0, 100.0, 41.0], "motion": [0, 0, -3.0], "friction": 0.2},
        ],
    },
    "contact_blocks": {
        "title": "7b. 블록 두 개 접촉 (변형체-변형체, 마찰)",
        "file": "two_blocks.stl",
        "size": 5.0,
        "note": "아래 블록 40x40x10(바닥 고정) 위에 20x20x10 블록. 윗면을 0.05 mm 누르면서 x로 0.1 mm 밀어 미끄러짐을 만듭니다. 접선력/수직력이 마찰계수 0.3 에 가까워야 합니다",
        "prompts": ["z 최소 면 고정", "body_2 와 body_1 사이 접촉, 마찰 0.3", "해석 실행"],
        "ops": [
            {"op": "set_solver", "solver": "tensormesh"},
            {"op": "add_bc", "type": "pinned", "name": "바닥 고정", "select": {"plane": {"axis": "z", "at": "min"}}},
            {"op": "add_bc", "name": "윗면 누름", "dofs": [3], "value": -0.05, "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "add_bc", "name": "윗면 x 이동", "dofs": [1], "value": 0.1, "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "add_bc", "name": "윗면 y 구속", "dofs": [2], "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "set_step", "type": "static", "increments": 4},
            {"op": "add_contact", "type": "pair", "a": {"group": "body_2"}, "b": {"group": "body_1"}, "friction": 0.3},
        ],
    },
    "tube_crush": {
        "title": "7c. 사각관 축방향 압괴 (자체 쉘, 탄소성, 자기 접촉, 감쇠)",
        "file": "square_tube.obj",
        "note": "40x40x120 알루미늄 사각관(t=1)의 위 끝을 강제 변위로 20 mm 누릅니다(하중-변위 곡선 자동). 자체 쉘 솔버 + J2 탄소성(다선형 곡선: 240, 275, 300, 330 MPa @ 0, 2, 6, 20%) + 자기 접촉(IPC, 마찰 0.2) + 레일리 감쇠 2%. 1차 좌굴 모드 0.2 mm 초기 결함. 벽이 접히며 하중이 톱니처럼 오르내립니다",
        "prompts": ["자체 쉘 솔버로", "E 70000, 포아송비 0.3, 밀도 2.7e-9", "소성 곡선 240@0, 275@0.02, 300@0.06, 330@0.2", "아래 고정, 위는 x y 구속", "위에 z -20 mm", "초기결함 0.2", "자기 접촉 추가 마찰 0.2", "레일리 감쇠 2%", "해석 실행"],
        "ops": [
            {"op": "set_solver", "solver": "shell"},
            {"op": "set_material", "name": "default", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9, "curve": [[0.0, 240.0], [0.02, 275.0], [0.06, 300.0], [0.2, 330.0]]},
            {"op": "set_section", "group": "*", "thickness": 1.0},
            {"op": "add_bc", "name": "아래 고정", "dofs": [1, 2, 3], "select": {"plane": {"axis": "z", "at": "min"}}},
            {"op": "add_bc", "name": "위 x y 구속", "dofs": [1, 2], "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "add_bc", "name": "위 강제 변위", "dofs": [3], "value": -20.0, "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "set_step", "type": "static", "increments": 40, "imperfection": 0.2, "damping": {"kind": "rayleigh", "zeta": 2.0}},
            {"op": "add_contact", "type": "self", "friction": 0.2},
        ],
    },
    "tube_platen": {
        "title": "7d. 사각관 강체 누름판 압축 (길이 절반, 에너지 분해)",
        "file": "square_tube_fine.obj",
        "note": "40x40x120 알루미늄 사각관(t=1, 벽당 20분할 2 mm 요소)을 위 끝 구속 없이 강체 평면(누름판)으로 60 mm(길이의 절반) 누릅니다. 아래 끝 완전 고정, 누름판·자기 접촉 마찰 0.2, 레일리 감쇠 2%(스텝 시간 0.65 s = 1차 주기의 1000배, 감쇠 소산이 외부 일의 5% 미만이 되도록), 1차 좌굴 모드 0.2 mm 초기 결함. 누름판 반력-변위와 탄성·소성·감쇠·마찰 에너지, 총 에너지를 그립니다. 2코어 기준 약 3시간",
        "prompts": ["자체 쉘 솔버로", "E 70000, 포아송비 0.3, 밀도 2.7e-9", "소성 곡선 240@0, 275@0.02, 300@0.06, 330@0.2", "아래 완전 고정", "위에서 강체 평면을 아래로 60 mm 눌러, 마찰 0.2", "자기 접촉 추가 마찰 0.2", "레일리 감쇠 2%, 스텝 시간 0.65초", "증분 150, 초기결함 0.2", "해석 실행"],
        "ops": [
            {"op": "set_solver", "solver": "shell"},
            {"op": "set_material", "name": "default", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9, "curve": [[0.0, 240.0], [0.02, 275.0], [0.06, 300.0], [0.2, 330.0]]},
            {"op": "set_section", "group": "*", "thickness": 1.0},
            {"op": "add_bc", "name": "아래 완전 고정", "dofs": [1, 2, 3, 4, 5, 6], "select": {"plane": {"axis": "z", "at": "min"}}},
            {"op": "add_contact", "type": "rigid", "shape": "plane", "normal": [0, 0, -1.0], "motion": [0, 0, -60.0], "friction": 0.2},
            {"op": "add_contact", "type": "self", "friction": 0.2},
            {"op": "set_step", "type": "static", "increments": 150, "imperfection": 0.2, "damping": {"kind": "rayleigh", "zeta": 2.0, "time": 0.65}},
            {"op": "add_plot", "title": "에너지-누름판 변위", "x": {"quantity": "contact.disp"}, "series": [{"quantity": "energy.strain"}, {"quantity": "energy.plastic"}, {"quantity": "energy.damping"}, {"quantity": "energy.friction"}, {"quantity": "energy.total"}, {"quantity": "energy.external"}]},
        ],
    },
    "tube_four_point": {
        "title": "7e. 사각관 4점 굽힘 (강체 롤러 4개)",
        "file": "four_point_tube.obj",
        "note": "40x40x400 알루미늄 사각관(t=1, 벽당 8분할 5 mm 요소)을 반지름 10 mm 강체 롤러 4개로 4점 굽힘합니다. 받침 롤러(support_l, support_r) 간격 340 mm 는 고정, 하중 롤러(load_l, load_r) 간격 120 mm 를 아래로 30 mm 누릅니다. 롤러는 OBJ 의 별도 그룹(닫힌 메쉬)이고 '강체 부품 접촉'으로 지정해 해석 모델에서 빠지고 움직이는 강체 면이 됩니다. 관은 받침 위에 놓였을 뿐이라 좌우 대칭으로 중앙 단면 ux=0, y=0 선 uy=0 만 주고, 나머지 강체 운동은 감쇠가 잡습니다. 마찰 0.1, 레일리 감쇠 2%(스텝 시간 1.6 s = 1차 주기의 약 1000배). 롤러 반력과 에너지를 그립니다. 2코어 기준 약 2시간 (최대 하중 7.85 kN, 에너지 균형 0.03%)",
        "prompts": ["자체 쉘 솔버로", "E 70000, 포아송비 0.3, 밀도 2.7e-9", "소성 곡선 240@0, 275@0.02, 300@0.06, 330@0.2", "두께 1", "load_l 을 강체로 아래로 30 mm 눌러, 마찰 0.1", "load_r 을 강체로 아래로 30 mm 눌러, 마찰 0.1", "support_l 강체 받침으로, 마찰 0.1", "support_r 강체 받침으로, 마찰 0.1", "레일리 감쇠 2%, 스텝 시간 1.6초", "증분 120", "해석 실행"],
        "ops": [
            {"op": "set_solver", "solver": "shell"},
            {"op": "set_material", "name": "default", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9, "curve": [[0.0, 240.0], [0.02, 275.0], [0.06, 300.0], [0.2, 330.0]]},
            {"op": "set_section", "group": "*", "thickness": 1.0},
            {"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": "load_l"}, "motion": [0, 0, -30.0], "friction": 0.1},
            {"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": "load_r"}, "motion": [0, 0, -30.0], "friction": 0.1},
            {"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": "support_l"}, "motion": [0, 0, 0], "friction": 0.1},
            {"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": "support_r"}, "motion": [0, 0, 0], "friction": 0.1},
            {"op": "add_bc", "name": "중앙 단면 x 대칭", "dofs": [1], "select": {"and": [{"group": "tube"}, {"plane": {"axis": "x", "at": 0}}]}},
            {"op": "add_bc", "name": "y=0 선 y 대칭", "dofs": [2], "select": {"and": [{"group": "tube"}, {"plane": {"axis": "y", "at": 0}}]}},
            {"op": "set_step", "type": "static", "increments": 120, "damping": {"kind": "rayleigh", "zeta": 2.0, "time": 1.6}},
            {"op": "add_plot", "title": "롤러 반력-하중 롤러 변위", "x": {"quantity": "contact.disp"}, "series": [{"quantity": "contact.ct1.force"}, {"quantity": "contact.ct2.force"}, {"quantity": "contact.ct3.force"}, {"quantity": "contact.ct4.force"}]},
            {"op": "add_plot", "title": "에너지-하중 롤러 변위", "x": {"quantity": "contact.disp"}, "series": [{"quantity": "energy.strain"}, {"quantity": "energy.plastic"}, {"quantity": "energy.damping"}, {"quantity": "energy.friction"}, {"quantity": "energy.total"}, {"quantity": "energy.external"}]},
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
    "bh_strip": {
        "title": "B1. [BH-web] 얇은 띠 4점 굽힘 (Woodruff 2020, 그림 5c)",
        "file": "bh_strip.obj",
        "note": "L=10, W=1, t=0.1, E=1000, ν=0, 교대 대각선 삼각형 a=6. 보 이론 에너지 6EIδ²/(s²(3L-4s)) 대비 논문 약 1.03, BH-web 1.030",
        "prompts": [],
        "ops": [
            {"op": "set_solver", "solver": "bh"},
            {"op": "set_section", "group": "*", "thickness": 0.1},
            {"op": "set_material", "name": "default", "E": 1000, "nu": 0.0, "rho": 1e-9},
            {"op": "add_bc", "name": "양 끝 z 지지", "dofs": [3], "select": {"or": [{"plane": {"axis": "x", "at": "min"}}, {"plane": {"axis": "x", "at": "max"}}]}},
            {"op": "add_bc", "name": "강체 이동 방지", "dofs": [1, 2], "select": {"nearest": [0, 0, 0]}},
            {"op": "add_bc", "name": "강체 회전 방지", "dofs": [1], "select": {"nearest": [0, 1, 0]}},
            {"op": "add_bc", "name": "강체 회전 방지 y", "dofs": [2], "select": {"nearest": [10, 0, 0]}},
            {"op": "add_bc", "name": "하중점 강제 변위", "dofs": [3], "value": -0.0005, "select": {"or": [{"plane": {"axis": "x", "at": 0.1666667, "tol": 1e-4}}, {"plane": {"axis": "x", "at": 9.8333333, "tol": 1e-4}}]}},
            {"op": "set_step", "type": "static", "nlgeom": True, "increments": 1},
        ],
    },
    "bh_simple_fold": {
        "title": "B2. [BH-web] 단일 주름 탄소성 접힘 (Tang 2026, 그림 4)",
        "file": "bh_simple_fold.obj",
        "note": "직각 이등변 삼각형 2장, 주름 AB=10. k=1 N·mm/rad/mm, 항복 0.2 N·mm/mm, H=0.5. 이론: 0.2 rad 에서 항복, 뒤 기울기 kH/(k+H)",
        "prompts": [],
        "ops": [
            {"op": "set_solver", "solver": "bh"},
            {"op": "set_section", "group": "*", "thickness": 0.5},
            {"op": "set_material", "name": "default", "E": 67000, "nu": 0.33, "rho": 2.7e-9},
            {"op": "set_bh", "fold_mode": "value", "fold_k": 1.0, "fold_yield": 0.2, "fold_H": 0.5, "watch": ["0-1"]},
            {"op": "add_bc", "name": "고정 패널 ABC", "dofs": [1, 2, 3], "select": {"plane": {"axis": "y", "at": "max", "tol": 5.01}}},
            {"op": "add_bc", "name": "D x 구속", "dofs": [1], "select": {"nearest": [5, -5, 0]}},
            {"op": "add_bc", "name": "D 들어올림", "dofs": [3], "value": 4.0, "select": {"nearest": [5, -5, 0]}},
            {"op": "set_step", "type": "static", "nlgeom": True, "increments": 40},
            {"op": "add_plot", "title": "주름 AB 모멘트-이면각", "x": {"quantity": "fold.e0-1"}, "series": [{"quantity": "moment.e0-1"}]},
        ],
    },
    "bh_kresling": {
        "title": "B3. [BH-web] 크레슬링 쌍안정 반복 하중 히스테리시스 (Tang 2026, 그림 12b 첫 칸)",
        "file": "bh_kresling.obj",
        "note": "육각 a=1, h0=1, 회전 11π/18. E=1e4, 바 항복 200·H 1e3, 접힘 k=0.2·항복 0.02·H 0.001. 위면 z 지정 변위 -0.6, 반복 5회, 되돌림 0.1 mm (-0.1 ↔ 0.6 왕복). 논문에 바 면적이 없어 A=0.045 로 맞춤(인장 끝 하중 -5). 가력·제하 모두 스냅(음의 강성) 구간",
        "prompts": [],
        "ops": [
            {"op": "set_solver", "solver": "bh"},
            {"op": "set_section", "group": "*", "thickness": 0.01},
            {"op": "set_material", "name": "default", "E": 10000, "nu": 0.3, "rho": 1e-9, "yield": 200, "hardening": 1000},
            {"op": "set_bh", "fold_mode": "value", "fold_k": 0.2, "bar_area": 0.045, "fold_yield": 0.02, "fold_H": 0.001, "cycles": 5, "cycle_back": 0.1, "watch": ["0-6", "1-6"]},
            {"op": "add_bc", "type": "pinned", "name": "아래 육각 고정", "select": {"plane": {"axis": "z", "at": "min"}}},
            {"op": "add_bc", "name": "위 육각 압축", "dofs": [3], "value": -0.6, "select": {"plane": {"axis": "z", "at": "max"}}},
            {"op": "set_step", "type": "static", "nlgeom": True, "increments": 60},
            {"op": "add_plot", "title": "주름 회전", "x": {"quantity": "time"}, "series": [{"quantity": "fold.e0-6"}, {"quantity": "fold.e1-6"}]},
        ],
    },
    "bh_flat_miura": {
        "title": "B4. [BH-web] 평면 미우라 강성 고유모드 (Filipov 2017, 그림 11)",
        "file": "bh_flat_miura.obj",
        "note": "4x3 패널, 위 α=70°, 아래 α=55°, E=1e6, ν=1/3, t=0.01, L*=1e4, Kv=λv 자유-자유. 논문: 7~11번 접힘 모드(λ 6.7e-6~1.3e-4) 뒤 12번 휨(0.139)",
        "prompts": [],
        "ops": [
            {"op": "set_solver", "solver": "bh"},
            {"op": "set_section", "group": "*", "thickness": 0.01},
            {"op": "set_material", "name": "default", "E": 1000000, "nu": 0.3333333, "rho": 1e-9},
            {"op": "set_bh", "Lstar": 10000, "use_mass": False},
            {"op": "set_step", "type": "modal", "modes": 8},
        ],
    },
    "bh_roll": {
        "title": "B5. [BH-web] 패널 3장 말아 접기 기하 접촉",
        "file": "bh_roll.obj",
        "note": "30x30 패널 3장, 두 주름의 목표 이면각 30°(삼각 관을 넘어 말리는 각). 접촉을 켜면 세 번째 패널이 첫 패널에 닿아 멈추고 관통 없음",
        "prompts": [],
        "ops": [
            {"op": "set_solver", "solver": "bh"},
            {"op": "set_section", "group": "*", "thickness": 0.5},
            {"op": "set_material", "name": "default", "E": 3000, "nu": 0.3, "rho": 1.2e-9},
            {"op": "set_bh", "contact": True, "rest": {"1-5": 30.0, "2-6": 30.0}},
            {"op": "add_bc", "type": "pinned", "name": "첫 패널 고정", "select": {"plane": {"axis": "x", "at": "min"}}},
            {"op": "add_bc", "type": "pinned", "name": "첫 주름 고정", "select": {"box": {"min": [29.9, -1, -1], "max": [30.1, 31, 1]}}},
            {"op": "set_step", "type": "static", "nlgeom": True, "increments": 40},
        ],
    },
    "bh_miura": {
        "title": "B6. [BH-web] 미우라 강체 접힘 기구 해석 · 3D 해공간",
        "file": "miura.obj",
        "note": "4x4 미우라, 패널 강체 가정. 기구 자유도 1, 접힘 경로에서 tan(ρ곧은/2)/tan(ρ지그재그/2) = cos γ",
        "prompts": [],
        "ops": [
            {"op": "set_solver", "solver": "bh"},
            {"op": "set_section", "group": "*", "thickness": 0.5},
            {"op": "set_material", "name": "default", "E": 3000, "nu": 0.3, "rho": 1.2e-9},
            {"op": "set_bh", "plot3": ["1-6", "5-6", "6-7"], "watch": ["1-6", "5-6"]},
            {"op": "set_step", "type": "kinematic", "increments": 60},
            {"op": "add_plot", "title": "주름 이면각", "x": {"quantity": "time"}, "series": [{"quantity": "fold.e1-6"}, {"quantity": "fold.e5-6"}, {"quantity": "fold.e6-7"}]},
        ],
    },
}

if __name__ == "__main__":
    cantilever_beam()
    shell_panel()
    square_tube_fine()
    four_point_tube()
    cylinder_shell()
    panel_chain()
    miura()
    cantilever_solid()
    cylinder_solid()
    two_blocks()
    bh_examples()
    write("presets.json", json.dumps(PRESETS, ensure_ascii=False, indent=1) + "\n")
    print("ok")
