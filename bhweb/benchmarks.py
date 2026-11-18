import numpy as np

from .model import BHModel, edges_of
from .shapes import strip
from .solve import Problem, solve_displacement


def _energy(rec):
    e = rec.E[-1]
    return e["stretch"] + e["fold"] + e["bend"]


def _all_bend(F):
    return {e: "bend" for e, fl in edges_of(F).items() if len(fl) == 2}


def strip_case(case, a, E=1000.0, nu=0.0, t=0.1, L=10.0, W=1.0, d=5e-4, alt=True, tri_bars="woodruff"):
    n = max(1, int(round(a * L / W)))
    X, F = strip(L, W, n, tri=True, alt=alt)
    m = BHModel(X, F, types=_all_bend(F), params={"E": E, "nu": nu, "t": t, "tri_bars": tri_bars})
    left = [0, n + 1]
    right = [n, 2 * n + 1]
    fixed = []
    disp = {}
    G = E / (2 * (1 + nu))
    D = E * t**3 / 12.0
    if case == "tension":
        for i in left:
            fixed += [3 * i, 3 * i + 1, 3 * i + 2]
        fixed += [3 * right[0] + 1, 3 * right[0] + 2, 3 * right[1] + 2]
        disp = {3 * right[0]: d, 3 * right[1]: d}
        Uth = 0.5 * E * W * t / L * d * d
    elif case == "torsion":
        for i in left:
            fixed += [3 * i, 3 * i + 1, 3 * i + 2]
        disp = {3 * right[0] + 2: d, 3 * right[1] + 2: -d}
        phi = 2 * d / W
        Uth = 0.5 * G * W * t**3 / 3.0 / L * phi * phi
    elif case == "bending":
        for i in left:
            fixed += [3 * i + 2]
        for i in right:
            fixed += [3 * i + 2]
        fixed += [3 * left[0], 3 * left[0] + 1, 3 * left[1], 3 * right[0] + 1]
        pa = [1, n + 2]
        pb = [n - 1, 2 * n]
        for i in pa + pb:
            disp[3 * i + 2] = -d
        s = L / n
        EI = D * W
        Uth = 6 * EI * d * d / (s * s * (3 * L - 4 * s))
    elif case == "inplane":
        for i in left:
            fixed += [3 * i, 3 * i + 1, 3 * i + 2]
        fixed += [3 * i + 2 for i in range(len(X))]
        disp = {3 * right[0] + 1: d, 3 * right[1] + 1: d}
        EI = E * t * W**3 / 12.0
        Uth = 0.5 * 3 * EI / L**3 * d * d
    else:
        raise ValueError(case)
    fixed = sorted(set(fixed) - set(disp))
    rec = solve_displacement(m, Problem(m, fixed=fixed, disp=disp), steps=1, tol=1e-12)
    return _energy(rec) / Uth


def woodruff_strips(aspects=(1, 2, 3, 4, 6, 8, 12)):
    out = {}
    for case in ("tension", "torsion", "bending", "inplane"):
        out[case] = [(a, strip_case(case, a)) for a in aspects]
    return out


if __name__ == "__main__":
    for k, v in woodruff_strips().items():
        print(k, " ".join(f"a={a}:{r:.3f}" for a, r in v))
