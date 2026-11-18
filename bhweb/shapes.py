import math

import numpy as np


def miura(a=50.0, b=50.0, gamma=60.0, theta=60.0, nx=4, ny=4):
    g = math.radians(gamma)
    t = math.radians(theta)
    H = a * math.sin(t) * math.sin(g)
    q = math.sqrt(1 + math.cos(t) ** 2 * math.tan(g) ** 2)
    S = b * math.cos(t) * math.tan(g) / q
    Lp = a * math.sqrt(1 - math.sin(t) ** 2 * math.sin(g) ** 2)
    V = b / q
    idx = lambda i, j: j * (nx + 1) + i
    X = [(i * S, j * Lp + (V if i % 2 else 0), H if j % 2 else 0) for j in range(ny + 1) for i in range(nx + 1)]
    F = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(ny) for i in range(nx)]
    cx = [(idx(i, j), idx(i + 1, j)) for j in range(1, ny) for i in range(nx)]
    cy = [(idx(i, j), idx(i, j + 1)) for i in range(1, nx) for j in range(ny)]
    return np.array(X), F, cx, cy


def strip(L=100.0, W=10.0, n=10, tri=False, alt=False):
    X = [(L * i / n, y, 0.0) for y in (0.0, W) for i in range(n + 1)]
    F = []
    for i in range(n):
        a, b, c, d = i, i + 1, n + 2 + i, n + 1 + i
        if not tri:
            F.append([a, b, c, d])
        elif alt and i % 2:
            F += [[a, b, d], [b, c, d]]
        else:
            F += [[a, b, c], [a, c, d]]
    return np.array(X), F


def grid(Lx, Ly, nx, ny, tri=False):
    X = [(Lx * i / nx, Ly * j / ny, 0.0) for j in range(ny + 1) for i in range(nx + 1)]
    idx = lambda i, j: j * (nx + 1) + i
    F = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, b, c], [a, c, d]] if tri else [[a, b, c, d]]
    return np.array(X), F


def kresling(n=6, a=1.0, h=1.0, phi=11 * math.pi / 18):
    R = a / (2 * math.sin(math.pi / n))
    X = [(R * math.cos(2 * math.pi * i / n), R * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    X += [(R * math.cos(2 * math.pi * i / n + phi), R * math.sin(2 * math.pi * i / n + phi), h) for i in range(n)]
    F = []
    mountain = []
    valley = []
    for i in range(n):
        b0, b1 = i, (i + 1) % n
        t0, t1 = n + i, n + (i + 1) % n
        F += [[b0, b1, t0], [b1, t1, t0]]
        valley.append((min(b1, t0), max(b1, t0)))
        mountain.append((min(b0, t0), max(b0, t0)))
    top = list(range(n, 2 * n))
    bottom = list(range(n))
    return np.array(X), F, mountain, valley, bottom, top
