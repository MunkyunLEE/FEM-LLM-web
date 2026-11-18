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


def strip(L=100.0, W=10.0, n=10, tri=False):
    X = [(L * i / n, y, 0.0) for y in (0.0, W) for i in range(n + 1)]
    F = []
    for i in range(n):
        a, b, c, d = i, i + 1, n + 2 + i, n + 1 + i
        F += [[a, b, c], [a, c, d]] if tri else [[a, b, c, d]]
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
