import re

import numpy as np

_SPLIT = re.compile(r"[,\t;]|\s+")


def _nums(line):
    toks = [t for t in _SPLIT.split(line.strip()) if t != ""]
    if not toks:
        return None
    try:
        return [float(t.replace("D", "E")) for t in toks]
    except ValueError:
        return None


def _names(line, n):
    toks = [t.strip().strip('"') for t in re.split(r"[,\t;]", line.strip())]
    toks = [t for t in toks if t]
    if len(toks) != n:
        toks = [t for t in re.split(r"\s{2,}", line.strip()) if t]
    if len(toks) != n:
        toks = [t for t in line.split() if t]
    return toks if len(toks) == n else None


def parse(text, name="Abaqus"):
    blocks = []
    cur = None
    header = None
    for raw in text.splitlines():
        v = _nums(raw)
        if v is None:
            if raw.strip():
                header = raw
            if cur:
                blocks.append(cur)
                cur = None
            continue
        if cur is None or len(v) != cur["n"]:
            if cur:
                blocks.append(cur)
            cur = {"n": len(v), "rows": [], "header": header}
        cur["rows"].append(v)
    if cur:
        blocks.append(cur)
    curves = []
    for b in blocks:
        A = np.array(b["rows"], float)
        if len(A) < 2 or A.shape[1] < 2:
            continue
        hn = _names(b["header"], A.shape[1]) if b["header"] else None
        for k in range(1, A.shape[1]):
            nm = hn[k] if hn else (name if A.shape[1] == 2 else f"{name} {k}")
            curves.append({"name": nm, "x": A[:, 0].tolist(), "y": A[:, k].tolist(), "xlabel": hn[0] if hn else "X"})
    if not curves:
        raise ValueError("숫자 열을 찾지 못했습니다. 첫 열이 X, 다음 열이 Y 인 CSV 나 Abaqus XY 보고서(.rpt)를 올려 주세요")
    return curves


def metrics(xo, yo, xr, yr):
    xo, yo, xr, yr = (np.asarray(a, float) for a in (xo, yo, xr, yr))
    o = np.argsort(xr, kind="stable")
    xr, yr = xr[o], yr[o]
    lo, hi = max(xo.min(), xr.min()), min(xo.max(), xr.max())
    m = (xo >= lo) & (xo <= hi)
    if hi <= lo or m.sum() < 2:
        return {"overlap": None}
    xs, ys = xo[m], yo[m]
    rs = np.interp(xs, xr, yr)
    scale = max(float(np.abs(rs).max()), 1e-30)
    po, pr = float(ys[np.argmax(np.abs(ys))]), float(rs[np.argmax(np.abs(rs))])
    order = np.argsort(xs)
    w = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    io, ir = float(w(ys[order], xs[order])), float(w(rs[order], xs[order]))
    return {
        "overlap": [float(lo), float(hi)],
        "peak": [po, pr],
        "peak_ratio": po / pr if pr else None,
        "mean": [io / (hi - lo), ir / (hi - lo)],
        "mean_ratio": io / ir if ir else None,
        "rms": float(np.sqrt(np.mean((ys - rs) ** 2)) / scale),
        "max_diff": float(np.abs(ys - rs).max() / scale),
    }


def auto_sign(yo, yr):
    a, b = float(np.mean(yo)), float(np.mean(yr))
    return -1.0 if a * b < 0 else 1.0
