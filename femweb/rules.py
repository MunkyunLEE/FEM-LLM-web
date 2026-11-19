import math
import re

import numpy as np

from .select import resolve

NUM = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
AX = "xyz"
G0 = 9.80665

UNITS = [
    ("density", r"kg\s*/\s*m\s*(?:3|³|\^3)", 1e-12),
    ("density", r"g\s*/\s*(?:cm\s*(?:3|³|\^3)|cc)", 1e-9),
    ("density", r"(?:tonne|ton|t)\s*/\s*mm\s*(?:3|³|\^3)", 1.0),
    ("pressure", r"kn\s*/\s*m\s*(?:2|²|\^2)", 1e-3),
    ("pressure", r"n\s*/\s*mm\s*(?:2|²|\^2)", 1.0),
    ("pressure", r"n\s*/\s*m\s*(?:2|²|\^2)", 1e-6),
    ("pressure", r"kgf\s*/\s*cm\s*(?:2|²|\^2)", 0.0980665),
    ("pressure", r"kgf\s*/\s*mm\s*(?:2|²|\^2)", 9.80665),
    ("pressure", r"gpa|기가\s*파스칼", 1e3),
    ("pressure", r"mpa|메가\s*파스칼", 1.0),
    ("pressure", r"kpa|킬로\s*파스칼", 1e-3),
    ("pressure", r"pa(?![a-z])|파스칼", 1e-6),
    ("pressure", r"bar(?![a-z])|바(?=\s|$)", 0.1),
    ("pressure", r"psi(?![a-z])", 0.00689476),
    ("pressure", r"atm(?![a-z])|기압", 0.101325),
    ("line", r"kn\s*/\s*m(?![a-z0-9²])", 1.0),
    ("line", r"n\s*/\s*mm(?![a-z0-9²])", 1.0),
    ("line", r"n\s*/\s*m(?![a-z0-9²])", 1e-3),
    ("line", r"kgf\s*/\s*mm(?![a-z0-9²])", 9.80665),
    ("line", r"kgf\s*/\s*m(?![a-z0-9²])", 0.00980665),
    ("moment", r"kn\s*[·*.\-]?\s*m(?![a-z])", 1e6),
    ("moment", r"n\s*[·*.\-]?\s*mm(?![a-z])", 1.0),
    ("moment", r"n\s*[·*.\-]?\s*m(?![a-z])", 1e3),
    ("moment", r"kgf\s*[·*.\-]?\s*cm(?![a-z])", 98.0665),
    ("moment", r"kgf\s*[·*.\-]?\s*m(?![a-z])", 9806.65),
    ("force", r"kn(?![a-z])|킬로\s*뉴턴", 1e3),
    ("force", r"kgf(?![a-z])|kg\s*중|킬로그램\s*힘", G0),
    ("force", r"tf(?![a-z])|ton(?:ne)?f?(?![a-z])|톤", 1000 * G0),
    ("force", r"lbf(?![a-z])", 4.44822),
    ("force", r"kg(?![a-z/])|킬로그램|킬로(?![가-힣])", G0),
    ("force", r"n(?![a-z/])|뉴턴", 1.0),
    ("length", r"mm(?![a-z])|밀리미터|밀리(?![가-힣])", 1.0),
    ("length", r"cm(?![a-z])|센티미터|센티(?![가-힣])", 10.0),
    ("length", r"(?:um|μm|µm)(?![a-z])|마이크로미터|미크론", 1e-3),
    ("length", r"inch(?:es)?|인치", 25.4),
    ("length", r"m(?![a-z0-9/])|미터", 1e3),
    ("angle", r"°|deg(?:rees?)?(?![a-z])|도(?![가-힣])", math.pi / 180),
    ("angle", r"rad(?:ians?)?(?![a-z])|라디안", 1.0),
    ("ratio", r"%|퍼센트|프로(?![가-힣])", 0.01),
    ("times", r"배(?![가-힣])|times(?![a-z])", 1.0),
]

MATERIALS = [
    (r"스테인리스|스텐|스테인레스|sus\s*\d*|stainless", "스테인리스강", 193000.0, 0.29, 8.0e-9),
    (r"강철|스틸|연강|탄소강|구조용\s*강|steel|강재|(?<![가-힣])강(?=\s*(?:으로|로|재료|$))", "강", 210000.0, 0.3, 7.85e-9),
    (r"알루미늄|알미늄|aluminium|aluminum|al\s*6061|a6061", "알루미늄", 70000.0, 0.33, 2.7e-9),
    (r"티타늄|titanium|ti-?6al", "티타늄", 110000.0, 0.34, 4.5e-9),
    (r"구리|동(?=\s*(?:으로|로|재료))|copper", "구리", 117000.0, 0.34, 8.96e-9),
    (r"황동|brass", "황동", 100000.0, 0.34, 8.5e-9),
    (r"콘크리트|concrete", "콘크리트", 30000.0, 0.2, 2.4e-9),
    (r"유리|glass", "유리", 70000.0, 0.22, 2.5e-9),
    (r"목재|나무|wood|timber", "목재", 11000.0, 0.3, 0.5e-9),
    (r"hdpe|pe\s*100|pe\s*80|폴리에틸렌|polyethylene", "PE", 1000.0, 0.42, 0.95e-9),
    (r"(?<![a-z])pp(?![a-z])|폴리프로필렌|polypropylene", "PP", 1500.0, 0.42, 0.9e-9),
    (r"pvc|폴리염화비닐", "PVC", 3000.0, 0.38, 1.4e-9),
    (r"(?<![a-z])pla(?![a-z])", "PLA", 3500.0, 0.36, 1.25e-9),
    (r"(?<![a-z])abs(?![a-z])", "ABS", 2300.0, 0.35, 1.05e-9),
    (r"아크릴|pmma|acryl", "아크릴", 3000.0, 0.37, 1.18e-9),
    (r"(?<![a-z])pet(?![a-z])|페트", "PET", 2800.0, 0.38, 1.38e-9),
    (r"종이|paper|카드보드|판지", "종이", 3000.0, 0.3, 0.8e-9),
    (r"cfrp|탄소\s*섬유|carbon\s*fib", "CFRP(등방 근사)", 70000.0, 0.3, 1.6e-9),
    (r"고무|rubber", "고무", 10.0, 0.49, 1.1e-9),
]

FILLER = (
    r"해\s*주세요|해\s*주십시오|해\s*줘요|해\s*줘라|해\s*줘|해\s*봐라|해\s*봐|해\s*라|해\s*요|합니다|한다|하자|할게|할래|하고\s*싶어|하고\s*싶다|하고|해서|하면|하여|하는|한\s*뒤|한\s*다음|한\s*후|해(?![가-힣])|"
    r"주세요|줘요|줘라|줘|주고|줄래|주자|걸어|걸고|가해|가하고|가한|넣어|넣고|적용|추가|설정|부여|입력|지정|만들어|바꿔|변경|수정|으로써|"
    r"으로|에서|에다가|에다|에게|부터|까지|이며|이고|이다|입니다|이야|인데|"
    r"뽑아|추출|켜고|켜서|켜|끄고|꺼|돌아가|되돌아가|올려|얹어|실어|매달아|구해|구하고|구하|찾아|계산|알려|보여|나머지|바꾸고|바꿔서|바꾸어|변경하고|설정하고|스팬|span|적당히|대충|약|대략|"
    r"좀|이제|우선|먼저|다음|그다음|그리고|또한|또|그냥|다시|혹시|제발|한번|전체적으로|각각|모두|전부|전체|모든|총|합계|합|크기|값|만큼|정도|짜리|"
    r"방향|쪽|측|면|변|끝단|끝|단|부분|부위|위치|지점|곳|절점들|절점|노드|엣지|모서리|라인|선|점|것|거|걸|게|수|때|중|"
    r"하중|힘|구속|조건|경계|해석|모델|구조|구조물|형상|재료|재질|단면|요소|메쉬|메시|"
    r"please|kindly|apply|add|set|make|put|use|with|the|an|to|at|on|of|in|by|is|as|be|it|its|this|that|and|then|for|from|into|a|"
    r"direction|side|face|edge|end|node|nodes|load|force|value|total|condition|boundary|model|analysis|all|"
    r"을|를|이|가|은|는|에|의|로|도|만|과|와|랑|및|인|임|함|들|요|야|다"
)


class Clause:
    def __init__(self, text):
        self.raw = text
        self.w = text.lower()

    def take(self, pat):
        m = re.search(pat, self.w)
        if m:
            self.w = self.w[: m.start()] + " " * (m.end() - m.start()) + self.w[m.end() :]
        return m

    def takeall(self, pat):
        out = []
        while True:
            m = self.take(pat)
            if not m:
                return out
            out.append(m)

    def has(self, pat):
        return re.search(pat, self.w)

    def blank(self, a, b):
        self.w = self.w[:a] + " " * (b - a) + self.w[b:]

    def leftover(self):
        s = re.sub(r"[\s\.\,\!\?\~\:\;\(\)\[\]\"'`·\-\+\=\/\\\*_<>→←^#@&|{}]+", " ", self.w)
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"(?<![a-z])(?:" + FILLER + r")(?![a-z])", " ", s)
        toks = [t for t in s.split() if t]
        return [t for t in toks if re.search(r"\d", t) or len(re.findall(r"[가-힣]", t)) >= 1 or len(re.findall(r"[a-z]", t)) >= 2]


def _f(m, i=1):
    return float(m.group(i))


def split_clauses(text):
    t = text.replace("−", "-").replace("–", "-").replace("×", "x").replace("，", ",")
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    t = re.sub(r"\(([^()]*)\)", lambda m: "(" + m.group(1).replace(",", "\x01") + ")", t)
    t = re.sub(r"\[([^\[\]]*)\]", lambda m: "(" + m.group(1).replace(",", "\x01") + ")", t)
    t = re.sub(r"(?<![A-Za-z])([xyzXYZ])\s*,\s*(?=[xyzXYZ](?![A-Za-z]))", "\\1\x01", t)
    t = re.sub(r"(?<![A-Za-z])([xyzXYZ])\s+and\s+(?=[xyzXYZ](?![A-Za-z]))", "\\1\x01", t)
    t = re.sub(r"(\d)\s*,\s*(?=[-+]?\d)", "\\1\x01", t)
    t = re.sub(r",\s*(?=각도|angle)", " ", t)
    t = re.sub(r",\s*(?=\d+\s*차)", "\x01 ", t)
    t = re.sub(r"[^\s,]+\s*(?:말고|대신에?|이?\s*아니라|이?\s*아니고)\s+", "", t)
    parts = re.split(r"[\n;]|,|(?<!\d)\.(?!\d)|\s그리고\s|\s또한\s|\s또\s|\sand then\s|\sthen\s|\sand\s|(?<=하고)\s+(?!나머지)|(?<=고정|구속|지지)\s+(?=(?:왼|오른|좌측|우측|우단|좌단|위|윗|아래|아랫|상단|하단|상부|하부|바닥|꼭대기|끝|자유단|반대|중앙|가운데|양단|양쪽|테두리|[+-]?[xyzXYZ]\s*(?:최|=|가\s|좌표|[+-]))\S*\s+\S)|(?<=잡고)\s+|(?<=막고)\s+|(?<=주고)\s+|(?<=걸고)\s+|(?<=넣고)\s+|(?<=키고)\s+|(?<=한 뒤)\s+|(?<=한 후)\s+|\s후에\s|한\s*다음에?\s", t)
    return [p.replace("\x01", ",").strip() for p in parts if p and p.strip()]


def quantities(c):
    out = []
    for kind, unit, scale in UNITS:
        for m in list(re.finditer(NUM + r"\s*(?:" + unit + r")", c.w)):
            if c.w[m.start() : m.end()].strip() == "":
                continue
            seg = c.w[m.start() : m.end()]
            if seg != m.group(0):
                continue
            out.append({"kind": kind, "value": float(m.group(1)) * scale, "pos": m.start(), "signed": m.group(1)[0] in "+-"})
            c.blank(m.start(), m.end())
    out.sort(key=lambda q: q["pos"])
    return out


def bare(c, near=None):
    if near:
        m = c.take(r"(?:" + near + r")\s*(?:을|를|은|는|이|가|=|:)?\s*" + NUM) or c.take(NUM + r"\s*(?:의)?\s*(?:" + near + r")")
        if m:
            return float(m.group(1))
    m = c.take(r"(?<![a-z\d])" + NUM + r"(?![a-z\d])")
    return float(m.group(1)) if m else None


def longest_axis(model):
    lo, hi = model.bbox()
    return int(np.argmax(hi - lo))


def vertical_axis(model):
    return 2


def sel_planes(sel):
    out = []
    if not sel:
        return out
    if "plane" in sel:
        out.append(sel["plane"])
    for k in ("and", "or"):
        for s in sel.get(k, []):
            out += sel_planes(s)
    return out


def fixed_plane(spec, axis=None):
    for b in spec["bcs"]:
        if len(b["dofs"]) >= 3:
            for p in sel_planes(b["select"]):
                if p.get("at") in ("min", "max") and (axis is None or p["axis"] == axis):
                    return p
    return None


def direction(c, model):
    m = c.take(r"(음|양|마이너스|플러스|negative|positive)\s*(?:의)?\s*([xyz])\s*(?:축)?\s*(?:의)?\s*(?:방향|direction)?(?:으로|로)?")
    if m:
        v = [0.0, 0.0, 0.0]
        v[AX.index(m.group(2))] = -1.0 if m.group(1) in ("음", "마이너스", "negative") else 1.0
        return {"vec": v, "literal": True}
    m = c.take(r"([-+]?)\s*([xyz])\s*(?:축)?\s*(?:의)?\s*(음|양|마이너스|플러스|\-|\+)?\s*(?:의)?\s*(?:방향|direction|쪽으로|축으로|축\s*방향)(?:으로|로)?")
    if m:
        v = [0.0, 0.0, 0.0]
        neg = (m.group(1) == "-") != (m.group(3) in ("음", "마이너스", "-"))
        v[AX.index(m.group(2))] = -1.0 if neg else 1.0
        return {"vec": v, "literal": True}
    m = c.take(r"(?<![a-z])in\s+([-+]?)([xyz])(?![a-z])|along\s+([-+]?)([xyz])(?![a-z])")
    if m:
        s, a = (m.group(1), m.group(2)) if m.group(2) else (m.group(3), m.group(4))
        v = [0.0, 0.0, 0.0]
        v[AX.index(a)] = -1.0 if s == "-" else 1.0
        return {"vec": v, "literal": True}
    words = [
        (r"아래\s*로|아래\s*방향(?:으로)?|아랫\s*방향(?:으로)?|하방(?:으로)?|하향(?:으로)?|연직\s*하(?:방|향)?(?:으로)?|수직\s*아래(?:로)?|밑으로|아래\s*$|downwards?|down(?![a-z])", [0.0, 0.0, -1.0]),
        (r"위\s*로|위\s*방향(?:으로)?|윗\s*방향(?:으로)?|상방(?:으로)?|상향(?:으로)?|(?<![가-힣])위\s*$|upwards?|(?<![a-z])up(?![a-z])", [0.0, 0.0, 1.0]),
        (r"왼쪽\s*으로|좌측\s*으로|왼쪽\s*방향(?:으로)?|leftwards?|to the left", [-1.0, 0.0, 0.0]),
        (r"오른쪽\s*으로|우측\s*으로|오른쪽\s*방향(?:으로)?|rightwards?|to the right", [1.0, 0.0, 0.0]),
        (r"앞\s*으로|앞쪽\s*으로|전방(?:으로)?|forwards?", [0.0, -1.0, 0.0]),
        (r"뒤\s*로|뒤쪽\s*으로|후방(?:으로)?|backwards?", [0.0, 1.0, 0.0]),
    ]
    for pat, v in words:
        if c.take(pat):
            return {"vec": v, "literal": False}
    return None


def loose_axis(c):
    m = c.take(r"(?<![a-z\d])([-+]?)\s*([xyz])(?![a-z\d])")
    if not m:
        return None
    v = [0.0, 0.0, 0.0]
    v[AX.index(m.group(2))] = -1.0 if m.group(1) == "-" else 1.0
    return {"vec": v, "literal": True}


def tuple3(c, after=""):
    m = c.take(r"\(\s*" + NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\)" + after)
    return m


def _plane(axis, at):
    return {"plane": {"axis": axis, "at": at}}


def selection(c, model, spec, ctx):
    lo, hi = model.bbox()
    span = float(np.linalg.norm(hi - lo)) or 1.0
    planes = []
    others = []
    notes = []
    m = c.take(r"(?:좌표|점|절점|위치|node|point)?\s*\(\s*" + NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\)\s*(?:mm)?\s*(?:근처|부근|주변|에\s*가장\s*가까운)?")
    if m:
        others.append({"nearest": [_f(m, 1), _f(m, 2), _f(m, 3)]})
    m = c.take(r"(?:절점|노드|node)\s*(?:번호|id|#)?\s*((?:\d+\s*(?:,|와|과|및)?\s*)+)(?:번)?")
    if m:
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if all(0 <= i < len(model.nodes) for i in ids):
            others.append({"nodes": ids})
        else:
            notes.append(f"절점 번호는 0~{len(model.nodes) - 1} 범위여야 합니다")
    for m in c.takeall(r"([xyz])\s*(?:좌표)?\s*(?:가|이|는)?\s*" + NUM + r"\s*(?:mm)?\s*(?:~|에서|부터|-)\s*" + NUM + r"\s*(?:mm)?\s*(?:사이|구간|범위|까지)?(?:인|의)?"):
        a = AX.index(m.group(1))
        b0 = (lo - span).tolist()
        b1 = (hi + span).tolist()
        b0[a], b1[a] = sorted((_f(m, 2), _f(m, 3)))
        others.append({"box": {"min": [round(v, 6) for v in b0], "max": [round(v, 6) for v in b1]}})
    for m in c.takeall(r"([xyz])\s*(?:좌표)?\s*(?:가|이|는)?\s*(?:(>=|<=|>|<|≥|≤)\s*" + NUM + r"|" + NUM + r"\s*(?:mm)?\s*(이상|이하|초과|미만|보다\s*큰|보다\s*작은))\s*(?:인|의)?"):
        a = AX.index(m.group(1))
        op = m.group(2) or m.group(5)
        v = float(m.group(3) or m.group(4))
        b0 = (lo - span).tolist()
        b1 = (hi + span).tolist()
        if op in (">=", ">", "≥", "이상", "초과") or op.startswith("보다 큰") or op.replace(" ", "") == "보다큰":
            b0[a] = v
        else:
            b1[a] = v
        others.append({"box": {"min": [round(x, 6) for x in b0], "max": [round(x, 6) for x in b1]}})
    for m in c.takeall(r"([-+])\s*([xyz])\s*(?:쪽|측)?\s*(?:면|끝단|끝|변|단|엣지|모서리|side|face|end|edge)"):
        planes.append((m.group(2), "max" if m.group(1) == "+" else "min"))
    for m in c.takeall(r"(?<![a-z])([xyz])\s*([-+])\s*(?:쪽|측)?\s*(?:면|끝단|끝|변|단|엣지|모서리|side|face|end|edge)"):
        planes.append((m.group(1), "max" if m.group(2) == "+" else "min"))
    for m in c.takeall(r"(?<![a-z])([xyz])\s*(?:좌표)?\s*(?:축|방향)?\s*(?:가|이|의|=|:)?\s*(최소|최대|최솟값|최댓값|가장\s*작은|가장\s*큰|제일\s*작은|제일\s*큰|min(?:imum)?|max(?:imum)?|중간|mid(?:dle)?|중앙)(?:값)?(?:인|이\s*되는)?"):
        w = m.group(2)
        at = "min" if re.match(r"최소|최솟|가장\s*작|제일\s*작|min", w) else "max" if re.match(r"최대|최댓|가장\s*큰|제일\s*큰|max", w) else "mid"
        planes.append((m.group(1), at))
    for m in c.takeall(r"(?<![a-z])(min(?:imum)?|max(?:imum)?)\s*(?:imum)?\s*([xyz])(?![a-z])"):
        planes.append((m.group(2), "min" if m.group(1).startswith("min") else "max"))
    for m in c.takeall(r"(?<![a-z])([xyz])\s*(?:좌표)?\s*(?:=|가|이)\s*" + NUM + r"\s*(?:mm)?\s*(?:인|위치|지점|평면|단면|에서)?"):
        planes.append((m.group(1), _f(m, 2)))
    for m in c.takeall(r"(?<![a-z])([xyz])\s*" + NUM + r"\s*(?:mm)?\s*(?:위치|지점|평면|단면|인\s*곳)"):
        planes.append((m.group(1), _f(m, 2)))
    L = AX[longest_axis(model)]
    m = c.take(r"(?<![a-z])([xyz])\s*(?:축|방향)?\s*(?:의)?\s*(?:양단|양\s*끝단|양\s*끝|양쪽\s*끝|양쪽|both ends)")
    if m:
        others.append({"or": [_plane(m.group(1), "min"), _plane(m.group(1), "max")]})
    elif c.take(r"양단|양\s*끝단|양\s*끝|양쪽\s*끝|양쪽\s*변|양쪽|both ends|two ends"):
        others.append({"or": [_plane(L, "min"), _plane(L, "max")]})
    if c.take(r"(?:고정\s*단|고정\s*된\s*(?:쪽|끝|면)|fixed end)\s*(?:의|과|와)?\s*(?:반대\s*(?:쪽|편|측)?|맞은\s*편|opposite)"):
        ref = fixed_plane(spec)
        if ref:
            planes.append((ref["axis"], "max" if ref["at"] == "min" else "min"))
        else:
            notes.append("고정단으로 볼 구속이 아직 없습니다")
    if c.take(r"반대\s*쪽|반대\s*편|반대\s*측|맞은\s*편|다른\s*쪽\s*끝|다른\s*쪽|opposite(?:\s*(?:end|side))?"):
        ref = None
        if ctx.get("sel"):
            ps = [p for p in sel_planes(ctx["sel"]) if p.get("at") in ("min", "max")]
            ref = ps[0] if ps else None
        ref = ref or fixed_plane(spec)
        if ref:
            planes.append((ref["axis"], "max" if ref["at"] == "min" else "min"))
        else:
            notes.append("'반대쪽'의 기준이 되는 구속이나 선택이 없습니다")
    if c.take(r"고정\s*단|고정\s*된\s*(?:쪽|끝|면)|fixed end"):
        ref = fixed_plane(spec)
        if ref:
            planes.append((ref["axis"], ref["at"]))
        else:
            notes.append("고정단으로 볼 구속이 아직 없습니다")
    if c.take(r"자유\s*단|끝\s*단|선\s*단|자유\s*끝|free end|(?<![a-z])tip(?![a-z])"):
        ref = fixed_plane(spec, L) or fixed_plane(spec)
        if ref is None and ctx.get("sel"):
            ps = [p for p in sel_planes(ctx["sel"]) if p.get("at") in ("min", "max")]
            ref = ps[0] if ps else None
        if ref:
            planes.append((ref["axis"], "max" if ref["at"] == "min" else "min"))
        else:
            planes.append((L, "max"))
    ext = hi - lo
    V = "z" if ext[2] > 1e-6 * span else "y"
    if c.take(r"한\s*쪽\s*(?:끝단|끝|면|변)?|한\s*끝|one end|one side"):
        planes.append((L, "min"))
        ctx.setdefault("info", []).append(f"'한쪽'을 {L} 최소 쪽으로 보았습니다")
    words = [
        (r"바닥\s*면|바닥|하단|하부|아래\s*쪽|아랫\s*면|아랫\s*쪽|아래\s*면|밑\s*면|밑\s*바닥|맨\s*아래|bottom|아래(?!로|\s*방향|\s*$)|(?<![가-힣])밑(?=\s|에|을|은|의|쪽|$)", V, "min"),
        (r"상단|상부|윗\s*면|위\s*쪽|윗\s*쪽|위\s*면|꼭대기|맨\s*위|천장|top(?![a-z])|(?<![가-힣])위(?=\s+\S|에|를|는|은|의|쪽)", V, "max"),
        (r"왼\s*쪽|좌측|좌단|왼\s*편|left(?![a-z])", "x", "min"),
        (r"오른\s*쪽|우측|우단|오른\s*편|right(?![a-z])", "x", "max"),
        (r"앞\s*쪽|앞\s*면|전면|정면|front(?![a-z])", "y", "min"),
        (r"뒤\s*쪽|뒷\s*면|뒷\s*쪽|후면|배면|back(?![a-z])|rear(?![a-z])", "y", "max"),
    ]
    for pat, ax, at in words:
        if c.take(pat):
            planes.append((ax, at))
    if not planes and not others and c.take(r"(?<![가-힣])끝(?:\s*부분)?(?=에|을|은|의|\s|$)"):
        ref = fixed_plane(spec, L) or fixed_plane(spec)
        if ref is None and ctx.get("sel"):
            ps = [p for p in sel_planes(ctx["sel"]) if p.get("at") in ("min", "max")]
            ref = ps[0] if ps else None
        planes.append((ref["axis"], "max" if ref["at"] == "min" else "min") if ref else (L, "max"))
    if c.take(r"네\s*변|4\s*변|사\s*변|모든\s*변|전체\s*변|둘레|테두리|가장자리|외곽\s*(?:선|엣지|변)?|자유\s*(?:엣지|변)|경계\s*(?:엣지|변|선)|all\s*(?:four\s*)?edges|four edges|perimeter|outer edges?|free edges?"):
        others.append({"boundary": True})
    if c.take(r"네\s*(?:모서리|꼭짓점|꼭지점|귀퉁이|코너)|4\s*(?:모서리|꼭짓점|코너)|꼭짓점들?|꼭지점들?|귀퉁이|코너|(?:four\s*)?corners?"):
        if not planes:
            others.append({"corners": True})
    if not planes and not others and c.take(r"모서리들"):
        others.append({"corners": True})
    m = c.take(r"(정\s*중앙|중앙|가운데|한\s*가운데|중심|center|centre|mid\s*-?\s*span|midpoint|middle)\s*(단면|선|라인|line|section)?")
    if m:
        if m.group(2):
            planes.append((L, "mid"))
        elif planes:
            ctr = 0.5 * (lo + hi)
            pt = ctr.copy()
            for ax, at in planes:
                a = AX.index(ax)
                pt[a] = lo[a] if at == "min" else hi[a] if at == "max" else ctr[a] if at == "mid" else float(at)
            others.append({"nearest": [round(float(v), 6) for v in pt]})
            planes = []
        else:
            others.append({"nearest": [round(float(v), 6) for v in 0.5 * (lo + hi)]})
    names = sorted(model.group_names(), key=len, reverse=True)
    groups = []
    for g in names:
        if g == "default":
            continue
        pat = r"(?<![a-z0-9_])" + re.escape(g.lower()) + r"(?![a-z0-9_])"
        if c.take(pat):
            groups.append(g)
    for m in c.takeall(r"(?:face|면)\s*_?\s*(\d+)(?:\s*번)?|(\d+)\s*번\s*면"):
        g = "face_" + (m.group(1) or m.group(2))
        if g in model.group_names() and g not in groups:
            groups.append(g)
        elif g not in model.group_names():
            notes.append(f"{g} 그룹이 없습니다")
    if groups:
        c.take(r"그룹|group|파트|part")
        others.append({"group": groups[0]} if len(groups) == 1 else {"or": [{"group": g} for g in groups]})
    by_axis = {}
    for ax, at in planes:
        lst = by_axis.setdefault(ax, [])
        if at not in lst:
            lst.append(at)
    items = []
    for ax, ats in by_axis.items():
        items.append(_plane(ax, ats[0]) if len(ats) == 1 else {"or": [_plane(ax, a) for a in ats]})
    items += others
    if not items:
        if c.take(r"모든\s*절점|전체\s*절점|전\s*절점|전체\s*에|모델\s*전체|구조\s*전체|전체\s*면|all nodes|everything|whole (?:model|structure)|entire (?:model|structure)"):
            return {"all": True}, notes
        return None, notes
    return (items[0] if len(items) == 1 else {"and": items}), notes


def axes_list(s):
    return [AX.index(ch) + 1 for ch in re.findall(r"[xyz]", s)]


def bc_dofs(c):
    d = set()
    for m in c.takeall(r"(?<![a-z])(u|r|ur|rot)\s*([xyz123])(?![a-z0-9])"):
        k = AX.index(m.group(2)) if m.group(2) in AX else int(m.group(2)) - 1
        d.add(k + 1 if m.group(1) == "u" else k + 4)
    m = c.take(r"(?:자유도|dofs?)\s*((?:[1-6]\s*(?:,|와|과|및|and)?\s*)+)(?:번)?")
    if m:
        d.update(int(x) for x in re.findall(r"[1-6]", m.group(1)))
    for m in c.takeall(r"(?<![a-z])((?:[xyz]\s*(?:,|와|과|및|랑|and|/|·)?\s*){1,3})(?:축\s*)?(?:의\s*)?(?:회전|rotations?|모멘트)"):
        d.update(k + 3 for k in axes_list(m.group(1)))
    for m in c.takeall(r"(?<![a-z])((?:[xyz]\s*(?:,|와|과|및|랑|and|/|·)?\s*){1,3})(?:축\s*)?(?:의\s*)?(?:방향|변위|병진|이동|direction|translations?|displacements?)(?:\s*(?:변위|병진|이동))?(?:으로|로)?"):
        d.update(axes_list(m.group(1)))
    for m in c.takeall(r"(?<![a-z])((?:[xyz]\s*(?:,|와|과|및|랑|/|·)?\s*){1,3})(?=\s*(?:축)?\s*(?:만|은|는|을|를)?\s*(?:구속|고정|지지|막|잡|constrain|fix|restrain))"):
        d.update(axes_list(m.group(1)))
    if c.take(r"(?:모든|전체|전)?\s*회전\s*(?:자유도|방향)?(?=\s*(?:을|를|도|만|까지)?\s*(?:구속|고정|막|잡))|all rotations"):
        d.update((4, 5, 6))
    if c.take(r"(?:모든|전체|전)?\s*병진\s*(?:자유도|방향)?|all translations"):
        d.update((1, 2, 3))
    return sorted(d)


def edge_length(model, ids):
    s = set(int(i) for i in ids)
    total = 0.0
    seen = set()
    for a, b in model.beams:
        if a in s and b in s:
            seen.add((min(a, b), max(a, b)))
    for e in model.shell_edges():
        if e[0] in s and e[1] in s:
            seen.add(e)
    for a, b in seen:
        total += float(np.linalg.norm(model.nodes[a] - model.nodes[b]))
    if total == 0.0 and len(ids) > 1:
        p = model.nodes[list(s)]
        total = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    return total


def inward(model, sel):
    ps = [p for p in sel_planes(sel) if p.get("at") in ("min", "max")] if sel else []
    if len(ps) == 1 and "or" not in (sel or {}):
        v = [0.0, 0.0, 0.0]
        v[AX.index(ps[0]["axis"])] = 1.0 if ps[0]["at"] == "min" else -1.0
        return v
    if sel and "group" in sel and sel["group"] in model.patches:
        n = model.patches[sel["group"]].get("normal")
        if n is not None and float(np.linalg.norm(n)) > 0.5:
            return [-float(x) for x in n]
    return None


def fmt_vec(v):
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


def scale_word(c):
    m = c.take(NUM + r"\s*배(?:로)?")
    if m:
        return _f(m)
    if c.take(r"두\s*배(?:로)?|twice|double"):
        return 2.0
    if c.take(r"세\s*배(?:로)?|triple"):
        return 3.0
    if c.take(r"절반(?:으로)?|반\s*으로|half|halve"):
        return 0.5
    m = c.take(NUM + r"\s*(?:%|퍼센트)\s*(?:로|으로)?\s*(증가|늘려|키워|감소|줄여|increase|decrease)?")
    if m:
        p = _f(m) / 100
        w = m.group(2) or ""
        return 1 + p if re.match(r"증가|늘려|키워|increase", w) else 1 - p if re.match(r"감소|줄여|decrease", w) else p
    return None


HELP = (
    "이런 문장을 바로 처리합니다. 구속: 'x 최소 면 고정', '양단 단순지지', '테두리를 z 방향만 구속', 'x 대칭'. "
    "하중: '끝단에 아래로 2 kN', '윗면에 압력 0.5 MPa', 'x 최대 변에 선하중 10 N/mm 아래로', '자중'. "
    "강제변위: 'x 최대 면을 x 방향으로 -5 mm'. 재료·단면: '알루미늄으로', '두께 2 mm', '보 단면 20x40'. "
    "힌지: '쉘 사이 공유 엣지를 핀 조인트로', '힌지 강성 0.05'. 해석: '좌굴 모드 4개', '비선형 증분 30', '해석 실행'. "
    "접촉: '반지름 20 구를 아래로 3 mm 눌러, 마찰 0.2', '바닥에 강체 평면 접촉', 'body_1 과 body_2 사이 접촉 마찰 0.3', '마찰계수 0.1', '접촉 삭제'. "
    "좌굴 후 거동: '변위제어 5 mm', '초기결함 0.1', '강체 압판으로 변위제어', '하중제어로', '분기 전환 꺼줘'. "
    "그래프: '시간-총에너지 그래프', '하중-변위 곡선'. 수정: 'ld1 2배', 'bc2 삭제', '되돌려'."
)


def parse_plot(c, model, spec, ctx):
    t = c.w
    if re.search(r"좌굴\s*계수|buckling factor", t):
        return {"op": "add_plot", "title": "좌굴 계수", "x": {"quantity": "mode"}, "series": [{"quantity": "buckling.factor"}]}
    if re.search(r"고유\s*진동수|진동수|주파수|frequenc", t):
        return {"op": "add_plot", "title": "고유진동수", "x": {"quantity": "mode"}, "series": [{"quantity": "frequency"}]}
    ld = re.search(r"하중\s*[-–~/]?\s*변위|힘\s*[-–~/]?\s*변위|반력\s*[-–~/]?\s*변위|force\s*-?\s*disp|load\s*-?\s*disp", t)
    axm = re.search(r"(?<![a-z])([xyz])\s*(?:축|방향)?\s*(?:의)?\s*(?:변위|처짐|반력|displacement|reaction)", t) or re.search(r"(?:반력|reaction)\s*([xyz])(?![a-z])", t)
    cc = Clause(t)
    for pat in (r"그래프|플롯|plot|graph|그려|곡선|curve|이력|history|보여|차트|chart", r"시간|time|하중\s*계수|증분"):
        cc.takeall(pat)
    direction(cc, model)
    sel, _ = selection(cc, model, spec, ctx)
    series = []
    title = []
    if ld and spec["step"].get("control") == "displacement" and not axm:
        return {"op": "add_plot", "title": "하중-변위 곡선 (변위 제어)", "x": {"quantity": "control.disp"}, "series": [{"quantity": "control.force"}]}
    if ld:
        a = axm.group(1) if axm else None
        if a is None:
            mv = [b for b in spec["bcs"] if abs(b.get("value", 0.0)) > 0 and b["dofs"][0] <= 3]
            a = AX[mv[0]["dofs"][0] - 1] if mv else "z"
        x = {"quantity": "u" + a, "select": sel, "reduce": "mean"} if sel else {"quantity": "u.max"}
        return {"op": "add_plot", "title": "하중-변위", "x": x, "series": [{"quantity": f"reaction.{a}"}]}
    if re.search(r"에너지|energy", t):
        hit = False
        for pat, key in ((r"접힘|fold", "energy.fold"), (r"바\s*에너지|(?<![a-z])bar(?![a-z])|신장|스트레칭|stretch", "energy.bar"), (r"굽힘|bend", "energy.bend")):
            if re.search(pat, t):
                series.append({"quantity": key})
                hit = True
        if not hit or re.search(r"총|전체|토탈|total|변형", t):
            series.insert(0, {"quantity": "energy.total"})
        title.append("에너지")
    if re.search(r"접촉\s*(?:수직)?\s*(?:력|반력|하중)|contact\s*force", t):
        series.append({"quantity": "contact.normal"})
        if re.search(r"마찰|friction", t):
            series.append({"quantity": "contact.friction"})
        title.append("접촉력")
    elif re.search(r"마찰\s*력|friction\s*force", t):
        series.append({"quantity": "contact.friction"})
        title.append("마찰력")
    elif re.search(r"반력|reaction", t):
        a = axm.group(1) if axm else "z"
        b = re.search(r"(?<![a-z])(bc\d+)", t)
        series.append({"quantity": f"reaction.{b.group(1)}.{a}" if b else f"reaction.{a}"})
        title.append("반력" + (f" {b.group(1)}" if b else ""))
    if re.search(r"응력|stress|미세스|미제스|mises", t):
        series.append({"quantity": "vonMises", "select": sel, "reduce": "max"} if sel else {"quantity": "vonMises.max"})
        title.append("응력")
    if re.search(r"변위|처짐|displacement|deflection", t):
        a = re.search(r"(?<![a-z])([xyz])\s*(?:축|방향)?\s*(?:의)?\s*(?:변위|처짐|displacement|deflection)", t)
        q = "u" + a.group(1) if a else "u"
        series.append({"quantity": q, "select": sel, "reduce": "mean" if a else "max"} if sel else {"quantity": "u.max"})
        title.append("변위")
    if re.search(r"접힘\s*각|fold angle", t):
        series.append({"quantity": "fold.mean"})
        title.append("접힘각")
    if not series:
        return None
    return {"op": "add_plot", "title": " · ".join(title) + " 이력", "x": {"quantity": "time"}, "series": series}


def parse_contact(c, model, spec):
    cts = spec.get("contacts") or []
    idm = c.take(r"(?<![a-z])ct\s*(\d+)(?![a-z0-9])")
    target = "ct" + idm.group(1) if idm else None
    if target and not any(x["id"] == target for x in cts):
        return [], [f"{target} 항목이 없습니다"], [], [], {}
    if c.take(r"삭제|지워|제거|없애|빼\s*줘|remove|delete|drop|해제"):
        c.takeall(r"접촉|contacts?|조건|모두|전부|전체|모든|all|강체|구(?![가-힣])|평면")
        if target:
            return [{"op": "remove", "id": target}], [], [], c.leftover(), {}
        if not cts:
            return [], ["삭제할 접촉이 없습니다"], [], [], {}
        return [{"op": "clear", "what": "contacts"}], [], [], c.leftover(), {}
    selfc = bool(c.take(r"자기\s*접촉|자체\s*접촉|셀프\s*접촉|self\s*-?\s*contact"))
    mu = None
    if c.take(r"마찰\s*(?:이|은|는)?\s*없(?:이|는|음|게)|무마찰|frictionless|no friction|without friction"):
        mu = 0.0
    else:
        m = c.take(r"(?:마찰\s*계수|마찰|friction(?:\s*coefficient)?|(?<![a-z])mu(?![a-z])|μ)\s*(?:는|은|=|를|을|이|가|:)?\s*" + NUM)
        if m:
            mu = _f(m)
    if selfc:
        c.takeall(r"접촉|contacts?|추가|넣어|설정|켜|적용|해\s*줘|해\s*라|줘|모든|면끼리|포함|조건|으로|로|를|을|도")
        old = next((x for x in cts if x["type"] == "self"), None)
        if old:
            return [{"op": "update", "id": old["id"], "friction": mu if mu is not None else old.get("friction", 0.0)}], [], [], c.leftover(), {}
        return [{"op": "add_contact", "type": "self", "friction": mu or 0.0}], [], [], c.leftover(), {}
    rad = None
    m = c.take(r"(?:반지름|반경|radius|(?<![a-z])r)\s*(?:=|이|가|은|는|:)?\s*" + NUM + r"\s*(mm|cm|m)?(?![a-z])")
    if m:
        rad = _f(m) * {"mm": 1.0, "cm": 10.0, "m": 1000.0, None: 1.0}[m.group(2)]
    else:
        m = c.take(r"(?:지름|직경|diameter|(?<![a-z])d|φ|ø)\s*(?:=|이|가|은|는|:)?\s*" + NUM + r"\s*(mm|cm|m)?(?![a-z])")
        if m:
            rad = 0.5 * _f(m) * {"mm": 1.0, "cm": 10.0, "m": 1000.0, None: 1.0}[m.group(2)]
    cen = None
    m = c.take(r"(?:중심|center|centre)\s*(?:은|는|이|가|을|를|=|:|at)?\s*\(?\s*" + NUM + r"\s*[, ]\s*" + NUM + r"\s*[, ]\s*" + NUM + r"\s*\)?")
    if m:
        cen = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    sphere = c.take(r"강체\s*구|구체|(?<![가-힣])구(?=를|로|가|\s|$)|sphere|(?<![a-z])ball(?![a-z])|인덴터|indenter") is not None
    floor = c.take(r"바닥|지면|floor|ground") is not None
    plane = c.take(r"강체\s*(?:평면|판|벽|면)|평면|(?<![a-z])plane(?![a-z])|rigid\s*(?:plate|wall|surface)|압판|platen") is not None or floor
    pair_word = c.take(r"두\s*(?:물체|바디|몸체|부품|파트)|물체\s*(?:사이|끼리|간)|서로|between|each other|사이") is not None
    names = model.group_names() if model is not None else []
    found = []
    for g in sorted(names, key=len, reverse=True):
        if c.take(r"(?<![a-z0-9_])" + re.escape(g.lower()) + r"(?![a-z0-9_])"):
            found.append(g)
    if len(found) == 1 and not (sphere or plane or pair_word or rad is not None) and c.has(r"강체|rigid"):
        c.takeall(r"강체\s*(?:로|처럼|화)?|rigid(?:\s*body)?")
        d = direction(c, model) or loose_axis(c)
        m = c.take(NUM + r"\s*(mm|cm|m)(?![a-z])\s*(?:만큼|까지|씩)?")
        dist = abs(_f(m)) * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[m.group(2)] if m else 0.0
        c.takeall(r"부품|파트|그룹|을|를|은|는|만|으로|로|설정|지정|만들어|해\s*줘|해줘|두고|하고|취급")
        moving = c.take(r"눌러|누르|압입|밀어|내려|올려|이동|움직|press|push|move") is not None
        contact = c.take(r"접촉\w*|contact|고정\s*(?:된|한)?\s*(?:받침|지지)?|받침|지지\w*|support") is not None
        if dist > 0 or moving or contact:
            v = d["vec"] if d else [0.0, 0.0, -1.0]
            op = {"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": found[0]}, "motion": [x * dist + 0.0 for x in v], "friction": mu or 0.0}
            return [op], [], [], c.leftover(), {}
        c.takeall(r"묶어\w*|묶고|연결\w*")
        return [{"op": "add_coupling", "select": {"group": found[0]}, "rigid": True, "name": f"강체 {found[0]}"}], [], [], c.leftover(), {}
    if not (sphere or plane or pair_word or len(found) >= 2 or rad is not None):
        if mu is not None:
            c.takeall(r"접촉|contact|으로|로|바꿔|변경|수정|설정|적용")
            if target:
                return [{"op": "update", "id": target, "friction": mu}], [], [], c.leftover(), {}
            return [{"op": "_friction", "friction": mu}], [], [], c.leftover(), {}
        if target or c.has(r"접촉|contact"):
            return [], ["어떤 접촉인지 알려주세요 (예: 반지름 20 구를 아래로 3 mm 눌러 / 바닥에 강체 평면 접촉 / body_1 과 body_2 사이 접촉, 마찰 0.3)"], [], [], {}
        return None
    c.takeall(r"접촉|(?<![a-z])contacts?(?![a-z])|강체|rigid|조건|추가|정의|설정|걸어|넣어|만들어|add|define|with")
    if sphere or rad is not None and not plane:
        if rad is None:
            return [], ["구의 반지름을 알려주세요 (예: 반지름 20 구를 아래로 3 mm 눌러)"], [], [], {}
        d = direction(c, model) or loose_axis(c)
        m = c.take(NUM + r"\s*(mm|cm|m)(?![a-z])\s*(?:만큼|까지|씩)?")
        dist = abs(_f(m)) * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[m.group(2)] if m else 0.0
        c.takeall(r"눌러|누르|압입|밀어|내려|이동|움직|찍어|press|push|indent|move|위에서|위에|by|into")
        v = d["vec"] if d else [0.0, 0.0, -1.0]
        op = {"op": "add_contact", "type": "rigid", "shape": "sphere", "radius": rad, "motion": [x * dist + 0.0 for x in v], "friction": mu or 0.0}
        if cen:
            op["center"] = cen
        notes = [] if dist > 0 else ["이동량이 없어 구가 제자리에 있습니다. '아래로 3 mm 눌러'처럼 이동량을 주면 압입합니다"]
        return [op], [], notes, c.leftover(), {}
    if plane:
        d = direction(c, model) or loose_axis(c)
        m = c.take(NUM + r"\s*(mm|cm|m)(?![a-z])\s*(?:만큼|까지|씩)?")
        dist = abs(_f(m)) * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[m.group(2)] if m else 0.0
        c.takeall(r"눌러|누르|압입|밀어|내려|올려|이동|움직|press|push|move|위에서|위에|아래에|밑에|에|by|into")
        if d:
            v = d["vec"]
        elif floor:
            v = [0.0, 0.0, 1.0]
        else:
            v = [0.0, 0.0, -1.0]
        op = {"op": "add_contact", "type": "rigid", "shape": "plane", "normal": v, "motion": [x * dist + 0.0 for x in v], "friction": mu or 0.0}
        return [op], [], [], c.leftover(), {}
    bodies = [g for g in names if g.startswith("body_")]
    if len(found) >= 2:
        a, b = found[0], found[1]
        pos = {g: c.w.find(g.lower()) for g in found}
    elif len(bodies) == 2 and not found:
        a, b = bodies[1], bodies[0]
    else:
        return [], ["접촉시킬 두 그룹을 알려주세요. 사용 가능: " + (", ".join(names) or "없음")], [], [], {}
    return [{"op": "add_contact", "type": "pair", "a": {"group": a}, "b": {"group": b}, "friction": mu or 0.0}], [], [], c.leftover(), {}


CP_WORD = r"기준점|커플링|coupling|reference\s*point|(?<![a-z])rp(?![a-z])|(?<![a-z])cp\s*\d+"


def parse_coupling(c, model, spec, ctx):
    cps = spec.get("couplings") or []
    idm = c.take(r"(?<![a-z])cp\s*(\d+)(?![a-z0-9])")
    target = "cp" + idm.group(1) if idm else None
    if target and not any(x["id"] == target for x in cps):
        return [], [f"{target} 기준점이 없습니다"], [], [], {}
    create = c.has(r"묶|연결|만들|추가|생성|붙여|커플링\s*(?:해|걸|추가)|couple|tie|create|add")
    if create and not target:
        c.takeall(r"기준점\s*(?:에|으로|로|과|와)?|커플링|coupling|reference\s*point|(?<![a-z])rp(?![a-z])|(?:강체\s*(?:로|처럼)?)")
        sel, notes = selection(c, model, spec, ctx)
        if not sel:
            return [], ["기준점에 묶을 면을 알려주세요 (예: 위 끝면을 기준점에 묶어줘)"], [], [], {}
        c.takeall(r"묶어\w*|묶고|묶기|연결\w*|만들어\w*|추가\w*|생성\w*|붙여\w*|해\s*줘|해줘|을|를|에|면|끝면|단면")
        return [{"op": "add_coupling", "select": sel}], [], notes, c.leftover(), {}
    if not cps:
        return [], ["기준점이 없습니다. 먼저 '위 끝면을 기준점에 묶어줘'처럼 만들어 주세요"], [], [], {}
    if target is None:
        target = cps[-1]["id"]
    c.takeall(r"기준점\s*(?:의|에|을|를|은|는)?|(?<![a-z])rp(?![a-z])\s*(?:의|에)?")
    op = {"op": "update", "id": target}
    deg = None
    m = c.take(r"(?<![a-z])([xyz])\s*축?\s*(?:둘레|기준|방향|중심)?\s*(?:로|으로|에|의)?\s*(?:모멘트|moment|토크|torque|비틀\w*)\s*(?:를|을)?\s*" + NUM + r"\s*(kn\s*[·.*]?\s*m|n\s*[·.*]?\s*m(?!m)|n\s*[·.*]?\s*mm|nmm)?")
    if not m:
        m2 = c.take(r"(?:모멘트|moment|토크|torque)\s*" + NUM + r"\s*(kn\s*[·.*]?\s*m|n\s*[·.*]?\s*m(?!m)|n\s*[·.*]?\s*mm|nmm)?\s*(?:를|을)?\s*(?<![a-z])([xyz])\s*축?")
        if m2:
            m = (m2.group(3), m2.group(1), m2.group(2))
    else:
        m = (m.group(1), m.group(2), m.group(3))
    if m:
        unit = (m[2] or "nmm").replace(" ", "").replace("·", "").replace(".", "").replace("*", "")
        f = {"knm": 1e6, "nm": 1e3, "nmm": 1.0}.get(unit, 1.0)
        v = [0.0, 0.0, 0.0]
        v[AX.index(m[0])] = float(m[1]) * f
        op["moment"] = v
    m = c.take(r"(?<![a-z])([xyz])\s*축?\s*(?:둘레|기준|중심)?\s*(?:로|으로|에)?\s*" + NUM + r"\s*(rad|라디안|도|deg|°)\s*(?:만큼)?\s*(?:강제\s*)?(?:회전|돌려|rotat\w*)")
    if m:
        ang = float(m.group(2)) * (math.pi / 180.0 if m.group(3) in ("도", "deg", "°") else 1.0)
        op["fix"] = {str(AX.index(m.group(1)) + 4): ang}
    else:
        m = c.take(r"(?<![a-z])([-+]?)\s*([xyz])\s*(?:축)?\s*(?:방향)?\s*(?:으로|로)?\s*" + NUM + r"\s*(mm)\s*(?:만큼)?\s*(?:강제\s*변위|이동|밀어|눌러|당겨|변위)?")
        if m:
            val = float(m.group(3)) * (-1.0 if m.group(1) == "-" else 1.0)
            op.setdefault("fix", {})[str(AX.index(m.group(2)) + 1)] = val
    m = c.take(r"(?<![a-z])([-+]?)\s*([xyz])\s*(?:축)?\s*(?:방향)?\s*(?:으로|로|에)?\s*(?:힘|하중|force)?\s*" + NUM + r"\s*(kn|n)(?![a-z])")
    if m:
        val = float(m.group(3)) * (1e3 if m.group(4) == "kn" else 1.0) * (-1.0 if m.group(1) == "-" else 1.0)
        v = [0.0, 0.0, 0.0]
        v[AX.index(m.group(2))] = val
        op["force"] = v
    if c.take(r"(?:모든|전체)?\s*(?:자유도\s*)?(?:해제|풀어|free|release)"):
        op["fix"] = {}
        op["free"] = [1, 2, 3, 4, 5, 6]
    elif c.has(r"구속|고정|막|잡|fix|constrain"):
        d = bc_dofs(c)
        if c.take(r"완전\s*고정|모두\s*고정|전부\s*고정|고정\s*해|고정(?!\s*단)") and not d:
            d = [1, 2, 3, 4, 5, 6]
        c.takeall(r"구속\w*|고정\w*|막아\w*|잡아\w*")
        if d:
            op.setdefault("fix", {}).update({str(k): 0.0 for k in d if str(k) not in op.get("fix", {})})
    if len(op) == 2:
        return [], [f"{target} 에 무엇을 할지 알려주세요 (예: x축 모멘트 1e6 N·mm, z 방향 -5 mm, x축 0.1 rad 회전, 회전 구속)"], [], [], {}
    c.takeall(r"을|를|에|으로|로|주고|줘|해줘|걸어\w*|가해\w*|적용\w*|하중|강제|만큼")
    return [op], [], [], c.leftover(), {}


BH_RULES = [
    (r"l\s*\*\s*/\s*l\s*_?\s*f\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"|l\s*\*\s*비\s*(?:율)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "Lstar_ratio"),
    (r"l\s*\*\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*mm", "Lstar"),
    (r"(?:바|bar)\s*(?:단면적|면적|area)\s*(?:를|을|은|는|=|:)?\s*" + NUM, "bar_area"),
    (r"(?:바|bar)\s*(?:항복|yield)\s*(?:응력)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "bar_yield"),
    (r"(?:바|bar)\s*(?:경화|hardening|h)\s*(?:계수)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "bar_H"),
    (r"(?:접힘|폴딩|fold(?:ing)?)\s*(?:힌지)?\s*(?:항복|yield)\s*(?:모멘트)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "fold_yield"),
    (r"(?:접힘|폴딩|fold(?:ing)?)\s*(?:힌지)?\s*(?:경화|hardening|h)\s*(?:계수)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "fold_H"),
    (r"(?:휨|밴딩|벤딩|bend(?:ing)?)\s*(?:힌지)?\s*(?:항복|yield)\s*(?:모멘트)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "bend_yield"),
    (r"(?:휨|밴딩|벤딩|bend(?:ing)?)\s*(?:힌지)?\s*(?:경화|hardening|h)\s*(?:계수)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "bend_H"),
    (r"(?:반복|사이클|cycle)\s*(?:하중)?\s*" + NUM + r"\s*(?:회|번|cycles?)?|" + NUM + r"\s*(?:회|번)\s*(?:반복|왕복)", "cycles"),
    (r"(?:되돌림|되돌아|반대\s*(?:방향)?(?:으로)?)\s*(?:변위)?\s*(?:를|을|은|는|=|:)?\s*" + NUM, "cycle_back"),
    (r"접촉\s*간격\s*(?:를|을|은|는|=|:)?\s*" + NUM, "gap"),
    (r"(?:판정\s*각|자동\s*판정\s*각)\s*(?:를|을|은|는|=|:)?\s*" + NUM, "tol"),
    (r"(?:접힘|폴딩|fold)\s*(?:강성|stiffness)\s*(?:을|를)?\s*(?:직접|값)?\s*(?:k\s*)?(?:=|:)?\s*" + NUM + r"\s*n\s*[·\.]?\s*mm\s*/\s*rad", "fold_k"),
]


def parse_bh(c, spec):
    op = {}
    m = c.take(r"(?:패널\s*)?(?:굽힘\s*강성|휨\s*강성|b)\s*(?:의|대비)?\s*" + NUM + r"\s*(?:%|퍼센트|프로)")
    if m:
        op["fold_mode"] = "percent"
        op["fold_percent"] = float(m.group(1))
    for pat, key in BH_RULES:
        m = c.take(pat)
        if not m:
            continue
        v = float(next(g for g in m.groups() if g is not None))
        if key == "cycles":
            v = int(round(v))
        if key == "cycle_back":
            v = abs(v)
        op[key] = v
        if key == "Lstar_ratio":
            op["Lstar"] = None
            op.setdefault("fold_mode", "auto")
        if key == "Lstar":
            op.setdefault("fold_mode", "auto")
        if key == "fold_k":
            op["fold_mode"] = "value"
    if c.take(r"(?:접힘|폴딩|fold)\s*강성\s*(?:을|를)?\s*자동|merlin\s*2?\s*자동|auto\s*mode"):
        op.update({"fold_mode": "auto", "Lstar": None})
    if c.take(r"(?:바|bar)\s*(?:단면적|면적)\s*(?:을|를)?\s*(?:자동|두께\s*(?:로|기반)?)"):
        op["bar_area"] = None
    if c.take(r"반복\s*(?:하중)?\s*(?:을|를)?\s*(?:끄|꺼|없애|해제|빼)|단조"):
        op.update({"cycles": 0, "cycle_back": 0.0})
    if c.take(r"(?:모드|고유)\s*(?:해석)?\s*(?:에서)?\s*질량\s*(?:없이|빼|제외)|k\s*만으로|kv\s*=\s*λv|질량\s*(?:없이|빼고|제외)"):
        op["use_mass"] = False
    elif c.take(r"질량\s*(?:포함|넣)"):
        op["use_mass"] = True
    if c.take(r"merlin\s*2?\s*(?:식|면적)|merlin2"):
        op["tri_bars"] = "merlin2"
    elif c.take(r"woodruff"):
        op["tri_bars"] = "woodruff"
    if c.take(r"부가\s*설정\s*(?:을|를)?\s*(?:모두|전부|다)?\s*(?:초기화|리셋|지워|삭제|없애)"):
        op.update({"fold_k": None, "Lstar": None, "bar_area": None, "tri_bars": "woodruff", "bar_yield": None, "bar_H": 0.0, "fold_yield": None, "fold_H": 0.0, "bend_yield": None, "bend_H": 0.0, "cycles": 0, "cycle_back": 0.0, "gap": None, "tol": 3.0, "long_factor": 100.0, "use_mass": True, "bend_k": None, "watch": [], "rest": {}})
        if (spec.get("bh") or {}).get("fold_mode") == "value":
            op["fold_mode"] = "auto"
    if not op:
        return None
    op["op"] = "set_bh"
    c.takeall(r"(?:으로|로)\s*(?:바꿔|해|설정|변경|둬|놔)\s*(?:줘|주세요|봐)?|설정\s*(?:해)?\s*(?:줘)?|해\s*줘|변경|바꿔|켜|넣어|줘")
    return op


def parse_clause(text, model, spec, ctx):
    c = Clause(text)
    ops = []
    ask = []
    notes = []
    flags = {}
    t = c.w
    if re.search(r"도움말|사용법|뭘\s*할\s*수|무엇을\s*할\s*수|어떻게\s*(?:써|사용)|예시\s*(?:좀|를)?\s*(?:보여|알려)|(?<![a-z])help(?![a-z])|what can you", t):
        return [], [], [HELP], [], {"answered": True}
    if re.search(r"stl\s*(?:메쉬|메시|mesh)\s*크기|메쉬\s*크기|메시\s*크기|mesh size|element size|요소\s*크기", t):
        return [], [], ["메쉬 크기는 왼쪽 'STL 메쉬 크기' 칸을 바꾼 뒤 파일을 다시 올리면 적용됩니다. 보·쉘 모델은 '요소 분할 4'처럼 분할수로 조절합니다"], [], {"answered": True}
    if re.search(r"그래프|플롯|(?<![a-z])plot|graph|곡선|curve|차트|chart|이력\s*(?:을|를)?\s*(?:그려|보여)", t) and not re.search(r"(?:삭제|지워|제거|remove|delete|clear)", t) and not re.search(r"\d\s*(?:mpa)?\s*@\s*\d|소성\s*곡선|경화\s*곡선|plastic\s*curve|hardening\s*curve", t):
        p = parse_plot(c, model, spec, ctx)
        if p:
            return [p], [], [], [], {}
        return [], ["어떤 값을 그릴지 알려주세요 (에너지, 반력, 변위, 응력, 접힘각, 하중-변위, 좌굴 계수, 고유진동수)"], [], [], {}
    if spec.get("solver") == "bh" and not re.search(r"그래프|플롯|plot|graph", t):
        bo = parse_bh(c, spec)
        if bo:
            return [bo], [], [], c.leftover(), {}
    if re.search(CP_WORD, t):
        r = parse_coupling(c, model, spec, ctx)
        if r is not None:
            return r
    if re.search(r"되돌려|되돌리|실행\s*취소|방금\s*(?:거|것|한\s*거)?\s*(?:취소|삭제|지워)|이전\s*(?:상태)?로|(?<![a-z])undo(?![a-z])|revert", t):
        return [], [], [], [], {"undo": True}
    if re.search(r"접촉|마찰|(?<![a-z])contact|friction|강체|압입|인덴터|indent|rigid|(?<![가-힣])구(?:를|로|가|\s)|sphere|(?<![a-z])ball(?![a-z])", t) and not re.search(r"구속|구간|구조|변위\s*제어|displacement\s*-?\s*control", t.replace("구를", "").replace("구로", "")) or re.search(r"(?<![a-z])ct\s*\d+(?![a-z0-9])", t):
        r = parse_contact(c, model, spec)
        if r is not None:
            return r
    ctl_load = None
    if re.search(r"변위\s*제어|displacement\s*-?\s*control", t):
        m = c.take(r"(?<![a-z])ld\s*(\d+)(?![a-z0-9])\s*(?:을|를|으로|로)?\s*(?:기준(?:으로)?)?")
        if m:
            ctl_load = "ld" + m.group(1)
            if not any(x["id"] == ctl_load for x in spec["loads"]):
                return [], [f"{ctl_load} 하중이 없습니다"], [], [], {}
    idm = c.take(r"(?<![a-z])(bc|ld|pl)\s*(\d+)(?![a-z0-9])")
    target = None
    if idm:
        target = idm.group(1) + idm.group(2)
        pool = spec["bcs"] + spec["loads"] + spec["plots"]
        if not any(x["id"] == target for x in pool):
            return [], [f"{target} 항목이 없습니다"], [], [], {}
    if not c.has(r"항복|소성|yield|plastic|힌지|조인트|접힘\s*선|결함|비선형|hinge|joint") and c.take(r"삭제|지워|제거|없애|빼\s*줘|remove|delete|drop"):
        if target:
            return [{"op": "remove", "id": target}], [], [], c.leftover(), {}
        m_last = c.take(r"마지막|방금|최근|last|latest")
        allw = c.take(r"모두|전부|전체|모든|다(?=\s)|싹|all|every")
        what = "bcs" if c.take(r"구속|경계\s*조건|지지|bcs?|constraints?|supports?") else "loads" if c.take(r"하중|힘|압력|loads?|forces?") else "plots" if c.take(r"그래프|플롯|plots?|graphs?") else None
        if m_last and what:
            if spec[what]:
                return [{"op": "remove", "id": spec[what][-1]["id"]}], [], [], c.leftover(), {}
            return [], ["삭제할 항목이 없습니다"], [], [], {}
        if what and (allw or True):
            return [{"op": "clear", "what": what}], [], [], c.leftover(), {}
        if allw or c.take(r"조건"):
            return [{"op": "clear", "what": "all"}], [], [], c.leftover(), {}
        return [], ["무엇을 삭제할지 알려주세요 (예: bc1 삭제, 하중 모두 삭제)"], [], [], {}
    if c.take(r"초기화|리셋|reset|clear"):
        what = "bcs" if c.take(r"구속|경계\s*조건|지지|bcs?|constraints?|supports?|boundary conditions?") else "loads" if c.take(r"하중|힘|loads?|forces?") else "plots" if c.take(r"그래프|플롯|plots?|graphs?") else "all"
        return [{"op": "clear", "what": what}], [], [], c.leftover(), {}

    if re.search(r"(?:해석|계산|시뮬레이션|simulation)\s*(?:을|를|도)?\s*(?:실행|돌려|돌리|시작|수행|해\s*줘|해\s*봐|해\s*라|하자|ㄱㄱ|고고|go)|(?:^|\s)(?:run|solve)(?![a-z])|^\s*(?:돌려|실행)(?:해)?\s*(?:줘|봐)?\s*$|풀어\s*(?:줘|봐)", t):
        flags["run"] = True

    for pat, name in ((r"calculix|ccx|칼큘릭스|캘큘릭스", "calculix"), (r"바\s*-?\s*힌지|bar\s*-?\s*(?:and\s*-?\s*)?hinge|barhinge", "barhinge"), (r"tensor\s*mesh|텐서\s*메쉬|텐서\s*메시", "tensormesh"), (r"자체\s*쉘|자체\s*셸|쉘\s*솔버|셸\s*솔버|own\s*shell|shell\s*solver", "shell")):
        if c.has(pat) and c.has(r"솔버|solver|모델로|로\s*(?:풀|해석|바꿔|변경|계산)|사용|쓰자|써\s*줘|써서|use|switch|으로\s*(?:풀|해석|바꿔|변경|계산)") and not (name == "barhinge" and c.has(r"강성|kappa|핀|조인트")):
            c.take(pat)
            c.takeall(r"솔버|solver|모델로|사용|쓰자|써\s*줘|써서|switch|풀어|풀자|계산")
            ops.append({"op": "set_solver", "solver": name})

    if flags.get("run"):
        c.takeall(r"실행|돌려|돌리|시작|수행|ㄱㄱ|고고|(?<![a-z])(?:run|solve|go)(?![a-z])|풀어|계산|시뮬레이션|simulation")

    st = {}
    if c.has(r"결함|불완전|imperfection") and c.has(r"\d+\s*차|모드\s*\d+\s*(?:,|~|-|에|번)|modes?\s*\d+\s*(?:,|~|-)"):
        pairs = []
        while True:
            m = c.take(r"(\d+)\s*차\s*(?:좌굴)?\s*(?:모드)?\s*(?:는|은|에|=|:)?\s*" + NUM + r"\s*(?:mm)?")
            if not m:
                break
            pairs.append([int(m.group(1)), _f(m, 2)])
        if not pairs:
            m = c.take(r"(?:모드|modes?)\s*(\d+(?:\s*(?:,|~|-)\s*\d+)*)\s*(?:번)?\s*(?:에|에는|에서)?\s*(?:각각|respectively)?\s*((?:" + NUM[1:-1] + r")(?:\s*,?\s*(?:" + NUM[1:-1] + r"))*)\s*(?:mm)?")
            if m:
                ns = []
                for part in re.split(r"\s*,\s*", m.group(1)):
                    if re.search(r"[~-]", part):
                        a, b = (int(x) for x in re.split(r"\s*[~-]\s*", part))
                        ns += list(range(a, b + 1))
                    else:
                        ns.append(int(part))
                vs = [float(x) for x in re.findall(NUM, m.group(2))]
                if len(vs) == 1:
                    vs = vs * len(ns)
                pairs = [[n, v] for n, v in zip(ns, vs)]
        if pairs:
            c.takeall(r"초기\s*결함|초기\s*불완전|결함|imperfection|좌굴|모드|각각|으로|로|을|를|줘|넣어|합|더해|조합|에")
            st["imperfection_modes"] = pairs
            st["imperfection"] = 0.0
    if c.take(r"(?:선형\s*|고유치\s*)?좌굴\s*(?:계수|하중|모드)?|buckl(?:e|ing)"):
        st["type"] = "buckle"
    elif c.take(r"모달|모드\s*해석|고유\s*진동수?|진동\s*해석|고유치|고유\s*모드|modal|eigen\s*frequenc(?:y|ies)|natural frequenc(?:y|ies)|frequency analysis|(?<![a-z])frequency(?![a-z])"):
        st["type"] = "modal"
    elif c.take(r"호장법?|호\s*길이|arc\s*-?\s*length|riks|스냅\s*스루|snap\s*-?\s*through"):
        st["type"] = "arclength"
    elif c.take(r"(?:선형\s*)?정적\s*(?:해석)?|(?<![a-z])static(?![a-z])"):
        st["type"] = "static"
    m = c.take(r"(기하\s*)?비\s*선형|대\s*변형|nlgeom|non\s*-?\s*linear|large\s*(?:deformation|displacement)")
    if m:
        st["nlgeom"] = not bool(c.take(r"끄|꺼|off|해제|없이|말고|제외|disable|without"))
        c.take(r"켜(?:고|서)?|on(?![a-z])|고려|포함|enable")
    elif c.take(r"선형\s*(?:해석)?\s*으로|선형\s*해석|(?<![a-z])linear(?![a-z])"):
        st["nlgeom"] = False
    m = c.take(r"모드\s*(?:수|개수)?\s*(?:를|을|는|=)?\s*(\d+)\s*(?:개)?|(\d+)\s*(?:개\s*)?(?:의\s*)?모드|(\d+)\s*modes?|modes?\s*=?\s*(\d+)")
    if m:
        st["modes"] = int(next(g for g in m.groups() if g))
    m = c.take(r"(?:증분|increments?|스텝\s*수|step\s*수|하중\s*단계|단계)\s*(?:수)?\s*(?:를|을|는|=)?\s*(\d+)\s*(?:개|단계|스텝)?(?:로|으로)?|(\d+)\s*(?:개\s*)?(?:증분|increments?|단계\s*로|스텝으로|steps?)")
    if m:
        st["increments"] = int(next(g for g in m.groups() if g))
    m = c.take(r"(?:초기\s*결함|초기\s*불완전|imperfection)\s*(?:을|를|은|는|=|크기)?\s*" + NUM + r"\s*(?:mm)?")
    if m:
        st["imperfection"] = _f(m)
    if c.take(r"변위\s*제어|변위\s*증분|displacement\s*-?\s*control(?:led)?"):
        if c.take(r"끄|꺼|off|해제|없이|말고|제외|disable|without|취소"):
            st["control"] = "load"
        else:
            st["control"] = "displacement"
            m = c.take(r"(?:목표\s*변위|최대\s*변위|목표|target)?\s*(?:를|을|는|=)?\s*" + NUM + r"\s*(mm|cm|m)(?![a-z])\s*(?:까지|만큼)?")
            if m:
                st["control_disp"] = abs(_f(m)) * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[m.group(2).lower()]
            if ctl_load:
                st["control_load"] = ctl_load
            if "nlgeom" not in st and st.get("type") not in ("buckle", "modal"):
                st["nlgeom"] = True
            c.takeall(r"켜(?:고|서)?|적용|사용|으로|로")
    elif c.take(r"하중\s*제어|load\s*-?\s*control(?:led)?"):
        st["control"] = "load"
    if c.take(r"강체\s*압판|압판|platen|rigid\s*(?:plate|platen)|똑같이\s*밀"):
        st["control_kind"] = "rigid"
        st.setdefault("control", "displacement")
    elif c.take(r"하중\s*분포\s*(?:를)?\s*유지|평균\s*변위|분포\s*유지"):
        st["control_kind"] = "average"
        st.setdefault("control", "displacement")
    if c.has(r"감쇠|댐핑|damping|레일리|rayleigh"):
        dm = {}
        if c.take(r"(?:감쇠|댐핑|damping)\s*(?:를|을|는|은)?\s*(?:없이|없애|끄|꺼|해제|off|0\s*(?:으로|로)?(?![.\d]))|no damping|without damping"):
            dm["kind"] = "none"
        else:
            if c.take(r"레일리|rayleigh"):
                dm["kind"] = "rayleigh"
            elif c.take(r"질량\s*(?:비례)?|mass(?:\s*proportional)?"):
                dm["kind"] = "mass"
            elif c.take(r"강성\s*(?:비례)?|stiffness(?:\s*proportional)?"):
                dm["kind"] = "stiffness"
            else:
                dm["kind"] = "rayleigh"
            m = c.take(NUM + r"\s*(?:%|퍼센트|퍼|percent)")
            if m:
                dm["zeta"] = _f(m)
            m = c.take(r"(?:스텝\s*)?시간\s*(?:은|는|을|를|=)?\s*" + NUM + r"\s*(?:s|초)(?![a-z])")
            if m:
                dm["time"] = _f(m)
        c.takeall(r"감쇠비?|댐핑|damping|비례|1\s*[,·]?\s*2\s*차|모드|기준|에\s*대해|추가|넣어|설정|적용|켜")
        st["damping"] = dm
    elif c.has(r"스텝\s*시간|step\s*time"):
        m = c.take(r"(?:스텝\s*시간|step\s*time)\s*(?:은|는|을|를|=)?\s*" + NUM + r"\s*(?:s|초|sec)(?![a-z])")
        if m:
            st["damping"] = {"time": _f(m)}
            c.takeall(r"으로|로|설정|바꿔|변경|해줘|해")
    if c.take(r"(?:사각형|쿼드|4\s*절점|quad)\s*(?:쉘|셸)?\s*(?:요소|element)\s*(?:으로|로)?"):
        st["element"] = "quad"
    elif c.take(r"(?:삼각형|3\s*절점|tri(?:angle)?)\s*(?:쉘|셸)?\s*(?:요소|element)\s*(?:으로|로)?"):
        st["element"] = "tri"
    if c.take(r"(?<![a-z])(?:gpu|cuda)(?![a-z])\s*(?:로|으로|에서|를|을)?\s*(?:계산|사용|써서|써|풀어)?|그래픽\s*카드\s*(?:로|으로)?|쿠다\s*(?:로|으로)?"):
        st["device"] = "cuda"
    elif c.take(r"(?<![a-z])cpu(?![a-z])\s*(?:로|으로|에서|만)?\s*(?:계산|사용|써서|써|풀어)?"):
        st["device"] = "cpu"
    elif c.take(r"계산\s*장치\s*(?:를|을)?\s*자동"):
        st["device"] = "auto"
    if c.take(r"단정밀도|(?<![a-z])(?:float32|fp32)(?![a-z0-9])"):
        st["precision"] = "float32"
    elif c.take(r"배정밀도|(?<![a-z])(?:float64|fp64)(?![a-z0-9])"):
        st["precision"] = "float64"
    m = c.take(r"(?:cpu\s*)?(?:스레드|쓰레드|threads?)\s*(?:를|을|는|=)?\s*(\d+)\s*(?:개)?|(\d+)\s*(?:개\s*)?(?:스레드|쓰레드|threads?)")
    if m:
        st["threads"] = int(m.group(1) or m.group(2))
    if c.take(r"부드럽게\s*(?:증가|올려|가해)?|S\s*자\s*(?:곡선)?|smooth\s*step|smooth(?:ly)?"):
        st["amplitude"] = "smooth"
    if c.take(r"분기\s*(?:점)?\s*(?:전환|추적|감지|회피)?|가지\s*전환|branch\s*switch(?:ing)?|bifurcation"):
        st["branch"] = "primary" if c.take(r"끄|꺼|off|해제|없이|말고|제외|disable|without|하지\s*마|주\s*경로") else "auto"
        c.takeall(r"켜(?:고|서)?|적용|사용")
    if st.get("type") in ("buckle", "modal") and "modes" not in st:
        m = c.take(r"(\d+)\s*(?:개|차)(?:까지)?")
        if m:
            st["modes"] = int(m.group(1))
    if st:
        ops.append({"op": "set_step", **st})
        c.takeall(r"해석|analysis|으로\s*(?:바꿔|변경|설정)|로\s*(?:바꿔|변경|설정)|모드|(?<![a-z])step(?![a-z])")
    elif flags.get("run"):
        c.takeall(r"해석|analysis")

    m = c.take(r"(?:요소\s*)?(?:분할|divisions?|세분화?)\s*(?:수)?\s*(?:를|을|는|=)?\s*(\d+)\s*(?:개|등분)?(?:로|으로)?|(\d+)\s*(?:분할|등분)(?:로|으로)?")
    if m:
        ops.append({"op": "set_divisions", "divisions": int(next(g for g in m.groups() if g))})
    elif c.has(r"메쉬|메시|mesh|요소|격자") and c.take(r"촘촘|조밀|세밀|잘게|곱게|더\s*작게|finer?|refine|dense"):
        ops.append({"op": "set_divisions", "divisions": min(64, max(2, int(spec["divisions"]) * 2))})
        c.takeall(r"메쉬|메시|mesh|격자|더|하게|히")
    elif c.has(r"메쉬|메시|mesh|요소|격자") and c.take(r"성기|거칠|굵게|크게|coarse[nr]?"):
        ops.append({"op": "set_divisions", "divisions": max(1, int(spec["divisions"]) // 2)})
        c.takeall(r"메쉬|메시|mesh|격자|더|하게|히")

    group_hit = None
    for g in sorted(model.group_names(), key=len, reverse=True):
        if g != "default" and re.search(r"(?<![a-z0-9_])" + re.escape(g.lower()) + r"(?![a-z0-9_])", c.w):
            group_hit = g
            break

    hinge_ctx = c.has(r"힌지|조인트|접힘\s*선|접는\s*선|fold\s*line|crease|(?<![a-z])hinges?(?![a-z])|(?<![a-z])joints?(?![a-z])") and not c.has(r"힌지\s*지지|hinged?\s*support|핀\s*지지")
    if hinge_ctx:
        hg = group_hit if group_hit in model.hinges else "*"
        if group_hit in model.hinges:
            c.take(re.escape(group_hit.lower()))
        kauto = c.take(r"(?:강성|kappa|κ)\s*(?:를|을|은|는)?\s*(?:자동|기본|auto|default)(?:으로|값으로)?")
        auto = c.takeall(r"공유\s*(?:하는|된|되는)?|일치\s*(?:하는|한|된)?|맞닿\s*(?:은|는|아\s*있는)?|만나\s*는|겹치\s*는|겹친|접하\s*는|붙어\s*있는|이어\s*진|연결\s*(?:된|부|부위)|이음\s*(?:부|매)?|사이\s*(?:의)?|쉘\s*간|패널\s*간|판\s*간|끼리|shared|coincident|common|between|adjacent|자동\s*(?:으로)?|automatic(?:ally)?|auto")
        mode = None
        ang = None
        if c.take(r"꺾인|꺾이는|꺾임|접힌|접혀\s*있는|creases?|folded"):
            mode = "crease"
        ma = c.take(r"(?:각도|angle)\s*(?:가|이|를|을)?\s*" + NUM + r"\s*(?:도|°|deg)?\s*(?:이상|보다\s*큰|초과|넘는)?|" + NUM + r"\s*(?:도|°|deg)\s*(?:이상|보다\s*큰|초과|넘는)")
        if ma:
            ang = float(ma.group(1) or ma.group(2))
            mode = "crease"
        if c.take(r"그룹\s*(?:경계|사이|간)?|파트\s*(?:경계|사이|간)?|부품\s*(?:경계|사이|간)?|객체\s*(?:경계|사이|간)?|오브젝트\s*(?:경계|사이|간)?|groups?|parts?|objects?"):
            mode = mode or "groups"
        if c.take(r"(?:모든|전체|전부)\s*(?:내부\s*)?(?:엣지|모서리|변|선|요소\s*경계)|all (?:interior |internal )?edges"):
            mode = "all"
        km0 = c.take(r"(?:강성|kappa|κ|stiffness)\s*(?:비율|비|ratio)\s*(?:를|을|은|는|=)?\s*" + NUM)
        km1 = None if km0 else c.take(r"(?:회전\s*)?(?:강성|kappa|κ|stiffness)\s*(?:를|을|은|는|=|to|:)?\s*" + NUM + r"\s*(?:n\s*[·*]?\s*mm\s*/\s*rad\s*/\s*mm|n\s*/\s*rad|n\s*[·*]?\s*mm\s*/\s*rad)?")
        pin = c.take(r"핀\s*-?\s*조인트|핀\s*-?\s*힌지|핀\s*(?:으로|처리|연결|결합|접합)|(?<![가-힣])핀(?![가-힣])|자유\s*회전|회전\s*자유|자유롭게\s*회전|강성\s*(?:이|을|를)?\s*(?:없|제로)|마찰\s*없|pin(?:ned)?\s*-?\s*joints?|(?<![a-z])pin(?:ned)?(?![a-z])|free(?:ly)?\s*rotat\w*|frictionless")
        off = c.take(r"해제|없애|끄(?:고|기)?|꺼|취소|강접|강체\s*연결|강결|일체\s*(?:로|화)|용접|연속\s*(?:으로|체)|rigid|(?<![a-z])off(?![a-z])|disable|remove")
        if off and not pin:
            ops.append({"op": "auto_hinge", "mode": "off"})
        elif auto or mode:
            op = {"op": "auto_hinge", "mode": mode or "auto"}
            if ang is not None:
                op["angle"] = ang
            ops.append(op)
        if km0:
            ops.append({"op": "set_hinge", "group": hg, "ratio": _f(km0), "kappa": None, "pin": False})
        elif km1 and _f(km1) > 0:
            ops.append({"op": "set_hinge", "group": hg, "kappa": _f(km1), "pin": False})
        elif km1 or pin:
            ops.append({"op": "set_hinge", "group": hg, "pin": True})
        elif kauto:
            ops.append({"op": "set_hinge", "group": hg, "kappa": None, "pin": False})
        if not any(o["op"] in ("auto_hinge", "set_hinge") for o in ops):
            ask.append("힌지를 어떻게 할지 알려주세요 (예: 쉘 사이 공유 엣지를 핀 조인트로, 힌지 강성 0.05, 힌지 해제)")
        c.takeall(r"힌지|조인트|접힘\s*선|접는\s*선|fold\s*lines?|creases?|hinges?|joints?|쉘|패널|판(?![가-힣])|shells?|panels?|엣지|모서리|edges?|처리|취급|간주|연결|treat|as(?![a-z])|회전")
        return ops, ask, notes, c.leftover(), flags

    mat = {}
    mname = None
    for pat, nm, E, nu, rho in MATERIALS:
        if c.take(pat):
            mat.update({"E": E, "nu": nu, "rho": rho})
            mname = nm
            break
    m = c.take(r"(?:탄성\s*계수|영률|영\s*계수|young'?s?\s*modulus|elastic modulus|(?<![a-z0-9.])e(?![a-z]))\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(gpa|mpa|kpa)?")
    if m:
        mat["E"] = _f(m) * {"gpa": 1e3, "mpa": 1.0, "kpa": 1e-3, None: 1.0}[m.group(2)]
    m = c.take(r"(?:포아송\s*비?|푸아송\s*비?|프와송\s*비?|poisson'?s?\s*(?:ratio)?|(?<![a-z])nu(?![a-z])|ν)\s*(?:를|을|은|는|=|:)?\s*" + NUM)
    if m:
        mat["nu"] = _f(m)
    m = c.take(r"(?:밀도|density|rho|ρ)\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(kg\s*/\s*m\s*(?:3|³|\^3)|g\s*/\s*(?:cm\s*(?:3|³|\^3)|cc)|(?:tonne|ton|t)\s*/\s*mm\s*(?:3|³|\^3))?")
    if m:
        u = (m.group(2) or "").replace(" ", "")
        v = _f(m)
        mat["rho"] = v * (1e-12 if u.startswith("kg") else 1e-9 if u.startswith("g") else 1.0 if u else (1e-12 if v > 10 else 1e-9 if v > 1e-3 else 1.0))
    m = c.take(r"(?:항복\s*(?:응력|강도|점)?|yield(?:\s*(?:stress|strength))?)\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(gpa|mpa)?")
    if m:
        mat["yield"] = _f(m) * (1e3 if m.group(2) == "gpa" else 1.0)
    elif c.take(r"(?:항복|소성|yield|plastic\w*)\s*(?:을|를|은|는)?\s*(?:없애|없음|없이|끄|꺼|해제|제거|off|none)|탄성\s*(?:만|으로만)"):
        mat["yield"] = None
    if c.has(r"곡선|curve") and c.has(r"소성|응력|경화|plastic|stress|hardening") or c.has(NUM + r"\s*(?:mpa)?\s*@\s*" + NUM):
        pts = []
        while True:
            mm = c.take(NUM + r"\s*(?:mpa)?\s*@\s*" + NUM + r"\s*(%)?")
            if not mm:
                break
            pts.append([float(mm.group(2)) * (0.01 if mm.group(3) else 1.0), float(mm.group(1))])
        if c.take(r"(?:삭제|지워|제거|없애|없이|해제|remove|delete|off)"):
            mat["curve"] = None
        elif len(pts) >= 2:
            mat["curve"] = pts
        c.takeall(r"소성|응력|경화|진응력|변형률|곡선|curve|plastic|stress|strain|hardening|입력|설정|으로|로|는|은|,")
    m = c.take(r"(?:멱\s*법칙|거듭제곱|power\s*-?\s*law)\s*(?:경화|hardening)?\s*(?:k\s*(?:=|:)?\s*)?" + NUM + r"\s*(?:mpa)?\s*,?\s*(?:m|n|지수)\s*(?:=|:)?\s*" + NUM)
    if m:
        mat["power"] = [float(m.group(1)), float(m.group(2))]
        c.takeall(r"경화|hardening|으로|로")
    m = c.take(r"(?:경화\s*(?:계수|기울기)?|hardening)\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(gpa|mpa)?")
    if m:
        mat["hardening"] = _f(m) * (1e3 if m.group(2) == "gpa" else 1.0)
    if mat:
        ops.append({"op": "set_material", "name": "default", **mat})
        if mname:
            notes.append(f"{mname} 대표값 E={mat['E']:g} MPa, ν={mat['nu']:g} 적용")
        c.takeall(r"재료|재질|물성|material|로\s*(?:바꿔|변경|설정)|으로\s*(?:바꿔|변경|설정)")

    sg = group_hit if group_hit and group_hit not in model.hinges and group_hit not in model.patches else "*"
    sec = {}
    m = c.take(r"(?:두께|thickness|(?<![a-z])t(?![a-z]))\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(mm|cm|m(?![a-z]))?")
    if m:
        sec["thickness"] = _f(m) * {"mm": 1.0, "cm": 10.0, "m": 1000.0, None: 1.0}[m.group(2)]
    beam = {}
    if c.has(r"단면|보(?![가-힣])|빔|beam|section|각형|사각|rect|원형|circ|파이|ø|φ|지름|직경|반지름|반경|폭|너비|높이|춤"):
        m = c.take(NUM + r"\s*(?:mm)?\s*[x*]\s*" + NUM + r"\s*(?:mm)?")
        if m:
            beam.update({"shape": "rect", "b": _f(m, 1), "h": _f(m, 2)})
        m = c.take(r"(?:폭|너비|가로|width|(?<![a-z])b(?![a-z]))\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(?:mm)?")
        if m:
            beam.update({"shape": "rect", "b": _f(m)})
        m = c.take(r"(?:높이|춤|세로|height|depth|(?<![a-z])h(?![a-z]))\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(?:mm)?")
        if m:
            beam.update({"shape": "rect", "h": _f(m)})
        m = c.take(r"(?:반지름|반경|radius|(?<![a-z])r(?![a-z]))\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(?:mm)?")
        if m:
            beam = {"shape": "circ", "r": _f(m)}
        m = c.take(r"(?:지름|직경|diameter|파이|ø|φ|(?<![a-z])d(?![a-z]))\s*(?:를|을|은|는|=|:)?\s*" + NUM + r"\s*(?:mm)?")
        if m:
            beam = {"shape": "circ", "r": _f(m) / 2}
    if beam:
        sec["beam"] = beam
    if sec:
        if sg != "*":
            c.take(re.escape(sg.lower()))
        ops.append({"op": "set_section", "group": sg, **sec})
        c.takeall(r"단면|보(?![가-힣])|빔|beam|section|각형|사각형?|직사각형?|rect(?:angular)?|원형|circ(?:ular)?|쉘|shell|판(?![가-힣])|그룹|group")

    gm = c.take(r"중력|자중|gravity|self\s*-?\s*weight|dead\s*load")
    if gm:
        d = direction(c, model)
        gf = c.take(NUM + r"\s*g(?![a-z])")
        mag = 9810.0 * (abs(_f(gf)) if gf else 1.0)
        v = d["vec"] if d else [0.0, 0.0, -1.0]
        ops.append({"op": "add_load", "type": "gravity", "name": "자중", "vector": [x * mag + 0.0 for x in v]})
        rho = spec["materials"]["default"].get("rho") or 0
        if not rho and not any(o["op"] == "set_material" and "rho" in o for o in ops):
            notes.append("자중에는 밀도가 필요합니다")
        c.takeall(r"가속도|acceleration|고려|포함|작용")
        return ops, ask, notes, c.leftover(), flags

    vec_m = c.take(r"\(\s*" + NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\)\s*(kn(?![a-z])|n(?![a-z/])|kgf)")
    comp = {}
    for m in c.takeall(r"(?<![a-z])([fm])\s*([xyz])\s*(?:=|:|는|은)?\s*" + NUM + r"\s*(kn\s*[·*]?\s*m|n\s*[·*]?\s*mm|n\s*[·*]?\s*m|kn|kgf|n)?"):
        u = (m.group(4) or "").replace(" ", "").replace("·", "").replace("*", "")
        sc = {"": 1.0, "n": 1.0, "kn": 1e3, "kgf": G0, "nmm": 1.0, "nm": 1e3, "knm": 1e6}[u]
        comp.setdefault(m.group(1), [0.0, 0.0, 0.0])[AX.index(m.group(2))] += _f(m, 3) * sc

    if c.has(r"반경\s*방향|반지름\s*방향|radial|원주\s*방향|접선\s*방향|tangential|circumferential|원통\s*좌표"):
        return ops, ["반경·원주 방향 지정은 아직 지원하지 않습니다. x, y, z 방향으로 나눠서 알려주세요"], notes, [], flags
    is_bc_word = bool(c.has(r"(?<![가-힣])핀(?![가-힣])|(?<![a-z])pin(?![a-z])|고정|구속|지지|받쳐|받침|막아|막고|잡아|잡고|묶|못\s*움직|움직이지\s*(?:않|못)|(?<![a-z])fix|clamp|constrain|restrain|support|pinned|roller|롤러|이동단|가동단|회전단|대칭|symm|encast|용접|단순보|외팔보|캔틸레버|cantilever|simply"))
    scale = scale_word(c)
    qs = quantities(c)
    kinds = [q["kind"] for q in qs]

    def first(kind):
        return next((q for q in qs if q["kind"] == kind), None)

    flip = c.take(r"(?:방향\s*(?:을|를)?\s*)?(?:반대\s*(?:로|방향으로)|뒤집어|반전|reverse|flip|invert)") if (target or c.has(r"하중|힘|load|force")) and not c.has(r"반대\s*(?:쪽|편|측)") else None
    change = c.take(r"바꿔|변경|수정|키워|줄여|늘려|낮춰|높여|change|update|modify|scale|increase|decrease|reduce") or scale is not None or flip or (target is not None)
    if change and not is_bc_word:
        items = []
        if target:
            items = [x for x in spec["bcs"] + spec["loads"] if x["id"] == target]
        elif c.take(r"하중|힘|loads?|forces?"):
            items = [x for x in spec["loads"] if x["type"] != "gravity"]
            if not items:
                return ops, ["바꿀 하중이 없습니다"], notes, [], flags
            if len(items) > 1 and scale is None and not flip:
                return ops, ["하중이 여러 개입니다. 어느 것인지 알려주세요 (예: ld1 을 2000 N으로)"], notes, [], flags
        elif c.take(r"압력|pressure"):
            items = [x for x in spec["loads"] if x["type"] == "pressure"]
        elif c.take(r"강제\s*변위|지정\s*변위|변위|displacement"):
            items = [x for x in spec["bcs"] if abs(x.get("value", 0.0)) > 0]
        if items:
            d = direction(c, model)
            for it in items:
                up = {"op": "update", "id": it["id"]}
                if "dofs" in it:
                    q = first("length") or first("angle")
                    v = q["value"] if q else None
                    if scale is not None:
                        up["value"] = it.get("value", 0.0) * scale
                    elif flip:
                        up["value"] = -it.get("value", 0.0)
                    elif v is not None or (v := bare(c)) is not None:
                        up["value"] = v
                elif it["type"] == "pressure":
                    q = first("pressure")
                    if scale is not None:
                        up["value"] = it["value"] * scale
                    elif q or (b := bare(c)) is not None:
                        up["value"] = abs(q["value"] if q else b)
                else:
                    q = first("force") or first("moment")
                    vec = np.array(it["vector"], dtype=float)
                    if scale is not None:
                        up["vector"] = (vec * scale + 0.0).tolist()
                    elif flip:
                        up["vector"] = (-vec + 0.0).tolist()
                    else:
                        v = q["value"] if q else bare(c)
                        if v is not None:
                            if d is not None:
                                up["vector"] = [x * (v if d["literal"] else abs(v)) + 0.0 for x in d["vec"]]
                            else:
                                nrm = float(np.linalg.norm(vec))
                                up["vector"] = (vec / nrm * abs(v) + 0.0).tolist() if nrm > 0 else it["vector"]
                if len(up) > 2:
                    ops.append(up)
            if any(o["op"] == "update" for o in ops):
                c.takeall(r"으로|로|를|을|값|크기|magnitude|to(?![a-z])")
                return ops, ask, notes, c.leftover(), flags
            return ops, ["어떻게 바꿀지 알려주세요 (예: ld1 을 2000 N으로, 하중 2배)"], notes, [], flags

    sel, sn = selection(c, model, spec, ctx)
    ask += sn
    notes += ctx.pop("info", [])
    inherited = False
    L = longest_axis(model)

    if c.has(r"외팔보|캔틸레버|cantilever") and not qs:
        c.takeall(r"외팔보|캔틸레버|cantilever|조건|형태|로\s*만들어|으로\s*만들어")
        ops.append({"op": "add_bc", "name": "고정단", "type": "fixed", "select": sel or _plane(AX[L], "min")})
        ctx["sel"] = sel or _plane(AX[L], "min")
        return ops, ask, notes, c.leftover(), flags
    if c.has(r"단순\s*보|단순\s*지지\s*보|simply supported beam") and not qs and sel is None:
        c.takeall(r"단순\s*지지\s*보|단순\s*보|simply supported beam|조건|형태|로\s*만들어|으로\s*만들어")
        ops.append({"op": "add_bc", "name": "핀 지지", "type": "pinned", "select": _plane(AX[L], "min")})
        free = [d + 1 for d in range(3) if d != L]
        ops.append({"op": "add_bc", "name": "롤러 지지", "dofs": free, "select": _plane(AX[L], "max")})
        notes.append(f"{AX[L]} 최소 끝 핀, {AX[L]} 최대 끝 롤러({AX[L]} 방향 이동 자유)로 넣었습니다")
        return ops, ask, notes, c.leftover(), flags

    load_word = c.has(r"하중|힘(?![가-힣])|압력|내압|외압|면압|토크|모멘트|비틀|load|force|pressure|torque|moment|당기는|누르는|압축|인장")
    disp_word = c.has(r"변위|이동|밀어|밀고|당겨|당기|눌러|누르|늘려|늘리|줄여|벌려|접어|회전|돌려|압축|인장|displace|move|push|pull|rotate|compress|stretch|shift")
    has_load_q = any(k in kinds for k in ("force", "moment", "line")) or vec_m or comp
    has_pres_q = "pressure" in kinds and not mat

    if has_load_q or has_pres_q or (load_word and not is_bc_word and not ("length" in kinds or "angle" in kinds) and not ops):
        if sel is None and ctx.get("sel") is not None and not c.has(r"압력|내압|외압|pressure"):
            sel = ctx["sel"]
            inherited = True
        per_node = c.take(r"절점\s*당|절점\s*마다|노드\s*당|노드\s*마다|각\s*절점(?:에|마다)?|각\s*노드(?:에|마다)?|per node|each node|at each node")
        d = direction(c, model)
        side = c.take(r"옆\s*으로|수평\s*(?:으로|방향)|가로\s*(?:로|방향)|횡\s*방향|sideways|horizontal(?:ly)?|lateral(?:ly)?")
        if d is None and not side and c.take(r"올려|얹어|얹고|실어|매달|무게|weight|hang"):
            d = {"vec": [0.0, 0.0, -1.0], "literal": False}
        if side and d is None:
            ask.append("'옆으로'가 x 방향인지 y 방향인지 알려주세요 (예: x 방향으로 500 N)")
            return ops, ask, notes, [], flags
        if d is None and c.take(r"수직\s*(?=하중|으로|방향|힘)|연직\s*(?=하중|으로|방향|힘)|vertical(?:ly)?"):
            d = {"vec": [0.0, 0.0, -1.0], "literal": False}
            notes.append("수직 하중을 아래 방향(-z)으로 보았습니다")
        if d is None and not c.has(r"압축|인장|compress|tens|누르|눌러|당기|당겨|밀어"):
            d = loose_axis(c)
        comp_word = c.take(r"압축|compress\w*|누르는|눌러|밀어")
        tens_word = None if comp_word else c.take(r"인장|tens\w*|당기는|당겨|pull\w*")
        made = False

        def dir_for(val, need_abs=False):
            if d is not None:
                if d["literal"] and not need_abs:
                    return [x * val + 0.0 for x in d["vec"]]
                return [x * abs(val) + 0.0 for x in d["vec"]]
            if comp_word or tens_word:
                n = inward(model, sel)
                if n is not None:
                    s = 1.0 if comp_word else -1.0
                    return [x * s * abs(val) + 0.0 for x in n]
            return None

        if vec_m:
            sc = {"kn": 1e3, "n": 1.0, "kgf": G0}[vec_m.group(4)]
            if sel is None:
                ask.append("하중을 가할 위치를 알려주세요")
            else:
                ops.append({"op": "add_load", "type": "force", "name": "하중", "select": sel, "vector": [_f(vec_m, i) * sc for i in (1, 2, 3)]})
            made = True
        for key, typ in (("f", "force"), ("m", "moment")):
            if key in comp:
                if sel is None:
                    ask.append("하중을 가할 위치를 알려주세요")
                else:
                    ops.append({"op": "add_load", "type": typ, "name": "하중" if typ == "force" else "모멘트", "select": sel, "vector": comp[key]})
                made = True
        q = first("moment")
        if q:
            made = True
            ax = c.take(r"(?<![a-z])([xyz])\s*축\s*(?:기준|중심|둘레|에\s*대한|주위)?|about\s*([xyz])(?![a-z])|around\s*([xyz])(?![a-z])")
            if ax:
                v = [0.0, 0.0, 0.0]
                v[AX.index(next(g for g in ax.groups() if g))] = q["value"]
            elif d is not None:
                v = [x * q["value"] + 0.0 for x in d["vec"]]
            else:
                v = None
            if v is None:
                ask.append("모멘트의 축을 알려주세요 (예: y축 기준 500 N·m)")
            elif sel is None:
                ask.append("모멘트를 가할 위치를 알려주세요")
            else:
                ops.append({"op": "add_load", "type": "moment", "name": "모멘트", "select": sel, "vector": v})
            c.takeall(r"모멘트|토크|비틀림|비틀|moment|torque|twist")
        q = first("line")
        if q:
            made = True
            if sel is None:
                ask.append("선하중을 가할 변을 알려주세요 (예: x 최대 변에 10 N/mm 아래로)")
            else:
                ids = resolve(model, sel)
                length = edge_length(model, ids)
                v = dir_for(q["value"] * length)
                if v is None:
                    ask.append("선하중의 방향을 알려주세요 (예: z 방향으로 -10 N/mm)")
                elif length <= 0:
                    ask.append("선하중을 가할 변의 길이를 구하지 못했습니다")
                else:
                    ops.append({"op": "add_load", "type": "force", "name": f"선하중 {abs(q['value']):g} N/mm", "select": sel, "vector": v})
                    notes.append(f"선하중 {abs(q['value']):g} N/mm × 길이 {length:g} mm = 합 {abs(q['value']) * length:g} N")
            c.takeall(r"선\s*하중|분포\s*하중|등분포|line load|distributed")
        q = first("force")
        fv = q["value"] if q else None
        assumed = False
        if fv is None and not made and not has_pres_q and c.has(r"하중|힘(?![가-힣])|load|force") and not c.has(r"압력|pressure"):
            fv = bare(c, r"하중|힘|load|force")
            assumed = fv is not None
        if fv is not None:
            made = True
            if sel is None:
                ask.append("하중을 가할 위치를 알려주세요 (예: x 최대 끝단)")
            else:
                n = len(resolve(model, sel)) if per_node else 1
                v = dir_for(fv * n)
                if v is None:
                    ask.append("하중 방향을 알려주세요 (예: z 방향으로 -1000 N, 아래로 2 kN)")
                else:
                    ops.append({"op": "add_load", "type": "force", "name": "압축 하중" if comp_word else "인장 하중" if tens_word else "하중", "select": sel, "vector": v})
                    if per_node:
                        notes.append(f"절점 {n}개 × {abs(fv):g} N = 합 {abs(fv) * n:g} N")
                    if assumed:
                        notes.append("단위가 없어 N으로 보았습니다")
        q = first("pressure") if has_pres_q else None
        pv = q["value"] if q else None
        if pv is None and not made and c.has(r"압력|내압|외압|면압|pressure"):
            pv = bare(c, r"압력|내압|외압|면압|pressure")
            if pv is not None:
                notes.append("단위가 없어 MPa로 보았습니다")
        if pv is not None:
            made = True
            if sel is None and model.kind == "stl":
                ask.append("압력을 가할 면을 알려주세요 (예: face_2 에 압력 0.5 MPa)")
            else:
                op = {"op": "add_load", "type": "pressure", "name": "압력", "select": sel or {"all": True}, "value": abs(pv)}
                if d is not None:
                    op["direction"] = [x * (1 if pv >= 0 or not d["literal"] else -1) + 0.0 for x in d["vec"]]
                elif pv < 0:
                    notes.append("음수 압력은 방향을 함께 알려주세요. 크기만 사용했습니다")
                ops.append(op)
        if not made:
            ask.append("하중의 크기를 알려주세요 (예: 1000 N, 0.5 MPa)")
        if inherited and any(o["op"] == "add_load" for o in ops):
            notes.append("위치는 앞 문장과 같은 곳으로 보았습니다")
        if sel is not None and not inherited:
            ctx["sel"] = sel
        c.takeall(r"하중|힘|압력|내압|외압|면압|등분포|분포|집중|수직|연직|축\s*방향|축|작용|load|force|pressure|uniform(?:ly)?|distributed|concentrated|point|axial|vertical|으로|로")
        return ops, ask, notes, c.leftover(), flags

    if ("length" in kinds or "angle" in kinds) and (disp_word or not is_bc_word) and not sec and not mat:
        if sel is None and ctx.get("sel") is not None:
            sel = ctx["sel"]
            inherited = True
        ql = first("length")
        qa = first("angle")
        rot = bool(qa) and bool(c.has(r"회전|돌려|rotat|twist|비틀"))
        if rot:
            ax = c.take(r"(?<![a-z])([xyz])\s*축\s*(?:을|를)?\s*(?:기준|중심|둘레|주위|에\s*대해|으로|로)?|about\s*([xyz])(?![a-z])|around\s*([xyz])(?![a-z])")
            d = None if ax else direction(c, model)
            a = next(g for g in ax.groups() if g) if ax else AX[max(range(3), key=lambda i: abs(d["vec"][i]))] if d else None
            if a is None:
                ask.append("회전 축을 알려주세요 (예: x축 기준 10도 회전)")
            elif sel is None:
                ask.append("회전시킬 위치를 알려주세요")
            else:
                ops.append({"op": "add_bc", "name": "지정 회전", "select": sel, "dofs": [AX.index(a) + 4], "value": round(qa["value"], 9)})
                if model.kind == "stl":
                    notes.append("솔리드 요소에는 회전 자유도가 없어 지정 회전이 적용되지 않습니다")
        elif ql:
            d = direction(c, model)
            if d is None and not c.has(r"압축|인장|compress|tens|누르|눌러|당기|당겨|밀어|늘려|줄여|벌려"):
                d = loose_axis(c)
            comp_word = c.take(r"압축|compress\w*|눌러|누르|밀어|밀고|push\w*|줄여")
            tens_word = None if comp_word else c.take(r"인장|당겨|당기|늘려|늘리|벌려|stretch\w*|pull\w*|tens\w*")
            v = None
            if d is not None:
                a = max(range(3), key=lambda i: abs(d["vec"][i]))
                v = (ql["value"] if d["literal"] else abs(ql["value"])) * d["vec"][a]
            elif comp_word or tens_word:
                n = inward(model, sel)
                if n is not None:
                    a = max(range(3), key=lambda i: abs(n[i]))
                    v = abs(ql["value"]) * n[a] * (1.0 if comp_word else -1.0)
            if sel is None:
                ask.append("변위를 줄 위치를 알려주세요 (예: x 최대 면을 x 방향으로 -5 mm)")
            elif v is None:
                ask.append("변위 방향을 알려주세요 (예: x 방향으로 -5 mm, 아래로 3 mm)")
            else:
                ops.append({"op": "add_bc", "name": "지정 변위", "select": sel, "dofs": [a + 1], "value": round(v + 0.0, 9)})
        else:
            ask.append("변위 크기를 mm로 알려주세요")
        if sel is not None and not inherited:
            ctx["sel"] = sel
        if inherited and ops:
            notes.append("위치는 앞 문장과 같은 곳으로 보았습니다")
        c.takeall(r"강제\s*변위|지정\s*변위|강제|변위|이동|회전|비틀\w*|twist\w*|접어|접고|fold\w*|내려|올려|낮춰|들어|움직여|움직이|시켜|displace\w*|move\w*|rotat\w*|prescribed|enforced|만큼|으로|로")
        return ops, ask, notes, c.leftover(), flags

    if is_bc_word:
        if sel is None and ctx.get("sel") is not None and c.has(r"거기|같은\s*(?:곳|위치)|그\s*(?:곳|위치|면)|same"):
            sel = ctx["sel"]
            c.takeall(r"거기|같은\s*(?:곳|위치)|그\s*(?:곳|위치|면)|same")
        op = {"op": "add_bc"}
        anti = c.take(r"반\s*대칭|역\s*대칭|anti\s*-?\s*symm\w*|asymm\w*") is not None
        sym = c.take(r"(?<![a-z])([xyz])\s*(?:축|방향)?\s*대칭|(?<![a-z])([xyz])\s*-?\s*symm\w*|(xy|yx|yz|zy|zx|xz)\s*(?:평면|면|plane)?\s*(?:에\s*대해|에\s*대한)?\s*대칭|(xy|yx|yz|zy|zx|xz)\s*-?\s*(?:plane)?\s*symm\w*")
        if sym:
            g = sym.groups()
            a = g[0] or g[1] or next(ch for ch in AX if ch not in (g[2] or g[3]))
            op.update({"type": a + ("asymm" if anti else "symm"), "name": f"{a} {'반대칭' if anti else '대칭'}"})
            if sel is None:
                sel = _plane(a, "min")
                notes.append(f"대칭면 위치를 {a} 최소 면으로 보았습니다")
        elif anti or c.take(r"대칭|symm\w*"):
            ps = [p for p in sel_planes(sel)] if sel else []
            if len(ps) == 1:
                op.update({"type": ps[0]["axis"] + ("asymm" if anti else "symm"), "name": f"{ps[0]['axis']} {'반대칭' if anti else '대칭'}"})
            else:
                ask.append("대칭 방향을 알려주세요 (예: x 최소 면에 x 대칭)")
        else:
            invert = c.has(r"만\s*(?:자유|풀|허용|움직)|(?:은|는|만)?\s*(?:자유|풀어|풀고|허용|free|released?)|제외|빼고|except|나머지")
            dofs = bc_dofs(c)
            full = c.take(r"완전\s*(?:히)?\s*고정|전부\s*고정|모두\s*고정|완전\s*구속|전\s*자유도|모든\s*자유도|모든\s*방향|6\s*자유도|fully\s*(?:fixed|clamped)|all dofs|encastre?|clamp\w*|용접|built\s*-?\s*in")
            roller = c.take(r"롤러|roller|이동단|가동단")
            pinned = c.take(r"핀\s*지지|핀\s*고정|핀(?![가-힣])|힌지\s*지지|힌지|회전단|pinned|pin(?![a-z])|hinged?|단순\s*지지|simply\s*supported|simple support")
            if full:
                op.update({"type": "fixed", "name": "고정"})
            elif roller:
                if dofs:
                    rest = [k for k in (1, 2, 3) if k not in dofs]
                    op.update({"dofs": rest, "name": "롤러 지지"})
                    notes.append("롤러: " + ", ".join(AX[k - 1] for k in dofs if k <= 3) + " 방향 이동 자유, 나머지 병진 구속으로 보았습니다")
                else:
                    op.update({"dofs": [3], "name": "롤러 지지"})
                    notes.append("롤러를 z 방향만 구속으로 보았습니다. 다르면 방향을 알려주세요")
            elif dofs and invert and c.take(r"만\s*(?:자유|풀|허용|움직)\w*|(?:은|는|만)?\s*(?:자유\w*|풀어|풀고|허용|free|released?)|제외|빼고|except|나머지"):
                tr = [k for k in dofs if k <= 3]
                rest = [k for k in (1, 2, 3) if k not in tr]
                if any(k > 3 for k in dofs):
                    rest = [k for k in (1, 2, 3, 4, 5, 6) if k not in dofs]
                op.update({"dofs": rest, "name": "구속 " + "".join(AX[k - 1] if k <= 3 else "r" + AX[k - 4] for k in rest)})
            elif dofs:
                op.update({"dofs": dofs, "name": "구속 " + "".join(AX[k - 1] if k <= 3 else "r" + AX[k - 4] for k in dofs)})
            elif pinned or c.has(r"회전\s*(?:은|는|만)?\s*(?:자유|허용)"):
                op.update({"type": "pinned", "name": "핀 지지"})
                c.takeall(r"회전\s*(?:은|는|만)?\s*(?:자유|허용)\w*")
            elif c.has(r"고정|(?<![a-z])fix|구속|constrain|restrain|막아|막고|잡아|잡고|묶|못\s*움직|움직이지"):
                op.update({"type": "fixed", "name": "고정"})
            else:
                op.update({"type": "pinned", "name": "지지"})
                notes.append("'지지'를 병진 3방향 구속(핀)으로 보았습니다")
        if sel is None:
            if not ask:
                ask.append("구속할 위치를 알려주세요 (예: z 최소 면, 왼쪽 끝, 테두리)")
        elif "type" in op or "dofs" in op:
            op["select"] = sel
            if op.get("dofs") == []:
                ask.append("구속할 자유도가 남지 않습니다")
            else:
                ops.append(op)
            ctx["sel"] = sel
        c.takeall(r"완전\s*히?|고정|구속|지지|받쳐|받침|막아|막고|잡아|잡고|묶어|묶고|못\s*움직이게|움직이지\s*(?:않게|못하게)|단순\s*보|단순|조건|경계|자유도|fix\w*|constrain\w*|restrain\w*|support\w*|simply|boundary|condition|bc(?![a-z0-9])|으로|로|만")
        return ops, ask, notes, c.leftover(), flags

    if sel is not None and not ops and not flags:
        ask.append("그 위치에 무엇을 할지 알려주세요 (고정, 하중, 강제변위)")
        ctx["sel"] = sel
        return ops, ask, notes, [], flags
    return ops, ask, notes, c.leftover(), flags


def describe_op(o):
    from .select import describe

    k = o["op"]
    if k == "add_bc":
        d = o.get("dofs")
        what = {"fixed": "완전 고정", "pinned": "병진 구속(핀)", "xsymm": "x 대칭 (ux, ry, rz)", "ysymm": "y 대칭 (uy, rx, rz)", "zsymm": "z 대칭 (uz, rx, ry)", "xasymm": "x 반대칭 (uy, uz, rx)", "yasymm": "y 반대칭 (ux, uz, ry)", "zasymm": "z 반대칭 (ux, uy, rz)"}.get(o.get("type"), o.get("type") or "") if not d else "자유도 " + ",".join("u" + AX[i - 1] if i <= 3 else "r" + AX[i - 4] for i in d)
        v = o.get("value")
        return f"구속: {describe(o.get('select'))} → {what}" + (f" = {v:g}" if v else "")
    if k == "add_load":
        if o["type"] == "gravity":
            return f"자중: 가속도 {fmt_vec(o['vector'])} mm/s²"
        if o["type"] == "pressure":
            return f"압력: {describe(o.get('select'))} 에 {o['value']:g} MPa" + (f" 방향 {fmt_vec(o['direction'])}" if o.get("direction") else "")
        u = "N·mm" if o["type"] == "moment" else "N"
        return f"{'모멘트' if o['type'] == 'moment' else '하중'}: {describe(o.get('select'))} 에 합 {fmt_vec(o['vector'])} {u}"
    if k == "remove":
        return f"{o['id']} 삭제"
    if k == "clear":
        return {"bcs": "구속 전체 삭제", "loads": "하중 전체 삭제", "plots": "그래프 전체 삭제", "contacts": "접촉 전체 삭제", "couplings": "기준점 전체 삭제", "all": "조건 전체 삭제"}[o.get("what", "all")]
    if k == "add_contact":
        mu = f", 마찰 {o.get('friction', 0):g}"
        if o.get("type") == "self":
            return f"자기 접촉 추가{mu}"
        if o.get("type") == "pair":
            return f"접촉 쌍 추가: {o['a'].get('group', '영역 a')} ↔ {o['b'].get('group', '영역 b')}{mu}"
        mv = ", 이동 [" + ", ".join(f"{v:g}" for v in o.get("motion", [0, 0, 0])) + "] mm"
        if o.get("shape") == "sphere":
            return f"강체 구 접촉 추가: 반지름 {o['radius']:g} mm{mv}{mu}"
        if o.get("shape") == "mesh":
            return f"강체 부품 접촉 추가: {describe(o.get('body'))} 를 변형하지 않는 접촉 형상(강체)으로{mv}{mu}"
        return f"강체 평면 접촉 추가: 법선 [" + ", ".join(f"{v:g}" for v in o.get("normal", [0, 0, 1])) + f"]{mv}{mu}"
    if k == "add_coupling":
        if o.get("rigid"):
            return f"강체 지정: {describe(o.get('select'))} 의 모든 절점을 기준점 하나에 묶어 강체로 움직임 (기준점에 하중·변위·구속)"
        return f"기준점 커플링: {describe(o.get('select'))} 의 절점을 기준점 하나에 강체로 묶음"
    if k == "update" and str(o.get("id", "")).startswith("cp"):
        parts = []
        nm = ["ux", "uy", "uz", "rx", "ry", "rz"]
        if "free" in o:
            parts.append("모든 자유도 해제")
        for d, v in (o.get("fix") or {}).items():
            j = int(d)
            parts.append(f"{nm[j - 1]} = {v:g} {'rad' if j > 3 else 'mm'}" if v else f"{nm[j - 1]} 구속")
        if o.get("force"):
            parts.append(f"힘 {fmt_vec(o['force'])} N")
        if o.get("moment"):
            parts.append(f"모멘트 {fmt_vec(o['moment'])} N·mm")
        return f"기준점 {o['id']}: " + ", ".join(parts)
    if k == "update":
        return f"{o['id']} 수정: " + ", ".join(f"{a}={fmt_vec(b) if isinstance(b, list) else b}" for a, b in o.items() if a not in ("op", "id"))
    if k == "set_material":
        return "재료: " + ", ".join(f"{a}={b:g}" if isinstance(b, (int, float)) else f"멱법칙 경화 k={b[0]:g}, m={b[1]:g}" if a == "power" and isinstance(b, list) else f"소성 곡선 {len(b)}점" if isinstance(b, list) else f"{a}=없음" for a, b in o.items() if a not in ("op", "name"))
    if k == "set_section":
        s = []
        if "thickness" in o:
            s.append(f"두께 {o['thickness']:g} mm")
        if "beam" in o:
            b = o["beam"]
            s.append(f"원형 r={b['r']:g}" if b.get("shape") == "circ" else "사각 " + " ".join(f"{x}={b[x]:g}" for x in ("b", "h") if x in b))
        return f"단면({o.get('group', '*')}): " + ", ".join(s)
    if k == "set_hinge":
        if o.get("pin"):
            return f"힌지({o.get('group', '*')}): 핀 조인트 (회전 자유)"
        if "ratio" in o:
            return f"힌지({o.get('group', '*')}): 강성 비율 {o['ratio']:g}"
        return f"힌지({o.get('group', '*')}): " + ("강성 자동" if o.get("kappa") is None else f"κ={o['kappa']:g} N·mm/rad/mm")
    if k == "auto_hinge":
        return {"off": "자동 힌지 해제", "auto": "쉘 사이 공유 엣지(그룹 경계와 꺾인 엣지)를 힌지로 지정", "groups": "그룹 경계 엣지를 힌지로 지정", "crease": "꺾인 엣지를 힌지로 지정", "all": "모든 내부 엣지를 힌지로 지정"}[o["mode"]]
    if k == "set_step":
        names = {"static": "정적", "buckle": "좌굴", "modal": "모달", "arclength": "호장법"}
        s = []
        if "type" in o:
            s.append(names[o["type"]])
        if "nlgeom" in o:
            s.append("기하비선형 " + ("켬" if o["nlgeom"] else "끔"))
        for a, b in (("increments", "증분"), ("modes", "모드"), ("imperfection", "초기결함")):
            if a in o and not (a == "imperfection" and o.get("imperfection_modes")):
                s.append(f"{b} {o[a]:g}")
        if o.get("imperfection_modes"):
            s.append("초기결함 " + " + ".join(f"{n}차 모드 x {v:g} mm" for n, v in o["imperfection_modes"]))
        if "control" in o:
            s.append("변위 제어" if o["control"] == "displacement" else "하중 제어")
        if o.get("control_disp"):
            s.append(f"목표 변위 {o['control_disp']:g} mm")
        if o.get("control_load"):
            s.append(f"기준 하중 {o['control_load']}")
        if "control_kind" in o:
            s.append("강체 압판" if o["control_kind"] == "rigid" else "하중 분포 유지")
        if "branch" in o:
            s.append("분기 전환 " + ("켬" if o["branch"] == "auto" else "끔"))
        if "damping" in o:
            d = o["damping"]
            nm = {"none": "감쇠 없음", "mass": "질량 비례 감쇠", "stiffness": "강성 비례 감쇠", "rayleigh": "레일리 감쇠", "direct": "감쇠 직접 입력"}.get(d.get("kind"), "감쇠")
            s.append(nm + (f" {d['zeta']:g}%" if d.get("zeta") else "") + (f", 스텝 시간 {d['time']:g} s" if d.get("time") else ""))
        if o.get("element"):
            s.append("쉘 요소 " + {"auto": "자동", "quad": "사각형", "tri": "삼각형"}[o["element"]])
        if o.get("amplitude"):
            s.append("증가 곡선 " + ("S자" if o["amplitude"] == "smooth" else "선형"))
        if o.get("device"):
            s.append("계산 장치 " + {"auto": "자동", "cpu": "CPU", "cuda": "GPU (CUDA)", "mps": "GPU (Apple)"}[o["device"]])
        if o.get("precision"):
            s.append("강성 정밀도 " + {"auto": "자동", "float64": "배정밀도", "float32": "단정밀도"}[o["precision"]])
        if "threads" in o:
            s.append(f"CPU 스레드 {o['threads']}" if o["threads"] else "CPU 스레드 자동")
        return "해석 설정: " + ", ".join(s)
    if k == "set_solver":
        return f"솔버: {o['solver']}"
    if k == "set_divisions":
        return f"요소 분할수 {o['divisions']}"
    if k == "add_plot":
        return f"그래프: {o['title']}"
    return k


def _snap(model, sel, notes):
    if not isinstance(sel, dict):
        return sel
    for k in ("and", "or"):
        if k in sel:
            return {**sel, k: [_snap(model, x, notes) for x in sel[k]]}
    p = sel.get("plane")
    if p and not isinstance(p.get("at"), str):
        try:
            n = len(resolve(model, {"plane": p}))
        except Exception:
            n = 0
        if n == 0:
            a = AX.index(p["axis"])
            vals = model.nodes[:, a]
            v = float(vals[np.argmin(np.abs(vals - float(p["at"])))])
            notes.append(f"{p['axis']}={float(p['at']):g} 에 절점이 없어 가장 가까운 {p['axis']}={v:g} 로 맞췄습니다")
            return {**sel, "plane": {**p, "at": round(v, 6)}}
    return sel


def parse(text, model, spec):
    ops = []
    ask = []
    notes = []
    unknown = []
    flags = {}
    ctx = {"sel": None}
    work = spec
    for cl in split_clauses(text):
        try:
            o, a, n, u, f = parse_clause(cl, model, work, ctx)
        except Exception as e:
            o, a, n, u, f = [], [], [], [cl], {}
            notes.append(f"'{cl}': {e}")
        if not o and not a and not f and not u and not n:
            u = [cl]
        for x in o:
            if "select" in x:
                x["select"] = _snap(model, x["select"], n)
                try:
                    if len(resolve(model, x["select"])) == 0:
                        a.append(f"'{cl}': 해당 위치에 절점이 없습니다")
                except Exception as e:
                    a.append(str(e))
        ops += o
        ask += a
        notes += n
        if u:
            unknown.append(cl if not o else f"{cl} (남은 말: {' '.join(u)})")
        flags.update(f)
    fixed = []
    for o in ops:
        if o.get("op") != "_friction":
            fixed.append(o)
            continue
        prev = [x for x in fixed if x.get("op") == "add_contact"]
        if prev:
            prev[-1]["friction"] = o["friction"]
        elif spec.get("contacts"):
            fixed += [{"op": "update", "id": x["id"], "friction": o["friction"]} for x in spec["contacts"]]
        else:
            ask.append("마찰계수를 적용할 접촉이 없습니다. 먼저 접촉을 추가하세요 (예: 반지름 20 구를 아래로 3 mm 눌러, 마찰 0.2)")
    ops = fixed
    parts = []
    if ops:
        parts.append(" / ".join(describe_op(o) for o in ops))
    if flags.get("answered"):
        parts += notes
        notes = []
    if notes:
        parts.append("참고: " + " · ".join(notes))
    if ask:
        parts.append("확인 필요: " + " · ".join(ask))
    if unknown:
        parts.append("이해하지 못한 부분: " + " · ".join(unknown))
    if flags.get("undo"):
        parts.append("직전 상태로 되돌립니다")
    if flags.get("run"):
        parts.append("해석을 실행합니다")
    reply = " ".join(parts) if parts else "요청을 이해하지 못했습니다."
    info = {
        "complete": not unknown and not ask and bool(ops or flags),
        "ask": ask,
        "unknown": unknown,
        "run": bool(flags.get("run")),
        "undo": bool(flags.get("undo")),
        "answered": bool(flags.get("answered")),
        "notes": notes,
    }
    return ops, reply, info
