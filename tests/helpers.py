import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from femweb.io_obj import parse_obj, refine
from femweb.io_stl import stl_to_solid
from femweb.spec import apply_ops, default_spec
from femweb.hinges import with_auto


def load_example(name):
    presets = json.load(open(os.path.join(ROOT, "examples", "presets.json")))
    p = presets[name]
    path = os.path.join(ROOT, "examples", p["file"])
    if p["file"].endswith(".stl"):
        base = stl_to_solid(open(path, "rb").read(), p["file"], size=p.get("size"))
    else:
        base = parse_obj(open(path).read(), p["file"])
    spec, _ = apply_ops(default_spec(base), p["ops"], None)
    return with_auto(base, spec), spec


def active(base, spec):
    if spec["solver"] == "barhinge" or base.kind != "obj":
        return base
    return refine(base, spec["divisions"])
