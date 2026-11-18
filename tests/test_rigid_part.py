import numpy as np

from femweb import rules
from femweb.contact import split_rigid
from femweb.io_obj import parse_obj
from femweb.spec import apply_ops, default_spec


def _m():
    return parse_obj(open("examples/four_point_tube.obj").read(), "f.obj")


def test_rules_rigid_part():
    m = _m()
    sp = default_spec(m)
    assert rules.parse("load_l 을 강체로 아래로 20 mm 눌러, 마찰 0.1", m, sp)[0] == [{"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": "load_l"}, "motion": [0.0, 0.0, -20.0], "friction": 0.1}]
    assert rules.parse("support_l 강체 받침으로", m, sp)[0][0]["motion"] == [0.0, 0.0, 0.0]
    assert rules.parse("load_r 를 강체로 묶어줘", m, sp)[0] == [{"op": "add_coupling", "select": {"group": "load_r"}, "rigid": True, "name": "강체 load_r"}]


def test_split_rigid_maps_selections_and_offsets_surface():
    m = _m()
    ops = [{"op": "add_contact", "type": "rigid", "shape": "mesh", "body": {"group": g}, "motion": [0, 0, -1.0]} for g in ("load_l", "load_r", "support_l", "support_r")]
    ops.append({"op": "add_bc", "dofs": [1], "select": {"and": [{"group": "tube"}, {"plane": {"axis": "x", "at": 0}}]}})
    sp, _ = apply_ops(default_spec(m), ops, m)
    sub, sp2, used = split_rigid(m, sp, 0.75)
    assert len(sub.shells) == sum(1 for g in m.shell_group if g == "tube")
    ids = sp2["bcs"][0]["select"]["nodes"]
    assert len(ids) == 32 and np.abs(sub.nodes[ids, 0]).max() < 1e-9
    V = np.array(sp2["contacts"][0]["V"])
    r = np.hypot(V[:, 0] + 60.0, V[:, 2] - 30.5)
    side = np.abs(V[:, 1]) < 25.0
    assert sp2["contacts"][0]["closed"] and np.allclose(r[side], 10.0 - 0.75, atol=0.05)
