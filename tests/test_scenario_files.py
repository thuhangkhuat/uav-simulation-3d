"""Adapter scenario files: the nine-UAV scen3_3d_n9 stays consistent with its sources."""
import importlib.util
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    return yaml.safe_load((ROOT/path).read_text())


def _generator():
    spec = importlib.util.spec_from_file_location("make_n9", ROOT/"scripts"/"make_scen3_3d_n9.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scen3_3d_n9_keeps_scen3_path_and_five_drone_params():
    n9, scen3 = _load("scenarios/scen3_3d_n9.yaml"), _load("motion_planning_MPC/scenarios/scen3.yaml")
    box = _load("scenarios/scen1_3d_m3_box035.yaml")
    assert n9["target"] == scen3["target"]
    assert len(n9["starts"]) == 9 and all(s[2] == 30 for s in n9["starts"])
    # Every param of the five-drone 3D scenario, and scen3's own otherwise.
    assert n9["params"] == {**scen3["params"], **box["params"]}


def test_scen3_3d_n9_changes_are_small_and_keep_clearances():
    gen = _generator()
    n9, scen3 = _load("scenarios/scen3_3d_n9.yaml"), _load("motion_planning_MPC/scenarios/scen3.yaml")
    old, new = np.array(scen3["starts"])[:, :2], np.array(n9["starts"])[:, :2]
    moved = np.linalg.norm(new-old, axis=1)
    assert np.all((moved >= 30-1e-6) & (moved <= 80+1e-6))
    rects, circles = n9["obstacles"]["rects"], n9["obstacles"]["circles"]
    assert gen.clearance(new, rects, circles).min() >= 24-1e-6
    assert min(np.linalg.norm(new[i]-new[j]) for i in range(9) for j in range(i)) >= 100-1e-6
    # Obstacles not in scen3 (moved or added) obey the generator's spacing rules.
    wp = np.asarray(n9["target"]["waypoints"], float)
    legs = np.concatenate([np.linspace(a, b, int(np.linalg.norm(b-a))+2) for a, b in zip(wp[:-1], wp[1:])])
    fresh_r = [r for r in rects if r not in scen3["obstacles"]["rects"]]
    fresh_c = [c for c in circles if c not in scen3["obstacles"]["circles"]]
    assert 0 < len(fresh_r)+len(fresh_c) <= 15
    for r in fresh_r:
        assert gen.clearance(legs, [r], []).min() >= 30-1e-6
        assert gen.clearance(new, [r], []).min() >= 24-1e-6
    for c in fresh_c:
        assert gen.clearance(legs, [], [c]).min() >= 30-1e-6
        assert gen.clearance(new, [], [c]).min() >= 24-1e-6
