"""Build scenarios/scen3_3d_n9.yaml: the 2D nine-UAV scenario (upstream scen3)
with the 3D five-drone settings (scen1_3d_m3_box035) and slightly changed
starts and obstacles. Deterministic (SEED); rerun to regenerate the file.

    python3 scripts/make_scen3_3d_n9.py

Changes to upstream scen3 (the target path and its speeds are kept):
  * params: the same overrides as scenarios/scen1_3d_m3_box035.yaml
    (CBF_BOX_RATIO 0.35, INFLATE_MARGIN 3.5, CORRIDOR_MARGIN 3.8,
    STANDOFF_DISTANCE 8.75, BASELINE_LEADER_STANDOFF 12.5, APFC_RADIUS 12.5,
    IPOPT max_iter 200);
  * starts: each of the nine starts moved 30-80 m in a random direction, flown at
    30 m like scen1 (scen3 uses 3 m);
  * obstacles: a few shifted 15-40 m, a few removed, a few added.
Every new or moved obstacle keeps >= 30 m from the target's legs, >= 30 m from
its waypoints, >= 24 m from every start (scen3's smallest), >= 15 m from other obstacles and 10 m
from the map border (the rules of upstream's obstacle generator, scen3 comments).
"""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"motion_planning_MPC"/"scenarios"/"scen3.yaml"
OUT = ROOT/"scenarios"/"scen3_3d_n9.yaml"
SEED = 2026
ALTITUDE = 30.
SHIFTED, REMOVED, ADDED = 8, 3, 3
PATH_CLEAR, WAYPOINT_CLEAR, START_CLEAR, GAP, BORDER = 30., 30., 24., 15., 10.
START_MOVE, START_OBSTACLE_CLEAR, START_SEPARATION = (30., 80.), 24., 100.

PARAMS = """  # ─── scen3_3d_n9: the 3D settings of scen1_3d_m3_box035 ───
  CORRIDOR_MARGIN: 3.8      # scen1_3d_m3
  STANDOFF_DISTANCE: 8.75    # scen1_3d_m3_box035: = CBF box half side (ours only)
  BASELINE_LEADER_STANDOFF: 12.5  # pinned: baselines keep 0.5 * VIEWING_RADIUS
  APFC_RADIUS: 12.5               # pinned likewise
  IPOPT_OPTIONS: {ipopt.max_iter: 200, ipopt.print_level: 0, ipopt.tol: 1.0e-4,
                    ipopt.acceptable_tol: 1.0e-2, print_time: 0, ipopt.acceptable_iter: 15}
"""


def rect_points(r, step=1.):
    x, y, w, h = r
    u = np.linspace(0, 1, max(2, int((w+h)/step)))
    return np.r_[np.c_[x+w*u, np.full_like(u, y)], np.c_[x+w*u, np.full_like(u, y+h)],
                 np.c_[np.full_like(u, x), y+h*u], np.c_[np.full_like(u, x+w), y+h*u]]


def circle_points(c, step=1.):
    a = np.linspace(0, 2*np.pi, max(8, int(2*np.pi*c[2]/step)), endpoint=False)
    return np.c_[c[0]+c[2]*np.cos(a), c[1]+c[2]*np.sin(a)]


def clearance(points, rects, circles):
    """Distance from points (N, 2) to the nearest footprint (negative inside)."""
    p = np.asarray(points, float).reshape(-1, 2)
    d = np.full(len(p), np.inf)
    for x, y, w, h in rects:
        qx, qy = np.abs(p[:, 0]-(x+w/2))-w/2, np.abs(p[:, 1]-(y+h/2))-h/2
        d = np.minimum(d, np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))+np.minimum(np.maximum(qx, qy), 0))
    for cx, cy, r in circles:
        d = np.minimum(d, np.hypot(p[:, 0]-cx, p[:, 1]-cy)-r)
    return d


def main():
    text = SOURCE.read_text()
    data = yaml.safe_load(text)
    xmin, xmax = data["xlim"]
    ymin, ymax = data["ylim"]
    rng = np.random.default_rng(SEED)
    rects = [list(map(float, r)) for r in data["obstacles"]["rects"]]
    circles = [list(map(float, c)) for c in data["obstacles"]["circles"]]
    wp = np.asarray(data["target"]["waypoints"], float)
    legs = np.concatenate([np.linspace(a, b, int(np.linalg.norm(b-a))+2) for a, b in zip(wp[:-1], wp[1:])])

    # Starts first: 30-80 m away, inside the map, clear of obstacles and of each other.
    starts = np.asarray(data["starts"], float)[:, :2]
    new = []
    for s in starts:
        for _ in range(1000):
            r, a = rng.uniform(*START_MOVE), rng.uniform(0, 2*np.pi)
            q = s+r*np.array([np.cos(a), np.sin(a)])
            if (xmin+BORDER <= q[0] <= xmax-BORDER and ymin+BORDER <= q[1] <= ymax-BORDER
                    and clearance(q, rects, circles)[0] >= START_OBSTACLE_CLEAR
                    and all(np.linalg.norm(q-o) >= START_SEPARATION for o in new)):
                new.append(q)
                break
        else:
            raise RuntimeError(f"no valid new start near {s}")
    starts = np.array(new)

    def fits(kind, shape, rects, circles):
        pts = rect_points(shape) if kind == "rect" else circle_points(shape)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if lo[0] < xmin+BORDER or lo[1] < ymin+BORDER or hi[0] > xmax-BORDER or hi[1] > ymax-BORDER:
            return False
        one = ([shape], []) if kind == "rect" else ([], [shape])
        return (clearance(legs, *one).min() >= PATH_CLEAR and clearance(wp, *one).min() >= WAYPOINT_CLEAR
                and clearance(starts, *one).min() >= START_CLEAR
                and clearance(pts, rects, circles).min() >= GAP)

    def others(kind, k):
        return ([r for j, r in enumerate(rects) if not (kind == "rect" and j == k)],
                [c for j, c in enumerate(circles) if not (kind == "circle" and j == k)])

    def shift(kind, k, lo, hi):
        shape = (rects if kind == "rect" else circles)[k]
        for _ in range(400):
            r, a = rng.uniform(lo, hi), rng.uniform(0, 2*np.pi)
            moved = [shape[0]+r*np.cos(a), shape[1]+r*np.sin(a), *shape[2:]]
            if fits(kind, moved, *others(kind, k)):
                (rects if kind == "rect" else circles)[k] = [round(v, 3) for v in moved]
                return True
        return False

    log = []
    # Obstacles that the new starts now crowd are moved away.
    for kind, items in (("rect", rects), ("circle", circles)):
        for k, shape in enumerate(items):
            one = ([shape], []) if kind == "rect" else ([], [shape])
            if clearance(starts, *one).min() < START_CLEAR:
                ok = shift(kind, k, 15., 60.)
                log.append(f"moved {kind} {k} off a new start" + ("" if ok else " (FAILED, kept)"))
    # A few shifted, removed, added.
    pool = [("rect", k) for k in range(len(rects))]+[("circle", k) for k in range(len(circles))]
    order, done = rng.permutation(len(pool)), []
    for i in order:                    # the first SHIFTED obstacles that have a valid spot
        if len(done) == SHIFTED:
            break
        if shift(*pool[i], 15., 40.):
            done.append(i)
            log.append("shifted %s %d" % pool[i])
    rest = [i for i in order if i not in done]
    removed = sorted((pool[i] for i in rest[:REMOVED]), key=lambda p: -p[1])
    for kind, k in removed:
        (rects if kind == "rect" else circles).pop(k)
        log.append(f"removed {kind} {k}")
    added = 0
    for _ in range(5000):
        if added == ADDED:
            break
        if rng.random() < .6:
            w, h = rng.uniform(20, 60, 2)
            kind, shape = "rect", [rng.uniform(xmin, xmax-w), rng.uniform(ymin, ymax-h), w, h]
        else:
            r = rng.uniform(10, 25)
            kind, shape = "circle", [rng.uniform(xmin+r, xmax-r), rng.uniform(ymin+r, ymax-r), r]
        if fits(kind, shape, rects, circles):
            (rects if kind == "rect" else circles).append([round(v, 3) for v in shape])
            log.append(f"added {kind} at ({shape[0]:.0f}, {shape[1]:.0f})")
            added += 1

    # Write: upstream text with the starts, the obstacles and the params replaced.
    head = text[:text.index("starts:")]
    after_starts = text[text.index("\ntarget:"):text.index("\nobstacles:")]
    header = ("# ============================================================================\n"
              "# scen3_3d_n9: nine UAVs, built by scripts/make_scen3_3d_n9.py from upstream\n"
              "# scenarios/scen3.yaml (motion_planning_MPC @ 188edd7). Same target path and\n"
              "# speeds; starts moved 30-80 m (flown at 30 m like scen1); a few obstacles\n"
              f"# shifted/removed/added (seed {SEED}); params as scen1_3d_m3_box035. Do not edit\n"
              "# by hand: change the script and rerun it.\n"
              "# ============================================================================\n\n")
    head = head.replace(data["description"], data["description"]+" [scen3_3d_n9]", 1)
    starts_text = "starts:                   # [x, y, z]; scen3's starts moved 30-80 m\n" + "".join(
        f"- [{x:.3f}, {y:.3f}, {ALTITUDE:g}]\n" for x, y in starts)
    body = after_starts
    body = body.replace("  CBF_BOX_RATIO: 0.5 ", "  CBF_BOX_RATIO: 0.35  # scen1_3d_m3_box035 (scen3: 0.5)\n  #", 1)
    body = body.replace("  INFLATE_MARGIN: 1.0 ", "  INFLATE_MARGIN: 3.5  # scen1_3d_m3 (scen3: 1.0)\n  #", 1)
    body = body.rstrip("\n")+"\n"+PARAMS
    row = lambda v: "  - [" + ", ".join(repr(round(float(x), 3)) for x in v) + "]\n"
    obstacles = ("\nobstacles:\n  rects: # [x, y, w, h]\n" + "".join(map(row, rects))
                 + "  circles: # [cx, cy, r]\n" + "".join(map(row, circles)))
    OUT.write_text(header+head+starts_text+"\n"+body.lstrip("\n")+obstacles)

    check = yaml.safe_load(OUT.read_text())
    p = check["params"]
    assert (p["CBF_BOX_RATIO"], p["INFLATE_MARGIN"], p["CORRIDOR_MARGIN"]) == (.35, 3.5, 3.8)
    assert len(check["starts"]) == 9 and check["target"] == data["target"]
    assert all(isinstance(v, float) for shape in check["obstacles"]["rects"]+check["obstacles"]["circles"]
               for v in shape)
    print("\n".join(log))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rects)} rects, {len(circles)} circles "
          f"(scen3: {len(data['obstacles']['rects'])}, {len(data['obstacles']['circles'])})")
    print("start moves [m]:", np.round(np.linalg.norm(starts-np.asarray(data["starts"])[:, :2], axis=1), 1).tolist())


if __name__ == "__main__":
    main()
