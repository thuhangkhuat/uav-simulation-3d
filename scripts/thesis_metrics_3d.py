"""Thesis metrics (motion_planning_MPC/thesis_metrics.py) on uav3d run directories.

    uv run python scripts/thesis_metrics_3d.py runs/bench-scen1_3d_m3-r1-rep*/ --out results/m3-r1
    uv run python scripts/thesis_metrics_3d.py runs/ours-x runs/ours-y --thesis-dir ../motion_planning_MPC

The metric code is NOT reimplemented: run_metrics, exec_stats, search_split,
COLUMNS, COV_MODE and REFORM_MIN_WINDOW are imported from the thesis checkout
(--thesis-dir, default ../motion_planning_MPC next to this repository), so the
definitions are exactly the ones used for the 2D results. This script only
converts a uav3d run into main.py's data.pkl layout:

  * samples at t = (k+1)*TIMESTEP, k = 0..T-1, T = horizon / TIMESTEP, like
    main.py's path rows (state after step k) and tar_traj (target after update);
  * path rows [t, x, y, z, vx, vy, vz, u(3), a(3)] (a left 0, unused), status
    rows [TRACK, leader, fail streak] as recorded by the adapter;
  * compute_times = each planning call's measured duration (execution_s), per
    round and UAV, before the first collision.

Differences from the 2D pipeline, both forced by the 3D run itself:
  * the first collision t_c comes from the adapter's continuous collision check
    (result.json), not thesis_metrics.alive_mask: a 3D run stops AT contact
    (clearance exactly R, so `clearance < R` would miss it) instead of flying
    on through the obstacle. The team step kc is the first sample at or after t_c;
  * a run that stopped at t_c is padded to the full horizon with its last state.
    Every metric is cut at kc (FOVs dropped, steps lost), so the padding is
    never counted.
"""
import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_thesis(thesis_dir):
    thesis_dir = Path(thesis_dir).resolve()
    if not (thesis_dir/"thesis_metrics.py").is_file():
        raise SystemExit(f"thesis_metrics.py not found in {thesis_dir} (use --thesis-dir)")
    # thesis_metrics imports `metrics` and `evaluate` by name: they must come from
    # the thesis checkout, not from the pinned submodule.
    for name in ("thesis_metrics","metrics","evaluate"):
        sys.modules.pop(name,None)
    sys.path.insert(0,str(thesis_dir))
    tm = importlib.import_module("thesis_metrics")
    if Path(tm.__file__).resolve().parent != thesis_dir:
        raise SystemExit(f"imported {tm.__file__}, expected the copy in {thesis_dir}")
    return tm


def convert(run_dir):
    """(data dict in main.py's layout, kc, T, dt, fov_side, R, info) for one uav3d run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir/"config.json").read_text())
    res = json.loads((run_dir/"result.json").read_text())
    prm = cfg["params"]
    dt = float(cfg["planning_period"])
    T = int(round(cfg["horizon_s"]/dt))
    fov_side = 2*float(prm["VIEWING_RADIUS"])
    R = float(prm["ROBOT_RADIUS"])
    with np.load(run_dir/"trajectory.npz") as tr:
        time,aircraft,target,command,status = (tr[k] for k in ("time","aircraft","target","command","status"))
    n = aircraft.shape[1]
    ts = (np.arange(T)+1)*dt
    idx = np.minimum(np.searchsorted(time,ts-1e-9),len(time)-1)   # first sample at/after t (padding: last)
    path = np.concatenate([ts[:,None,None].repeat(n,1),aircraft[idx][...,0:6],command[idx][...,0:3],
                           np.zeros((T,n,3))],axis=2)                 # (T, n, 13)
    collision = res.get("collision")
    if res.get("status") in ("error","interrupted"):
        raise ValueError(f"{run_dir}: run ended with status {res.get('status')!r}")
    if collision:
        kc = min(T,max(0,math.ceil(collision["time"]/dt-1e-9)-1))
    else:
        if res["mission_time_s"] < cfg["horizon_s"]-1e-6:
            raise ValueError(f"{run_dir}: incomplete run without collision")
        kc = T
    events = [json.loads(l) for l in (run_dir/"events.jsonl").read_text().splitlines()]
    tc = collision["time"] if collision else math.inf
    rounds = {}
    for e in events:
        if e["observed_at"] < tc:
            rounds.setdefault(e["round"],{})[e["drone"]] = e["execution_s"]
    ct = [rounds[r][i] for r in sorted(rounds) if len(rounds[r]) == n for i in range(n)]
    data = {i:dict(path=path[:,i],tar_traj=target[idx],status=status[idx,i].astype(np.int16))
            for i in range(n)}
    data["meta"] = dict(compute_times=np.asarray(ct,float),
                        compute_time_mean=float(np.mean(ct)) if ct else float("nan"),
                        fov_side=fov_side,num_robot=n,scenario=cfg["scenario"])
    info = dict(run=str(run_dir),controller=cfg["controller"],scenario=cfg["scenario"],seed=cfg["seed"],
                compute_ratio=cfg.get("compute_ratio",1.),status=res["status"],
                collision=(f"{collision['reason']} {collision['drones']} @{collision['time']:.2f}s"
                           if collision else ""))
    return data,kc,T,dt,fov_side,R,info


def score(tm, run_dir):
    """thesis_metrics.score with the 3D collision step kc (see the module doc)."""
    data,kc,T,dt,fov_side,R,info = convert(run_dir)
    m = tm.run_metrics(data,fov_side=fov_side,robot_radius=R,timestep=dt,warmup_steps=0,
                       cov_mode=tm.COV_MODE)
    m["exec_ms"] = 1e3*m["exec_time"]
    m["search_path_len"] = tm.search_path_length(data)
    m["success"] = float(kc == T)
    m["first_crash_s"] = float(kc*dt) if kc < T else float("nan")
    m["completion_pct"] = 100.0*kc/T
    if kc < T:
        team = np.repeat((np.arange(T) < kc)[:,None],len([k for k in data if isinstance(k,int)]),axis=1)
        mm = tm.run_metrics(data,fov_side=fov_side,robot_radius=R,timestep=dt,warmup_steps=0,
                            alive=team,cov_mode=tm.COV_MODE)
        m["mission_visibility"],m["mission_cov_eff"] = mm["safe_visibility"],mm["safe_fov_coverage_eff"]
    else:
        m["mission_visibility"],m["mission_cov_eff"] = m["visibility"],m["fov_coverage_eff"]
    m["target_loss_pct"] = 100.0-m["mission_visibility"]
    m.update(tm.exec_stats(data,kc))
    m.update(tm.search_split(data,kc,dt))
    m.update(info)
    return m


def run_dirs(paths):
    out = []
    for p in map(Path,paths):
        if (p/"result.json").is_file():
            out.append(p)
        else:                                   # a benchmark directory: its method subdirectories
            out += sorted(q for q in p.iterdir() if (q/"result.json").is_file())
    return out


def fmt(fmtstr, values):
    vals = [v for v in values if isinstance(v,(int,float)) and not math.isnan(v)]
    if not vals:
        return "-"
    s = fmtstr.format(np.mean(vals))
    if len(vals) > 1:
        s += " ± "+fmtstr.format(np.std(vals, ddof=1))
    return s+("" if len(vals) == len(values) else f" ({len(vals)}/{len(values)})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs",nargs="+",help="run directories or benchmark directories")
    p.add_argument("--thesis-dir",default=str(ROOT.parent/"motion_planning_MPC"))
    p.add_argument("--out",help="write <out>.md and <out>.csv")
    args = p.parse_args(argv)
    tm = load_thesis(args.thesis_dir)
    rows = [score(tm,d) for d in run_dirs(args.runs)]
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"],r["compute_ratio"],r["controller"]),[]).append(r)
    head = ["Method","Runs","Success"]+[title for _,title,_ in tm.COLUMNS]
    lines = ["| "+" | ".join(head)+" |","|"+"---|"*len(head)]
    for (scen,ratio,ctrl),rs in groups.items():
        cells = [f"{ctrl} ({scen}, r={ratio:g})",str(len(rs)),f"{int(sum(r['success'] for r in rs))}/{len(rs)}"]
        cells += [fmt(f,[r.get(k) for r in rs]) for k,_,f in tm.COLUMNS]
        lines.append("| "+" | ".join(cells)+" |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.with_suffix(".md").write_text(
            "Thesis metrics (thesis_metrics.py definitions, cov_mode "
            f"{tm.COV_MODE}) on uav3d runs; mean ± std over runs.\n\n"+table+"\n")
        keys = ["run","controller","scenario","seed","compute_ratio","status","collision"]
        keys += [k for k,_,_ in tm.COLUMNS]+[k for k in tm.EXTRA if k not in keys]
        with out.with_suffix(".csv").open("w",newline="") as f:
            w = csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
