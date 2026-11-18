"""Commands always start experiments in fresh, single-thread-limited processes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from . import CONTROLLERS
from .physics import ATTITUDE_GUARDS, PHYSICS, SPEED_GUARDS
from .scheduler import COMMAND_EXECUTIONS, SCHEDULES


def parser():
    p = argparse.ArgumentParser(description="RotorPy adapter; default preset: upstream scen1")
    sub = p.add_subparsers(dest="command",required=True)
    for name in ("run","benchmark","_worker"):
        q = sub.add_parser(name,help="isolated experiment" if name=="run" else "all five methods" if name=="benchmark" else argparse.SUPPRESS)
        q.add_argument("--scenario",default="scen1",
                       help="upstream scenario name (default scen1), or a path to a scenario YAML, "
                            "e.g. scenarios/scen1_3d.yaml (an editable adapter-side copy)")
        q.add_argument("--controller",choices=CONTROLLERS,default="ours")
        q.add_argument("--drones",type=int,choices=(5,9),default=None)
        q.add_argument("--seed",type=int,default=0)
        q.add_argument("--duration",type=float,help="mission seconds; default: full target trajectory")
        q.add_argument("--physics-dt",type=float,default=.005)
        q.add_argument("--out",help="new output directory (must not already exist)")
        q.add_argument("--timings-from",help="run directory containing previously measured delays")
        q.add_argument("--scheduling",choices=SCHEDULES,default="parallel",
                       help="parallel (default): one worker per UAV, all commands delivered after "
                            "the slowest solve; parallel_own: one worker per UAV, each command "
                            "delivered after its own solve; sequential: one shared worker, delays "
                            "accumulate; async: one worker per UAV and no team round, each UAV "
                            "replans after its own delivery")
        q.add_argument("--compute-ratio",type=float,default=1.,
                       help="scale every measured planning duration by r before injecting it as "
                            "mission-time delay (e.g. 0.5: onboard computer twice as fast); default 1")
        q.add_argument("--physics",choices=PHYSICS,default="rotorpy",
                       help="rotorpy (default): 6-DoF Hummingbird; ideal: upstream's own planar "
                            "model (dynamics.py), to separate 3D model mismatch from other effects")
        q.add_argument("--param",action="append",metavar="KEY=VALUE",
                       help="override an upstream config constant through a scenario overlay "
                            "written to the run directory (VALUE is YAML; dicts merge), e.g. "
                            "--param 'IPOPT_OPTIONS={ipopt.max_iter: 100}'. Repeatable.")
        q.add_argument("--target-speed-cap",type=float,default=None,metavar="V",
                       help="scenario overlay: lower every target segment faster than V m/s to V "
                            "(e.g. scen1's 13 m/s segments); other segments and waypoints unchanged")
        q.add_argument("--start-jitter",type=float,default=None,metavar="M",
                       help="move every start up to M m, exactly as upstream main.py --start-jitter "
                            "(its jitter_starts, seeded by --seed and the scenario name)")
        q.add_argument("--jitter-key",default=None,metavar="NAME",
                       help="scenario name used to seed the start jitter (default: the loaded "
                            "scenario's name); e.g. scen1 to match upstream's 2D scen1 runs")
        q.add_argument("--speed-guard",choices=SPEED_GUARDS,default="off",
                       help="on: plant-side limit keeping every aircraft below VMAX - 0.5 m/s "
                            "(lag-aware, same for all controllers); off (default)")
        q.add_argument("--attitude-guard",choices=ATTITUDE_GUARDS,default="on",
                       help="on (default): keep the commanded vertical acceleration >= 0.5 g and "
                            "the commanded tilt <= 45 deg; off: the unguarded altitude-hold command")
        q.add_argument("--command-execution",choices=COMMAND_EXECUTIONS,default=None,
                       help="tracked_plan_with_watchdog (default for ours): follow the whole delivered "
                            "plan with the adapter tracking/watchdog layer; hold_latest (default for "
                            "the other methods): hold the latest upstream command until the next one")
        if name=="benchmark":
            q.add_argument("--controllers",nargs="+",choices=CONTROLLERS,default=list(CONTROLLERS))
    q = sub.add_parser("metrics",help="recompute timestamp-based metrics from recorded data")
    q.add_argument("directory")
    q = sub.add_parser("replay",help="offline 3D PNG/MP4 rendering; never runs planners")
    q.add_argument("directory")
    q.add_argument("--out",help=".png, .mp4 or .gif output path; default: run/replay.png "
                                "(.gif needs no FFmpeg: handy for quick local previews)")
    q.add_argument("--time",type=float,default=None,help="mission time for PNG; default: final recorded time")
    q.add_argument("--fps",type=int,default=24)
    q.add_argument("--speed",type=float,default=1.,help="mission seconds per video second")
    q.add_argument("--camera",choices=("follow","team","overview"),default="follow",
                   help="follow the target (default), follow the team (midway between the "
                        "aircraft and the target), or show the full map")
    q.add_argument("--zoom",type=float,default=None,
                   help="follow camera magnification relative to the full map (default: 3; "
                        "20 with --true-scale, a ~50 m window on a 1000 m map; see --follow-uav)")
    q.add_argument("--start",type=float,default=None,help="video: first mission second (default: 0)")
    q.add_argument("--end",type=float,default=None,help="video: last mission second (default: run end)")
    q.add_argument("--follow-uav",type=int,default=None,metavar="I",
                   help="camera follows UAV I instead of the target (close-up; default --zoom "
                        "100 with --true-scale, else 8)")
    q.add_argument("--elevation",type=float,default=None,metavar="DEG",
                   help="camera elevation above the horizon (default 55; 80 with --follow-uav, "
                        "35 for --camera overview); 90 = straight down")
    fit = q.add_mutually_exclusive_group()
    fit.add_argument("--fit-team",action="store_true",
                   help="target-follow camera: pick one zoom for the whole clip so that every "
                        "aircraft stays in frame (overrides --zoom)")
    fit.add_argument("--auto-zoom",action="store_true",
                   help="like --fit-team, but the zoom eases in and out over time (about 3 s "
                        "of video per change) to keep every aircraft in frame; --zoom (or its "
                        "default) is the closest it gets")
    q.add_argument("--uav-scale",type=float,default=None,metavar="K",
                   help="draw the aircraft K times larger (illustration only)")
    q.add_argument("--target-scale",type=float,default=None,metavar="K",
                   help="draw the target vehicle K times larger (e.g. 0.6 for smaller)")
    q.add_argument("--annotate",action="store_true",
                   help="draw the title band (scenario, method, time, camera) and the legend")
    q.add_argument("--true-scale",action="store_true",
                   help="physical proportions: aircraft at ROBOT_RADIUS, the target as a ~4.6 m car, "
                        "no vertical exaggeration, default --zoom 20")
    q.add_argument("--draft",action="store_true",
                   help="video: fast low-resolution preview (no 3x supersampling)")
    return p


def worker_command(args, controller, output):
    scenario = args.scenario
    if Path(scenario).suffix in (".yaml",".yml") and Path(scenario).is_file():
        scenario = str(Path(scenario).resolve())    # the worker may not share our cwd
    command = [sys.executable,"-m","uav3d.cli","_worker","--scenario",scenario,
               "--controller",controller,"--seed",str(args.seed),"--physics-dt",str(args.physics_dt),
               "--scheduling",args.scheduling,
               "--compute-ratio",repr(args.compute_ratio),"--physics",args.physics,"--attitude-guard",args.attitude_guard,
               "--speed-guard",args.speed_guard,
               "--out",str(output)]
    for item in getattr(args,"param",None) or ():
        command += ["--param",item]
    for name in ("drones","duration","timings_from","command_execution","target_speed_cap",
                 "start_jitter","jitter_key"):
        value = getattr(args,name)
        if value is not None:
            command += ["--"+name.replace("_","-"),str(value)]
    return command


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command in ("run","benchmark","_worker"):
        import math
        if not math.isfinite(args.physics_dt) or args.physics_dt<=0:
            p.error("--physics-dt must be finite and positive")
        if args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0):
            p.error("--duration must be finite and positive")
        if not math.isfinite(args.compute_ratio) or args.compute_ratio<0:
            p.error("--compute-ratio must be finite and nonnegative")
        if args.target_speed_cap is not None and (not math.isfinite(args.target_speed_cap)
                                                  or args.target_speed_cap<=0):
            p.error("--target-speed-cap must be finite and positive")
        methods = args.controllers if args.command=="benchmark" else [args.controller]
        if args.command_execution=="tracked_plan_with_watchdog" and set(methods)-{"ours"}:
            p.error("--command-execution tracked_plan_with_watchdog applies to ours only")
    if args.command=="_worker":
        from threadpoolctl import threadpool_limits
        from .experiment import run
        with threadpool_limits(limits=1):
            return run(args)
    if args.command in ("run","benchmark"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = Path(args.out or f"runs/{Path(args.scenario).stem}_{stamp}")
        env = os.environ.copy()
        env.update({k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")})
        env.update(PYTHONHASHSEED=str(args.seed),MPLBACKEND="Agg",PYTHONUNBUFFERED="1")
        if args.command=="run":
            return subprocess.call(worker_command(args,args.controller,out),env=env)
        if args.timings_from:
            p.error("--timings-from is supported by run, not a multi-method benchmark")
        out.mkdir(parents=True,exist_ok=False)
        summary = []
        for controller in args.controllers:
            child = out/controller
            code = subprocess.call(worker_command(args,controller,child),env=env)
            record = dict(controller=controller,exit_code=code,directory=str(child))
            if (child/"metrics.json").exists():
                record.update(json.loads((child/"metrics.json").read_text()))
            summary.append(record)
            (out/"benchmark.json").write_text(json.dumps(summary,indent=2)+"\n")
            if code==130:
                return code
        import csv
        keys = sorted({key for row in summary for key in row})
        with (out/"benchmark.csv").open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)
        return int(any(row["exit_code"] for row in summary))
    if args.command=="metrics":
        from .metrics import report
        print(json.dumps(report(args.directory),indent=2))
        return 0
    if args.command=="replay":
        import math
        if args.fps<=0 or args.speed<=0:
            p.error("--fps and --speed must be positive")
        if args.zoom is not None and (not math.isfinite(args.zoom) or args.zoom<=0):
            p.error("--zoom must be finite and positive")
        if args.elevation is not None and not 0 < args.elevation <= 90:
            p.error("--elevation must be in (0, 90]")
        for name in ("uav_scale","target_scale"):
            value = getattr(args,name)
            if value is not None and (not math.isfinite(value) or value<=0):
                p.error(f"--{name.replace('_','-')} must be finite and positive")
        from .replay import render
        render(args)
        return 0


if __name__=="__main__":
    raise SystemExit(main())
