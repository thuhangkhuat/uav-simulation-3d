import json
import os
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from uav3d.cli import parser
from uav3d.upstream import age_robot,set_tracker_interval,UPSTREAM


def test_cli_defaults():
    args=parser().parse_args(["run"])
    assert args.scenario=="scen1"
    assert args.controller=="ours"
    assert args.drones is None
    assert parser().parse_args(["benchmark"]).controllers==[
        "ours","apf","pure_mpc","cbf_qp_lag","impc_ob_rec"]


@pytest.mark.parametrize("count",[5,9])
def test_scenario_preset_preserved_in_fresh_process(count):
    source=f'''
import json,numpy as np,yaml
from uav3d.upstream import load,UPSTREAM
cfg,world,starts=load(drones={count})
raw=yaml.safe_load((UPSTREAM/'scenarios/scen1.yaml').read_text())
assert np.array_equal(starts[:5],raw['starts'])
assert cfg.TIMESTEP==raw['params']['TIMESTEP']
assert cfg.VMAX==raw['params']['VMAX']
assert cfg.CONTROLLER=='ours'
assert cfg.NUM_ROBOT=={count}
assert world.first_hit(starts,starts) is None
assert not [m for level,m in cfg.validate_config() if level=='ERROR']
print(json.dumps(starts.tolist()))
'''
    command=[sys.executable,"-c",source]
    first=subprocess.check_output(command,text=True)
    second=subprocess.check_output(command,text=True)
    assert json.loads(first)==json.loads(second)


def test_prediction_and_fallback_age_by_elapsed_time():
    values=np.arange(6.)[:,None]
    robot=SimpleNamespace(prediction_epoch=0.,states_prediction=values.copy(),
                          controls_prediction=values.copy(),_mpc_X=values.copy(),_mpc_U=values.copy(),
                          _backup=(values.copy(),values.copy()),_bk_i=0,_plan_left=5,
                          simple=SimpleNamespace(left=5,Xp=values.copy(),Up=values.copy()))
    age_robot(robot,.3,.1)
    assert robot.states_prediction[:,0].tolist()==[3,4,5,5,5,5]
    assert robot._bk_i==2  # Fallback advances the remaining one step on invocation.
    assert robot.simple.left==3
    before=robot.states_prediction.copy()
    age_robot(robot,.3,.1)
    np.testing.assert_array_equal(robot.states_prediction,before)


def test_estimator_interval_preserves_state_and_noise_density():
    sys.path.insert(0,str(UPSTREAM))
    from kalman_target import KalmanTargetTracker
    tracker=KalmanTargetTracker(dt=.1,process_noise_std=2.)
    tracker.initialize([3,4])
    set_tracker_interval(tracker,.3)
    assert tracker.F[0,2]==.3
    assert tracker.Q[2,2]==pytest.approx(4*.3**2)
    np.testing.assert_array_equal(tracker.x,[3,4,0,0])


def test_parse_params_and_config_default():
    from uav3d.experiment import parse_params
    from uav3d.upstream import config_default
    assert parse_params(["VMAX=9","IPOPT_OPTIONS={ipopt.max_iter: 100}"]) == \
        {"VMAX":9,"IPOPT_OPTIONS":{"ipopt.max_iter":100}}
    import pytest
    with pytest.raises(ValueError):
        parse_params(["VMAX"])
    assert config_default("IPOPT_OPTIONS")["ipopt.max_iter"] == 500


def test_scenario_overlay_merges_without_touching_upstream(tmp_path):
    import subprocess, sys, yaml
    from uav3d.upstream import overlay_scenario, scenario_file
    source = scenario_file("scen1")
    before = source.read_text()
    path,merged = overlay_scenario("scen1",{"IPOPT_OPTIONS":{"ipopt.max_iter":100},"VMAX":9},tmp_path)
    assert source.read_text() == before and path.name == source.name
    data = yaml.safe_load(path.read_text())
    assert data["params"]["VMAX"] == 9 and data["params"]["UMAX"] == 5     # scenario params kept
    assert merged["IPOPT_OPTIONS"]["ipopt.max_iter"] == 100
    assert merged["IPOPT_OPTIONS"]["ipopt.tol"] == 1e-4                     # merged with default
    # Upstream loads the overlay and recomputes derived parameters from it.
    code = ("from uav3d import upstream;"
            "cfg,_,_=upstream.load('scen1','ours',None,"
            "{'IPOPT_OPTIONS':{'ipopt.max_iter':100},'VMAX':9},%r);"
            "print(cfg.SCENARIO_NAME,cfg.VMAX,cfg.IPOPT_OPTIONS['ipopt.max_iter'],cfg.CORRIDOR_MARGIN)"
            % str(tmp_path/"run"))
    out = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True).stdout.split()
    assert out[:3] == ["scen1","9","100"]
    base = subprocess.run([sys.executable,"-c","from uav3d import upstream;"
                           "cfg,_,_=upstream.load('scen1');print(cfg.CORRIDOR_MARGIN)"],
                          capture_output=True,text=True,check=True).stdout.split()
    assert float(out[3]) < float(base[0])        # derived from VMAX, so it follows the override


def test_target_speed_cap_overlay(tmp_path):
    import subprocess, sys, yaml
    from uav3d.upstream import overlay_scenario, scenario_file
    original = yaml.safe_load(scenario_file("scen1").read_text())
    path,_ = overlay_scenario("scen1",{},tmp_path,target_speed_cap=9)
    data = yaml.safe_load(path.read_text())
    speeds = data["target"]["speeds"]
    assert max(speeds) == 9 and len(speeds) == len(original["target"]["speeds"])
    assert [s for s,o in zip(speeds,original["target"]["speeds"]) if o <= 9] == \
           [o for o in original["target"]["speeds"] if o <= 9]          # slower segments unchanged
    assert data["target"]["waypoints"] == original["target"]["waypoints"]
    code = ("from uav3d import upstream;import numpy as np;"
            "cfg,_,_=upstream.load('scen1','ours',None,None,%r,9.0);"
            "from target_manual import create_target;"
            "print(np.max(cfg.TAR_SPEEDS),len(create_target(verbose=False).trajectory))" % str(tmp_path/"run"))
    out = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True).stdout.split()
    assert float(out[0]) == 9.0
    assert int(out[1]) > 7490          # slower fast segments: a longer target trajectory than scen1


def test_adapter_scenario_copy_reproduces_scen1_and_can_be_edited(tmp_path):
    import subprocess, sys, yaml
    from pathlib import Path
    from uav3d.upstream import ROOT, scenario_file
    copy = ROOT/"scenarios"/"scen1_3d.yaml"
    assert yaml.safe_load(copy.read_text()) == yaml.safe_load(scenario_file("scen1").read_text())
    probe = ("from uav3d import upstream;import json,numpy as np;cfg,_,_=upstream.load(%r);"
             "d=cfg.SCENARIO_DEF;print(json.dumps([cfg.SCENARIO_NAME,d['rects'],d['circles'],d['speeds'],"
             "d['params'],float(cfg.CORRIDOR_MARGIN)]))")
    run = lambda s: __import__("json").loads(subprocess.run([sys.executable,"-c",probe % s],
                    capture_output=True,text=True,check=True).stdout)
    base,mine = run("scen1"),run(str(copy))
    assert mine[0] == "scen1_3d" and mine[1:] == base[1:]
    edited = tmp_path/"edited.yaml"                     # an edit takes effect, derived values follow
    data = yaml.safe_load(copy.read_text()); data["params"]["CORRIDOR_MARGIN"] = 2.8
    edited.write_text(yaml.safe_dump(data))
    assert run(str(edited))[5] == 2.8 and base[5] < 1.0
