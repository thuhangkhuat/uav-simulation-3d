import numpy as np
from uav3d.physics import RotorPyBackend


def flight(dt):
    p=RotorPyBackend([[0,0,3]],dt)
    for _ in range(round(10/dt)):
        p.step(dt)
    assert abs(p.positions()[0,2]-3)<.05
    initial_quat=p.states[0]["q"].copy()
    p.commands[0,0]=5
    for _ in range(round(2/dt)):
        p.step(dt)
    assert p.states[0]["v"][0]>5
    assert np.linalg.norm(p.states[0]["q"]-initial_quat)>.05
    p.commands[0,0]=-5
    for _ in range(round(2/dt)):
        p.step(dt)
    p.commands[0,0]=0
    for _ in range(round(4/dt)):
        p.step(dt)
    assert abs(p.positions()[0,2]-3)<.1
    return p.packed()


def test_hover_acceleration_braking_and_step_convergence():
    coarse=flight(.005)
    fine=flight(.0025)
    assert np.max(np.abs(coarse[:,:3]-fine[:,:3]))<.1
    assert np.max(np.abs(coarse[:,3:6]-fine[:,3:6]))<.1


def test_attitude_guard_keeps_vertical_thrust_and_tilt_limit():
    from uav3d.physics import TILT_MAX_DEG, commanded_acceleration
    # Nominal commands pass unchanged: UMAX and 1.5*UMAX at hover are below the limit.
    for u in ([5,0],[0,-7.5]):
        acc,vertical,tilt = commanded_acceleration(u,30.,0.,30.)
        assert np.allclose(acc,[*u,9.81]) and not vertical and not tilt
    # Climbing fast: the unguarded altitude hold points thrust below the horizon.
    raw,_,_ = commanded_acceleration([5,0],31.,3.,30.,guard=False)
    assert raw[2] < 0
    acc,vertical,tilt = commanded_acceleration([5,0],31.,3.,30.)
    assert vertical and tilt and np.isclose(acc[2],0.5*9.81)
    assert np.degrees(np.arctan2(np.linalg.norm(acc[:2]),acc[2])) <= TILT_MAX_DEG+1e-9
    assert np.allclose(acc[:2]/np.linalg.norm(acc[:2]),[1,0])   # direction is kept


def test_attitude_summary_flags_inversion():
    from uav3d.metrics import attitude_summary
    time = np.arange(4)*.5
    aircraft = np.zeros((4,1,17)); aircraft[...,9] = 1; aircraft[...,2] = 30
    aircraft[1:3,0,6:10] = [1,0,0,0]          # 180 degrees about x: upside down
    aircraft[2,0,2] = 27
    m = attitude_summary(time,aircraft,np.zeros((4,1,3)),[30.])
    assert m["tilt_deg_max"] > 179 and m["inverted_episodes"] == 1
    assert np.isclose(m["inverted_s"],1.0) and np.isclose(m["altitude_error_m_max"],3)
    assert "guard_tilt_limit_pct" not in m
