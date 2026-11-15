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


def test_ideal_backend_reproduces_upstream_dynamics():
    from uav3d.physics import IdealBackend
    drag,tau,dt = 0.1,0.2,0.1
    p = IdealBackend([[1.,2.,30.]],drag,tau)
    import dynamics       # upstream; IdealBackend put it on sys.path
    A,B = dynamics.model(dt,drag,tau,2)
    s = np.zeros(6); s[:2] = [1.,2.]           # [px, py, vx, vy, ax, ay]
    for u in ([5.,0.],[-3.,4.],[0.,0.]):
        p.commands[0,:2] = u
        for _ in range(20):                    # 20 physics steps of 5 ms = one period
            p.step(dt/20)
        s = A@s+B@np.array(u)
        assert np.allclose(p.positions()[0,:2],s[:2]) and np.allclose(p.velocities()[0,:2],s[2:4])
    assert p.positions()[0,2] == 30.
    pv,acc = p.observations()
    assert np.allclose(acc[0,:2]+drag*pv[0,3:5],s[4:6])   # adapter's a = dv/dt + D v
    snap = p.snapshot(); p.commands[0,:2] = [5.,5.]; p.step(.05); p.restore(snap)
    assert np.allclose(p.velocities()[0,:2],s[2:4])
    packed = p.packed()
    assert packed.shape == (1,17) and packed[0,9] == 1.


def test_cli_forwards_physics():
    from uav3d.cli import parser, worker_command
    command = worker_command(parser().parse_args(["run","--physics","ideal"]),"ours","runs/x")
    assert command[command.index("--physics")+1] == "ideal"
    assert parser().parse_args(["run"]).physics == "rotorpy"


def test_speed_guard_passes_safe_commands_and_caps_along_speed():
    from uav3d.physics import speed_guard_settings, speed_guarded
    assert speed_guard_settings("off",10,.2,5) is None
    g = speed_guard_settings("on",10,.2,5)
    assert g["limit"] == 9.5
    u,clamped = speed_guarded([5,0],[2,0],[0,0],g)               # far below: untouched
    assert np.allclose(u,[5,0]) and not clamped
    u,clamped = speed_guarded([3,4],[9.4,0],[3,0],g)             # w = 10 > 9.5: brake along x
    assert clamped and np.isclose(u[0],-2.5) and np.isclose(u[1],4)   # sideways part kept
    u,clamped = speed_guarded([-5,0],[9.4,0],[3,0],g)            # already braking harder
    assert np.allclose(u,[-5,0]) and not clamped
    u,clamped = speed_guarded([5,0],[30,0],[0,0],g)              # far above: brake at most UMAX
    assert np.isclose(u[0],-5)


def test_speed_guard_holds_the_limit_on_the_ideal_plant_without_overshoot():
    from uav3d.physics import IdealBackend, speed_guard_settings
    g = speed_guard_settings("on",10,.2,5)
    p = IdealBackend([[0.,0.,30.]],0.,.2,g)
    p.commands[0,:2] = [4.,3.]                  # |u| = UMAX, held for 20 s (a stuck planner)
    peak = 0.
    for k in range(4000):
        p.step(.005)
        peak = max(peak,np.linalg.norm(p.velocities()[0,:2]))
        if k == 3000:
            p.commands[0,:2] = [-3.,4.]         # sharp turn at top speed
    assert peak <= 9.5+1e-6 and peak > 9.4
    assert p.speed_clamped[0]
    # The planner's next-step speed bound stays feasible: u = 0 keeps |v| below VMAX.
    import dynamics
    A,B = dynamics.model(.1,0.,.2,2)
    s = np.r_[p.positions()[0,:2],p.velocities()[0,:2],p.states[0]["s"][:,2]]
    assert np.linalg.norm((A@s)[2:4]) <= 10
