"""RotorPy implementation of the replaceable physics boundary."""
from typing import Protocol
import numpy as np

G = 9.81
# Altitude hold (adapter): cmd_acc_z = g + KZ*(h - z) - KVZ*vz.
ALTITUDE_KP, ALTITUDE_KD = 4.0, 4.0
# Attitude guard (adapter safety layer, outside the planners). RotorPy's cmd_acc
# points the body z axis along the commanded vector and projects it onto the
# current body axis for thrust, so a nonpositive vertical part asks the aircraft
# to tilt past 90 degrees, after which thrust (and control authority) vanishes.
# The guard keeps the vertical part at or above AZ_MIN_RATIO*g and scales only the
# horizontal part down to the tilt limit. At hover thrust the limit allows
# g*tan(45 deg) = 9.81 m/s^2, above both UMAX and the tracking limit 1.5*UMAX.
AZ_MIN_RATIO, TILT_MAX_DEG = 0.5, 45.0
ATTITUDE_GUARDS = ("on", "off")


def guard_settings(mode="on"):
    """The record written to config.json; None when the guard is off."""
    if mode not in ATTITUDE_GUARDS:
        raise ValueError(f"Unknown attitude guard {mode!r}")
    return None if mode == "off" else dict(az_min_ratio=AZ_MIN_RATIO,tilt_max_deg=TILT_MAX_DEG,
                                           altitude_kp=ALTITUDE_KP,altitude_kd=ALTITUDE_KD)


# Speed guard (adapter safety layer at the plant, identical for every controller).
# With the first-order response a' = (u - a)/tau, the speed the aircraft settles
# at if the command dropped to zero now is w = v + tau*a, and w' = u exactly.
# Capping the command component along w by GAIN*(limit - |w|) keeps |w| <= limit,
# and v, a low-pass of w, follows below it without the lag overshoot of a guard
# on |v| alone. With limit = VMAX - MARGIN, a planner that observes v and a can
# always keep its next speed below VMAX (u = 0 already does).
SPEED_GUARD_MARGIN, SPEED_GUARD_GAIN = 0.5, 5.0
SPEED_GUARDS = ("off", "on")


def speed_guard_settings(mode, vmax, tau, umax):
    """The record written to config.json; None when the guard is off."""
    if mode not in SPEED_GUARDS:
        raise ValueError(f"Unknown speed guard {mode!r}")
    if mode == "off":
        return None
    return dict(limit=float(vmax)-SPEED_GUARD_MARGIN,margin=SPEED_GUARD_MARGIN,vmax=float(vmax),
                tau=float(tau),gain=SPEED_GUARD_GAIN,brake=float(umax))


def speed_guarded(u_xy, v_xy, a_xy, guard):
    """Horizontal command after the speed guard, and whether it changed it."""
    u = np.array(u_xy[:2],float)
    if guard is None:
        return u,False
    w = np.asarray(v_xy[:2],float)+guard["tau"]*np.asarray(a_xy[:2],float)
    speed = float(np.linalg.norm(w))
    if speed < 1e-9:
        return u,False
    e = w/speed
    along = float(u@e)
    cap = max(guard["gain"]*(guard["limit"]-speed),-guard["brake"])
    if along <= cap:
        return u,False
    return u+(cap-along)*e,True


def commanded_acceleration(u_xy, z, vz, altitude, guard=True):
    """cmd_acc for RotorPy, plus (vertical_clamped, tilt_limited) flags.

    u_xy is the horizontal acceleration delivered by the planner path; it is
    passed through unchanged unless the guard must clamp it.
    """
    u = np.array(u_xy[:2],float)
    az = G+ALTITUDE_KP*(altitude-z)-ALTITUDE_KD*vz
    vertical = tilt = False
    if guard:
        if az < AZ_MIN_RATIO*G:
            az,vertical = AZ_MIN_RATIO*G,True
        limit = az*np.tan(np.radians(TILT_MAX_DEG))
        norm = float(np.linalg.norm(u))
        if norm > limit:
            u,tilt = u*(limit/norm),True
    return np.array([u[0],u[1],az]),vertical,tilt


class PhysicsBackend(Protocol):
    commands: np.ndarray
    def positions(self) -> np.ndarray: ...
    def velocities(self) -> np.ndarray: ...
    def observations(self) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, duration: float) -> None: ...
    def snapshot(self) -> list[dict]: ...
    def restore(self, snapshot: list[dict]) -> None: ...
    def packed(self) -> np.ndarray: ...


class RotorPyBackend:
    def __init__(self, starts, physics_dt=0.005, attitude_guard="on", speed_guard=None):
        from rotorpy.vehicles.multirotor import Multirotor
        from rotorpy.vehicles.hummingbird_params import quad_params
        self.params = dict(quad_params)
        self.params["motor_noise_std"] = 0.0
        self.dt = physics_dt
        guard_settings(attitude_guard)
        self.guard = attitude_guard == "on"
        self.altitudes = np.asarray(starts, float)[:,2].copy()
        self.commands = np.zeros((len(starts), 3))
        self.speed_guard = speed_guard
        # Horizontal acceleration estimated from the last physics step (for the speed guard).
        self.accel_estimate = np.zeros((len(starts), 2))
        self.speed_clamped = np.zeros(len(starts), bool)
        hover = np.sqrt(self.params["mass"]*9.81/(4*self.params["k_eta"]))
        self.states = [dict(x=np.array(p, float), v=np.zeros(3), q=np.array([0.,0.,0.,1.]),
                            w=np.zeros(3), wind=np.zeros(3), rotor_speeds=np.full(4, hover))
                       for p in starts]
        self.vehicles = [Multirotor(self.params, initial_state=s, control_abstraction="cmd_acc",
                                   aero=False, enable_ground=False,
                                   integrator_kwargs={"rtol":1e-6, "atol":1e-9})
                         for s in self.states]

    def control(self, i):
        s = self.states[i]
        u,self.speed_clamped[i] = speed_guarded(self.commands[i,:2],s["v"],self.accel_estimate[i],
                                                self.speed_guard)
        acc,_,_ = commanded_acceleration(u,s["x"][2],s["v"][2],self.altitudes[i],self.guard)
        return {"cmd_acc": acc}

    def positions(self):
        return np.array([s["x"] for s in self.states])

    def velocities(self):
        return np.array([s["v"] for s in self.states])

    def observations(self):
        pv = np.array([np.r_[s["x"],s["v"]] for s in self.states])
        acc = np.array([v.statedot(s, self.control(i), self.dt)["vdot"]
                        for i,(s,v) in enumerate(zip(self.states,self.vehicles))])
        return pv, acc

    def step(self, duration):
        before = self.velocities()[:,:2]
        self.states = [v.step(s, self.control(i), duration)
                       for i,(v,s) in enumerate(zip(self.vehicles,self.states))]
        if duration > 1e-4:   # skip the tiny remainders of split steps
            self.accel_estimate = (self.velocities()[:,:2]-before)/duration

    def snapshot(self):
        return dict(states=[{k:np.array(v,copy=True) for k,v in s.items()} for s in self.states],
                    accel=self.accel_estimate.copy())

    def restore(self, snapshot):
        self.states,self.accel_estimate = snapshot["states"],snapshot["accel"]

    def packed(self):
        return np.array([np.r_[s["x"],s["v"],s["q"],s["w"],s["rotor_speeds"]]
                         for s in self.states])


PHYSICS = ("rotorpy", "ideal")


class IdealBackend:
    """Upstream's own plant model (dynamics.py), for isolating 3D model mismatch.

    Each horizontal axis follows exactly what the upstream MPC predicts and what
    upstream's own simulator integrates: p' = v, v' = a - D v, a' = (u - a)/tau,
    discretised exactly (zero-order hold) for every step duration, so splitting
    a planning period into physics steps reproduces the period's transition.
    Altitude is held, attitude stays level, and there are no rotors or attitude
    guard; the speed guard applies as on RotorPy.
    """

    def __init__(self, starts, drag, tau, speed_guard=None):
        import sys
        from .upstream import UPSTREAM
        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0,str(UPSTREAM))
        import dynamics
        self._axis_model = dynamics.axis_model
        self.drag, self.tau = float(drag), float(tau)
        starts = np.asarray(starts,float)
        n = len(starts)
        self.commands = np.zeros((n,3))
        self.speed_guard = speed_guard
        self.speed_clamped = np.zeros(n,bool)
        # Per drone: rows (x, y), columns [p, v, a]; z held at its start.
        self.states = [dict(s=np.column_stack([p[:2],np.zeros(2),np.zeros(2)]),z=float(p[2]))
                       for p in starts]
        self._models = {}

    def _model(self, dt):
        key = round(dt,12)
        if key not in self._models:
            self._models[key] = self._axis_model(dt,self.drag,self.tau)
        return self._models[key]

    def positions(self):
        return np.array([[*st["s"][:,0],st["z"]] for st in self.states])

    def velocities(self):
        return np.array([[*st["s"][:,1],0.] for st in self.states])

    def observations(self):
        # The adapter maps measured dv/dt to upstream's a = dv/dt + D v; return dv/dt.
        pv = np.column_stack([self.positions(),self.velocities()])
        acc = np.array([[*(st["s"][:,2]-self.drag*st["s"][:,1]),0.] for st in self.states])
        return pv,acc

    def step(self, duration):
        Ad,Bd = self._model(duration)
        for i,st in enumerate(self.states):
            u,self.speed_clamped[i] = speed_guarded(self.commands[i,:2],st["s"][:,1],st["s"][:,2],
                                                    self.speed_guard)
            st["s"] = st["s"]@Ad.T+u[:,None]*Bd[:,0][None]

    def snapshot(self):
        return [dict(s=st["s"].copy(),z=st["z"]) for st in self.states]

    def restore(self, snapshot):
        self.states = snapshot

    def packed(self):
        # Same 17-column layout as RotorPy: level attitude, no rotors.
        return np.array([np.r_[st["s"][:,0],st["z"],st["s"][:,1],0.,0.,0.,0.,1.,np.zeros(7)]
                         for st in self.states])
