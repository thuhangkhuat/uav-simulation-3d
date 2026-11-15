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
    def __init__(self, starts, physics_dt=0.005, attitude_guard="on"):
        from rotorpy.vehicles.multirotor import Multirotor
        from rotorpy.vehicles.hummingbird_params import quad_params
        self.params = dict(quad_params)
        self.params["motor_noise_std"] = 0.0
        self.dt = physics_dt
        guard_settings(attitude_guard)
        self.guard = attitude_guard == "on"
        self.altitudes = np.asarray(starts, float)[:,2].copy()
        self.commands = np.zeros((len(starts), 3))
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
        acc,_,_ = commanded_acceleration(self.commands[i,:2],s["x"][2],s["v"][2],
                                         self.altitudes[i],self.guard)
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
        self.states = [v.step(s, self.control(i), duration)
                       for i,(v,s) in enumerate(zip(self.vehicles,self.states))]

    def snapshot(self):
        return [{k:np.array(v,copy=True) for k,v in s.items()} for s in self.states]

    def restore(self, snapshot):
        self.states = snapshot

    def packed(self):
        return np.array([np.r_[s["x"],s["v"],s["q"],s["w"],s["rotor_speeds"]]
                         for s in self.states])
