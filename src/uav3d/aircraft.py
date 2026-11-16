"""Reusable replay glyphs (quadcopter, ground vehicle) in body-local coordinates."""
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation


def _ring(radius, z, count, phase=0.):
    a = phase + np.linspace(0, 2*np.pi, count, endpoint=False)
    return np.column_stack((radius*np.cos(a), radius*np.sin(a), np.full(count, float(z))))


def _prism(ring_bottom, ring_top):
    """Caps and side quads of a prism between two matching vertex rings."""
    k = len(ring_bottom)
    sides = [np.array([ring_bottom[j], ring_bottom[(j+1) % k], ring_top[(j+1) % k], ring_top[j]])
             for j in range(k)]
    return [ring_bottom[::-1], ring_top, *sides]


def _box(center, size):
    """Six faces of an axis-aligned box (body frame)."""
    c, (dx, dy, dz) = np.asarray(center, float), np.asarray(size, float)/2
    v = c + np.array([[x, y, z] for z in (-dz, dz) for y in (-dy, dy) for x in (-dx, dx)])
    faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [v[list(f)] for f in faces]


def _shade(color, factor):
    r, g, b, a = to_rgba(color)
    return (min(1., r*factor), min(1., g*factor), min(1., b*factor), a)


class _Glyph:
    """Rigid set of artists: world placement and layering shared by all glyphs."""

    def __init__(self):
        self._parts = []          # (artist, local geometry, kind)
        self._layers = []

    def _add(self, ax, artist, local, kind):
        (ax.add_collection3d(artist) if kind != "line" else None)
        self._parts.append((artist, local, kind))
        self._layers.append((artist, artist.get_zorder()))

    def set_layer(self, offset=0., scale=1.):
        """Move the whole glyph to another layer, keeping its internal order."""
        for artist, zorder in self._layers:
            artist.set_zorder(offset+scale*zorder)

    def _place(self, rotation, position, extra=None):
        def world(points):
            points = np.asarray(points, float)
            return rotation.apply(points.reshape(-1, 3)).reshape(points.shape) + position
        # World geometry of each part, in _parts order (also handy for tests).
        self.world = []
        for artist, local, kind in self._parts:
            local = extra.get(id(artist), local) if extra else local
            if kind == "poly":
                placed = [world(face) for face in local]
                artist.set_verts(placed)
            elif kind == "segments":
                placed = [world(seg) for seg in local]
                artist.set_segments(placed)
            else:
                placed = world(local)
                artist.set_data_3d(placed[:, 0], placed[:, 1], placed[:, 2])
            self.world.append(placed)


class QuadcopterGlyph(_Glyph):
    """Field-scale X-frame quadcopter following a recorded rigid pose.

    Body +X is the nose (the simulation's attitude convention). The two front
    motors are white, the body carries the UAV colour, and the translucent rotor
    discs carry two-blade propellers that spin with mission time.
    """

    ARM = .72                     # motor distance from the centre along each axis
    OUTER_RADIUS = ARM*np.sqrt(2)+.34   # centre to rotor-disc edge, in glyph units

    def __init__(self, ax, color, scale):
        super().__init__()
        s, a = float(scale), self.ARM
        self.motors = np.array([[a, a, 0.], [a, -a, 0.], [-a, -a, 0.], [-a, a, 0.]])
        dark, frame = "#1e293b", "#334155"
        # Arms: flat beams from the hub to each motor (X configuration).
        beams = []
        for m in self.motors:
            u = m[:2]/np.linalg.norm(m[:2]); n = np.array([-u[1], u[0]])*.045
            p0, p1 = np.r_[n, .02], np.r_[m[:2]+n, .02]
            q0, q1 = np.r_[-n, .02], np.r_[m[:2]-n, .02]
            beams.append(np.array([p0, p1, q1, q0]))
        self._add(ax, Poly3DCollection([s*b for b in beams], facecolors=frame, edgecolors=dark,
                                       linewidths=.4, zorder=5), [s*b for b in beams], "poly")
        # Motor pods (hexagonal cylinders); front pair white for heading.
        pods, pod_colors = [], []
        for k, m in enumerate(self.motors):
            faces = _prism(_ring(.10, -.04, 6)+m, _ring(.10, .10, 6)+m)
            pods += faces
            pod_colors += [("#f8fafc" if k < 2 else "#475569")]*len(faces)
        self._add(ax, Poly3DCollection([s*f for f in pods], facecolors=pod_colors, edgecolors=dark,
                                       linewidths=.3, zorder=6), [s*f for f in pods], "poly")
        # Hub: octagonal body in the UAV colour with a lighter canopy.
        hub = _prism(_ring(.30, -.06, 8, np.pi/8), _ring(.30, .12, 8, np.pi/8))
        canopy = _prism(_ring(.20, .12, 8, np.pi/8), _ring(.13, .22, 8, np.pi/8))
        hub_colors = [_shade(color, .55), color, *[_shade(color, .8)]*8]
        canopy_colors = ["#cbd5e1", "#e2e8f0", *["#94a3b8"]*8]
        body = hub+canopy
        self._add(ax, Poly3DCollection([s*f for f in body], facecolors=hub_colors+canopy_colors,
                                       edgecolors=dark, linewidths=.4, zorder=7),
                  [s*f for f in body], "poly")
        # Rotor discs: translucent fill plus a crisp rim.
        discs = [_ring(.34, .14, 24)+m for m in self.motors]
        self._add(ax, Poly3DCollection([s*d for d in discs], facecolors=[to_rgba(color, .22)],
                                       edgecolors=[to_rgba(color, .9)], linewidths=1.0, zorder=8),
                  [s*d for d in discs], "poly")
        # Propeller blades (rebuilt every update so they spin).
        self._scale = s
        self.blades = Line3DCollection(self._blade_segments(0.), colors=dark, linewidths=1.1, zorder=9)
        self._add(ax, self.blades, self._blade_segments(0.), "segments")
        # A white nose stripe on the canopy gives an attitude cue without an arrow.
        nose = s*np.array([[.05, 0, .23], [.42, 0, .23]])
        self.indicator, = ax.plot([], [], [], color="#f8fafc", lw=1.6, zorder=10)
        self._add(ax, self.indicator, nose, "line")

    def _blade_segments(self, t):
        segments = []
        for k, m in enumerate(self.motors):
            angle = (1 if k % 2 else -1)*7.3*t + k   # neighbours counter-rotate
            d = .31*np.array([np.cos(angle), np.sin(angle), 0.])
            segments.append(self._scale*np.array([m-d+[0, 0, .15], m+d+[0, 0, .15]]))
        return segments

    def update(self, position, quaternion, t=0.):
        self._place(Rotation.from_quat(quaternion), np.asarray(position, float),
                    {id(self.blades): self._blade_segments(t)})


class VehicleGlyph(_Glyph):
    """Ground vehicle for the target: body, glass cabin, wheels, headlights.

    Body +X is the front; `update` yaws it to the given heading on the ground.
    """

    def __init__(self, ax, color="#dc2626", scale=1.):
        super().__init__()
        s = float(scale)
        dark = "#111827"
        L, W = 4.6, 2.0
        body = _box((0, 0, .75), (L, W, .8))
        hood = _box((1.55, 0, 1.2), (1.4, W*.94, .12))
        cabin = _box((-.35, 0, 1.45), (2.3, W*.9, .6))
        glass = [f*[1, 1, 1] for f in _box((-.35, 0, 1.45), (2.32, W*.86, .5))[2:]]
        faces = body+hood+cabin
        colors = [_shade(color, .5), color, *[_shade(color, .8)]*4,
                  _shade(color, .6), _shade(color, 1.05), *[_shade(color, .85)]*4,
                  _shade(color, .6), _shade(color, 1.1), *[_shade(color, .9)]*4]
        self._add(ax, Poly3DCollection([s*f for f in faces], facecolors=colors, edgecolors=dark,
                                       linewidths=.4, zorder=3.2), [s*f for f in faces], "poly")
        self._add(ax, Poly3DCollection([s*f for f in glass], facecolors="#1f3a52", edgecolors=dark,
                                       linewidths=.3, alpha=.9, zorder=3.4), [s*f for f in glass], "poly")
        # Wheels: octagonal cylinders with their axle along body Y.
        wheels = []
        for x in (1.45, -1.45):
            for y in (W/2+.02, -W/2-.02):
                a = np.linspace(0, 2*np.pi, 8, endpoint=False)
                rim = np.column_stack((.38*np.cos(a)+x, np.zeros(8), .38*np.sin(a)+.38))
                inner, outer = rim+[0, y-np.sign(y)*.14, 0], rim+[0, y+np.sign(y)*.14, 0]
                wheels += _prism(inner, outer)
        self._add(ax, Poly3DCollection([s*f for f in wheels], facecolors="#1f2937", edgecolors="#0b0f17",
                                       linewidths=.3, zorder=3.0), [s*f for f in wheels], "poly")
        lights = [np.array([[L/2+.01, y-.25, .85], [L/2+.01, y+.25, .85],
                            [L/2+.01, y+.25, 1.05], [L/2+.01, y-.25, 1.05]]) for y in (.6, -.6)]
        self._add(ax, Poly3DCollection([s*f for f in lights], facecolors="#fde68a", edgecolors="#92400e",
                                       linewidths=.3, zorder=3.5), [s*f for f in lights], "poly")

    def update(self, position, heading):
        self._place(Rotation.from_euler("z", float(heading)), np.asarray(position, float))


def headings(points, time, window=.5):
    """Heading [rad] of a ground track at every sample, from motion over ±window s.

    Samples where the track does not move keep the previous heading.
    """
    points, time = np.asarray(points, float), np.asarray(time, float)
    lo = np.searchsorted(time, time-window)
    hi = np.minimum(np.searchsorted(time, time+window, side="right")-1, len(time)-1)
    d = points[hi, :2]-points[lo, :2]
    moving = np.linalg.norm(d, axis=1) > 1e-6
    out = np.arctan2(d[:, 1], d[:, 0])
    last = 0.
    for k in range(len(out)):
        last = out[k] if moving[k] else last
        out[k] = last
    return out
