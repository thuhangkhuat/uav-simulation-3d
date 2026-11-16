"""Decorative replay geometry. Never used by sensing or collision detection."""
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _mesh(meshes, faces, colors, **kwargs):
    # Batch obstacle faces so mplot3d sorts individual faces rather than whole
    # trees/buildings. Ground is kept in its own background layer below them.
    for index, face in enumerate(faces):
        color = colors if isinstance(colors, str) else colors[index % len(colors)]
        meshes.append((face, to_rgba(color),
                       to_rgba(kwargs.get("edgecolors", color)),
                       kwargs.get("linewidths", 0)))


def _box(meshes, x, y, width, depth, bottom, top, colors):
    vertices = np.array([
        [x, y, bottom], [x+width, y, bottom],
        [x+width, y+depth, bottom], [x, y+depth, bottom],
        [x, y, top], [x+width, y, top],
        [x+width, y+depth, top], [x, y+depth, top],
    ])
    faces = vertices[np.array([(0, 1, 5, 4), (1, 2, 6, 5),
                               (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)])]
    _mesh(meshes, faces, colors, edgecolors="#343a40", linewidths=.25)


def _building(meshes, x, y, width, depth, height, index):
    palettes = [
        ["#aa9586", "#84776e", "#918175", "#b6a293", "#d0c7bd"],
        ["#9eabb3", "#74838e", "#84939d", "#b0bbc2", "#d2d9dc"],
        ["#b29e8b", "#8c7968", "#9b8878", "#c1ad98", "#ded2c4"],
    ]
    palette = palettes[index % len(palettes)]
    # Tile each facade around its windows instead of laying glass over a huge
    # wall polygon. This avoids mplot3d hiding lower windows during face sorting.
    rows = 4
    for side, span in enumerate((width, depth, width, depth)):
        columns = max(2, min(7, int(span / 9)))

        def panel(u0, u1, z0, z1, color):
            if side in (0, 2):
                wall = y if side == 0 else y+depth
                face = [(x+u0, wall, z0), (x+u1, wall, z0),
                        (x+u1, wall, z1), (x+u0, wall, z1)]
            else:
                wall = x+width if side == 1 else x
                face = [(wall, y+u0, z0), (wall, y+u1, z0),
                        (wall, y+u1, z1), (wall, y+u0, z1)]
            _mesh(meshes, [face], color)

        for row in range(rows):
            bottom, top = height*row/rows, height*(row+1)/rows
            for col in range(columns):
                left, right = span*col/columns, span*(col+1)/columns
                lo, hi = left+(right-left)*.20, right-(right-left)*.20
                z0, z1 = bottom+height*.055, top-height*.045
                door = row == 0 and col == columns//2 and side in (0, 3)
                if door:
                    z0 = 0
                panel(left, lo, bottom, top, palette[side])
                panel(hi, right, bottom, top, palette[side])
                panel(lo, hi, bottom, z0, palette[side])
                panel(lo, hi, z1, top, palette[side])
                # A dark frame, narrow mullion and two blue glass panes.
                frame = min(span/columns*.045, height*.012)
                panel(lo, hi, z0, z0+frame, "#46545c")
                panel(lo, hi, z1-frame, z1, "#46545c")
                panel(lo, lo+frame, z0+frame, z1-frame, "#46545c")
                panel(hi-frame, hi, z0+frame, z1-frame, "#46545c")
                mid = (lo+hi)/2
                panel(mid-frame/2, mid+frame/2, z0+frame, z1-frame, "#46545c")
                lit = (index*3+row*7+col+side) % 13 == 0
                panel(lo+frame, mid-frame/2, z0+frame, z1-frame,
                      "#ebd6a5" if lit else "#6894a6")
                panel(mid+frame/2, hi-frame, z0+frame, z1-frame,
                      "#f5e5bf" if lit else "#9fc1cd")
            # Pale stone courses and darker foundations emphasize each floor.
            band = height*(.018 if row else .035)
            panel(0, span, bottom, bottom+band, "#6a7072" if row == 0 else palette[4])

    # Inset roof, solid coping and four parapets, all inside the footprint.
    rim = min(width, depth)*.035
    _box(meshes, x+rim, y+rim, width-2*rim, depth-2*rim,
         height*.995, height, ["#667078"]*5)
    for bx, by, bw, bd in ((x, y, width, rim), (x, y+depth-rim, width, rim),
                            (x, y+rim, rim, depth-2*rim),
                            (x+width-rim, y+rim, rim, depth-2*rim)):
        _box(meshes, bx, by, bw, bd, height*.98, height*1.055,
             [palette[0], palette[1], palette[2], palette[3], "#e0ddd4"])
    _box(meshes, x+.60*width, y+.57*depth, .23*width, .25*depth,
         height, height*1.09, ["#899397"]*4+["#c5cccd"])
    # Small sloping solar panels create recognizable rooftop silhouettes.
    for row in range(2):
        for col in range(3):
            px, py = x+width*(.14+col*.13), y+depth*(.23+row*.22)
            face = [(px, py, height*1.015), (px+width*.11, py, height*1.015),
                    (px+width*.11, py+depth*.16, height*1.065),
                    (px, py+depth*.16, height*1.065)]
            _mesh(meshes, [face], "#24485e", edgecolors="#9db2ba", linewidths=.35)


def _branch(meshes, start, end, radius):
    start, end = np.asarray(start), np.asarray(end)
    axis = end-start
    axis = axis/np.linalg.norm(axis)
    across = np.cross(axis, [0., 1., 0.])
    across /= np.linalg.norm(across)
    other = np.cross(axis, across)
    angles = np.arange(9)*2*np.pi/8
    ring = np.array([np.cos(a)*across+np.sin(a)*other for a in angles])
    faces = [[start+radius*ring[i], start+radius*ring[i+1],
              end+radius*.55*ring[i+1], end+radius*.55*ring[i]] for i in range(8)]
    _mesh(meshes, faces, ["#70503a", "#826044", "#5d4331", "#987454"])


def _crown(meshes, center, radii, rng, base, sectors=10, bands=6):
    # Irregular triangulated ellipsoids with a consistent sunlight direction.
    # Adjacent triangles share vertices: no cracks or regular striped bands.
    vertices = []
    for band in range(bands+1):
        latitude = np.pi*band/bands
        phase = (band % 2)*np.pi/sectors
        ring = []
        for sector in range(sectors):
            angle = sector*2*np.pi/sectors+phase
            rough = rng.uniform(.90, 1.08) if band not in (0, bands) else 1.
            ring.append(np.asarray(center)+np.asarray(radii)*rough*np.array([
                np.sin(latitude)*np.cos(angle), np.sin(latitude)*np.sin(angle),
                np.cos(latitude)]))
        vertices.append(ring)
    sun = np.array([-.45, -.6, .8])
    sun /= np.linalg.norm(sun)
    faces, colors = [], []
    for band in range(bands):
        for sector in range(sectors):
            nxt = (sector+1) % sectors
            quad = [vertices[band][sector], vertices[band+1][sector],
                    vertices[band+1][nxt], vertices[band][nxt]]
            for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                length = np.linalg.norm(normal)
                if length < 1e-10:
                    continue
                normal /= length
                if np.dot(normal, np.mean(triangle, axis=0)-center) < 0:
                    normal = -normal
                light = .72+.35*max(0, np.dot(normal, sun))+rng.uniform(-.045, .045)
                faces.append(triangle)
                colors.append(tuple(np.clip(np.asarray(base)*light, 0, 1)))
    _mesh(meshes, faces, colors)


def _tree(meshes, x, y, radius, height, index):
    rng = np.random.default_rng(4100+index)
    # Each tree has a branched trunk and an asymmetric, clustered leaf crown.
    # Offsets plus lobe radii stay strictly inside the original obstacle disk.
    top = height*(.92+.07*rng.random())
    _branch(meshes, (x, y, 0), (x, y, top*.66), radius*.10)
    base = [( .24, .47, .27), (.29, .51, .29), (.34, .53, .27)][index % 3]
    for lobe in range(3):
        angle = index*.73+lobe*2*np.pi/3
        cx, cy = x+radius*.34*np.cos(angle), y+radius*.34*np.sin(angle)
        cz = top*(.57+.035*(lobe % 2))
        _branch(meshes, (x, y, top*.30), (cx, cy, cz), radius*.045)
        _crown(meshes, (cx, cy, cz), (radius*.57, radius*.57, top*.30), rng, base)
    _crown(meshes, (x-radius*.08, y+radius*.04, top*.76),
           (radius*.62, radius*.62, top*.24), rng, base)


def _grove(meshes, x, y, radius, height, index):
    """A circular obstacle as a grove of life-size trees (true-scale replay).

    Crowns (3.5-6 m) stay inside the obstacle disk and the trees reach close to
    the obstacle column height, so the drawing still matches what the planners
    avoid: a vertical column, not just the space under a low canopy.
    """
    rng = np.random.default_rng(7300+index)
    crown = float(np.clip(radius*.25, 3.5, 6.))
    reach = max(0., radius-crown)
    step = 1.7*crown
    count = int(np.ceil(reach/step))
    spots = []
    for i in range(-count, count+1):
        for j in range(-count, count+1):
            px = x+step*(i+.5*(j % 2))+rng.uniform(-.25, .25)*crown
            py = y+step*j*.87+rng.uniform(-.25, .25)*crown
            if np.hypot(px-x, py-y) <= reach:
                spots.append((px, py))
    if not spots:
        spots = [(x, y)]
    palette = [(.24, .47, .27), (.29, .51, .29), (.34, .53, .27), (.27, .44, .25)]
    for k, (px, py) in enumerate(spots):
        top = height*rng.uniform(.86, .98)
        r = crown*rng.uniform(.85, 1.)
        _branch(meshes, (px, py, 0), (px, py, top*.62), max(.25, r*.09))
        _crown(meshes, (px, py, top*.70), (r, r, top*.29), rng,
               palette[(index+k) % len(palette)], sectors=8, bands=4)


def _street(path, width, true_scale):
    """Asphalt ribbon along a ground track, with white edge lines.

    Returns (faces, colors): quads on z slightly above the pavement. The track
    is resampled, and each joint uses the mean of its neighbouring segment
    normals so consecutive quads meet without gaps on curves.
    """
    path = np.asarray(path, float)[:, :2]
    step = 3. if true_scale else 6.
    keep = [0]
    for k in range(1, len(path)):
        if np.linalg.norm(path[k]-path[keep[-1]]) >= step:
            keep.append(k)
    if keep[-1] != len(path)-1:
        keep.append(len(path)-1)
    points = path[keep]
    if len(points) < 2:
        return [], []
    tangent = np.diff(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    seg_normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    normal = np.r_[seg_normal[:1], seg_normal[:-1]+seg_normal[1:], seg_normal[-1:]]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    line = .25 if true_scale else 1.2
    bands = [(-width/2, width/2, .02, "#2f3236"),                         # asphalt
             (width/2-.6-line, width/2-.6, .04, "#e5e7eb"),               # edge lines
             (-width/2+.6, -width/2+.6+line, .04, "#e5e7eb")]
    faces, colors = [], []
    for lo, hi, z, color in bands:
        for k in range(len(points)-1):
            a, b = points[k], points[k+1]
            na, nb = normal[k], normal[k+1]
            faces.append([(*(a+lo*na), z), (*(b+lo*nb), z), (*(b+hi*nb), z), (*(a+hi*na), z)])
            colors.append(color)
    return faces, colors


def _clearance(points, rects, circles):
    """Distance from each point (N, 2) to the nearest obstacle footprint."""
    points = np.asarray(points, float).reshape(-1, 2)
    d = np.full(len(points), np.inf)
    if len(rects):
        r = np.asarray(rects, float).reshape(-1, 4)
        qx = np.abs(points[:, None, 0]-(r[None, :, 0]+r[None, :, 2]/2))-r[None, :, 2]/2
        qy = np.abs(points[:, None, 1]-(r[None, :, 1]+r[None, :, 3]/2))-r[None, :, 3]/2
        inside = np.minimum(np.maximum(qx, qy), 0)
        d = np.minimum(d, (np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))+inside).min(axis=1))
    if len(circles):
        c = np.asarray(circles, float).reshape(-1, 3)
        d = np.minimum(d, (np.hypot(points[:, None, 0]-c[None, :, 0], points[:, None, 1]-c[None, :, 1])
                           - c[None, :, 2]).min(axis=1))
    return d


def street_network(track, rects, circles, bounds, width=12., spacing=140., avenues=3, seed=3):
    """The target's track plus a grid of through streets, so the road network
    does not give away the route.

    North-south and east-west streets every `spacing` metres run across the
    map, broken wherever they would come within 4 m (plus half the road width)
    of an obstacle footprint. Each free stretch keeps only the part between its
    outermost junctions (crossings with a kept street or the target's road, or
    the map edge), repeated until nothing changes, so no street dead-ends.
    Deterministic. Returns a list of polylines, the target's track first.
    """
    track = np.asarray(track, float)[:, :2]
    xmin, ymin, xmax, ymax = bounds
    clear = width/2+4.
    step = 2.
    lines = [track]
    if len(track) < 2:
        return lines
    # Free stretches of each grid line: (axis, coordinate, lo, hi, lo_edge, hi_edge).
    stretches = []
    for axis, (c0, c1), (u0, u1) in ((0, (xmin, xmax), (ymin, ymax)), (1, (ymin, ymax), (xmin, xmax))):
        span = c1-c0
        for c in c0+np.arange(spacing/2, span, spacing):
            u = np.arange(u0, u1+1e-9, step)
            pts = np.c_[np.full_like(u, c), u] if axis == 0 else np.c_[u, np.full_like(u, c)]
            free = _clearance(pts, rects, circles) >= clear
            edges = np.flatnonzero(np.diff(np.r_[0, free.astype(int), 0]))
            for lo, hi in zip(edges[::2], edges[1::2]-1):
                if hi > lo:
                    stretches.append([axis, c, u[lo], u[hi], lo == 0, hi == len(u)-1])

    def track_crossings(axis, c, lo, hi):
        # Positions along a grid line where the target's track crosses it.
        v = track[:, axis]-c
        out = []
        for k in np.flatnonzero(np.sign(v[:-1])*np.sign(v[1:]) <= 0):
            if v[k] == v[k+1]:
                continue
            f = v[k]/(v[k]-v[k+1])
            pos = track[k, 1-axis]+f*(track[k+1, 1-axis]-track[k, 1-axis])
            if lo <= pos <= hi:
                out.append(pos)
        return out

    alive = [True]*len(stretches)
    for _ in range(20):
        changed = False
        for i, (axis, c, lo, hi, lo_edge, hi_edge) in enumerate(stretches):
            if not alive[i]:
                continue
            junctions = track_crossings(axis, c, lo, hi)
            junctions += [lo] if lo_edge else []
            junctions += [hi] if hi_edge else []
            for j, (axis2, c2, lo2, hi2, _, _) in enumerate(stretches):
                if alive[j] and axis2 != axis and lo <= c2 <= hi and lo2 <= c <= hi2:
                    junctions.append(c2)
            if len(junctions) < 2 or max(junctions)-min(junctions) < width:
                alive[i] = False
                changed = True
                continue
            new_lo, new_hi = min(junctions), max(junctions)
            if (new_lo, new_hi) != (lo, hi):
                stretches[i][2:6] = [new_lo, new_hi, lo_edge and new_lo == lo, hi_edge and new_hi == hi]
                changed = True
        if not changed:
            break
    for keep, (axis, c, lo, hi, _, _) in zip(alive, stretches):
        if keep:
            u = np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/6.))+1))
            lines.append(np.c_[np.full_like(u, c), u] if axis == 0 else np.c_[u, np.full_like(u, c)])
    # The target's road starts and ends mid-map: join each end to the nearest
    # reachable point of a grid street (or the map edge) with a straight,
    # obstacle-free connector, so the target's road does not dead-end either.
    grid = np.concatenate(lines[1:]) if len(lines) > 1 else np.empty((0, 2))
    for end in (track[0], track[-1]):
        candidates = [np.array(p) for p in grid]
        candidates += [np.array([xmin, end[1]]), np.array([xmax, end[1]]),
                       np.array([end[0], ymin]), np.array([end[0], ymax])]
        for goal in sorted(candidates, key=lambda q: np.linalg.norm(q-end)):
            length = np.linalg.norm(goal-end)
            if length < 1.:
                break
            samples = end+np.outer(np.arange(0., length, step)/length, goal-end)
            if (_clearance(samples, rects, circles) >= clear).all():
                lines.append(np.array([end, goal]))
                break
    lines += _avenues(lines, rects, circles, bounds, width, clear, avenues, seed)
    return lines


def _avenues(lines, rects, circles, bounds, width, clear, count, seed, tries=80):
    """A few long, gently curving avenues across the map, so the target's road
    is not the only curved one. Each runs diagonally to the grid from one map
    edge to the opposite one, bending smoothly, steering round obstacles (keeping `clear` from them)
    and crossing the grid; one that runs alongside another road (near it and
    within ~25 degrees of parallel) for more than 3x the road width, cannot pass an obstacle, or does not reach the far edge
    within twice the map size is dropped. Both ends lie on the map edge, so
    they never dead-end. Deterministic."""
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds
    span = max(xmax-xmin, ymax-ymin)
    step = 2.

    def turn(h, a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([c*h[0]-s_*h[1], s_*h[0]+c*h[1]])

    def resampled(line):
        line = np.asarray(line, float)[:, :2]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        at = np.r_[np.arange(0., arc[-1], step), arc[-1]]
        return np.c_[np.interp(at, arc, line[:, 0]), np.interp(at, arc, line[:, 1])]

    existing = np.concatenate([resampled(l) for l in lines])
    found = []
    for _ in range(tries):
        if len(found) >= count:
            break
        # Diagonal to the grid (about 30-45 degrees), so an avenue crosses grid
        # streets instead of running along them: west edge, low -> east edge, high,
        # or south edge, right -> north edge, left; alternating.
        w, h = xmax-xmin, ymax-ymin
        a0, a1 = rng.uniform(.05, .35), rng.uniform(.65, .95)
        if len(found) % 2 == 0:
            start, goal = np.array([xmin, ymin+a0*h]), np.array([xmax, ymin+a1*h])
        else:
            start, goal = np.array([xmin+a1*w, ymin]), np.array([xmin+a0*w, ymax])
        heading = (goal-start)/np.linalg.norm(goal-start)
        wave, phase = rng.uniform(250., 450.), rng.uniform(0, 2*np.pi)
        amplitude = rng.uniform(.004, .008)                # rad per metre of curvature
        path, p, ok = [start.copy()], start.copy(), False
        for travelled in np.arange(step, 2*span, step):
            # Smoothly varying curvature, pulled back towards the far edge.
            to_goal = (goal-p)/max(np.linalg.norm(goal-p), 1e-9)
            pull = np.arctan2(heading[0]*to_goal[1]-heading[1]*to_goal[0], heading@to_goal)
            heading = turn(heading, amplitude*step*np.sin(2*np.pi*travelled/wave+phase) + .01*pull)
            if _clearance(p+6.*heading, rects, circles)[0] < clear+2.:
                for a in (.3, -.3, .6, -.6):
                    if _clearance(p+6.*turn(heading, a), rects, circles)[0] >= clear+2.:
                        heading = turn(heading, a)
                        break
                else:
                    break
            p = p+step*heading
            if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
                if travelled > span/2:
                    path.append(np.clip(p, (xmin, ymin), (xmax, ymax)))
                    ok = True
                break
            path.append(p.copy())
        if not ok:
            continue
        line = np.array(path)
        # Reject one that runs alongside another road (merging asphalt): near
        # another road AND nearly parallel to it (within ~25 degrees) for more
        # than 3x the road width. Crossings at an angle are fine.
        from scipy.spatial import cKDTree
        tree = cKDTree(existing)
        d, k = tree.query(line)
        along = existing[np.minimum(k+1, len(existing)-1)]-existing[np.maximum(k-1, 0)]
        along /= np.maximum(np.linalg.norm(along, axis=1, keepdims=True), 1e-9)
        own = np.gradient(line, axis=0)
        own /= np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-9)
        parallel = (d < width) & (np.abs(np.sum(along*own, axis=1)) > np.cos(np.radians(25)))
        flag = np.r_[0, parallel.astype(int), 0]
        runs = np.flatnonzero(np.diff(flag) == -1)-np.flatnonzero(np.diff(flag) == 1)
        if len(runs) and runs.max()*step > 3*width:
            continue
        found.append(line)
        existing = np.concatenate([existing, resampled(line)])
    return found


def draw_city(ax, bounds, rects, circles, height, true_scale=False, street=None):
    """Draw concrete pavement, an optional street, buildings and trees.

    Display scale (default): large slabs and one large tree per circular
    obstacle stay legible on a kilometre-scale map. true_scale: 6 m slabs and
    each circular obstacle as a grove of life-size trees. street: a ground
    track (e.g. the target's) drawn as a 12 m asphalt road with edge lines,
    plus a grid of through streets (street_network) so the road hides the route.
    """
    xmin, ymin, xmax, ymax = bounds
    # Square concrete slabs; coarse on kilometre-scale maps to bound mesh
    # complexity. Geometry and colors are deterministic across frames.
    brick = 6. if true_scale else max(xmax-xmin, ymax-ymin)/48
    # Extend the decorative floor on every side of the simulation domain.
    # A fixed world-span apron keeps paving under the panning camera near map
    # edges, including negative X/Y, without moving the world axes or obstacles.
    apron = 48*brick if not true_scale else 200.
    ground_bounds = (xmin-apron, ymin-apron, xmax+apron, ymax+apron)
    gxmin, gymin, gxmax, gymax = ground_bounds
    faces, colors = [], []
    palette = ["#b9bdc1", "#b4b8bc", "#bec2c5", "#afb3b7", "#babdc0"]
    rows = int(np.ceil(apron/brick))
    for row in range(-rows, int(np.ceil((ymax-ymin)/brick))+rows):
        y = ymin+row*brick
        top = min(gymax, y+brick)
        for column in range(-rows, int(np.ceil((xmax-xmin)/brick))+rows):
            x = xmin+column*brick
            left, right = max(gxmin, x), min(gxmax, x+brick)
            faces.append([(left, y, 0), (right, y, 0),
                          (right, top, 0), (left, top, 0)])
            colors.append(palette[(row*7+column*3) % len(palette)])
    street_faces, street_colors = [], []
    if street is not None:
        network = street_network(street, rects, circles, bounds)
        # Streets that reach the map edge carry on to the edge of the pavement,
        # so they leave the picture instead of stopping at the simulation bounds.
        extended = [network[0]]
        for line in network[1:]:
            line = np.asarray(line, float)
            for at, inward in ((0, 1), (-1, -2)):
                end, before = line[at], line[inward]
                on_edge = min(abs(end[0]-xmin), abs(end[0]-xmax), abs(end[1]-ymin), abs(end[1]-ymax)) < 1.
                if on_edge and np.linalg.norm(end-before) > 1e-9:
                    out = (end-before)/np.linalg.norm(end-before)
                    far = end+out*apron
                    line = np.vstack([far, line]) if at == 0 else np.vstack([line, far])
            extended.append(line)
        network = extended

        def resampled(line, spacing=3.):
            line = np.asarray(line, float)[:, :2]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
            at = np.arange(0., arc[-1]+1e-9, spacing)
            return np.c_[np.interp(at, arc, line[:, 0]), np.interp(at, arc, line[:, 1])]
        centrelines = [resampled(line) for line in network]
        for index, line in enumerate(network):
            f, c = _street(line, 12., true_scale)
            # Break this road's edge lines where they cross another road.
            others = [o for k, o in enumerate(centrelines) if k != index]
            others = np.concatenate(others) if others else np.empty((0, 2))
            for face, color in zip(f, c):
                if color == "#e5e7eb" and len(others):
                    centre = np.mean(np.asarray(face)[:, :2], axis=0)
                    if np.min(np.sum((others-centre)**2, axis=1)) < (6.5)**2:
                        continue
                street_faces.append(face)
                street_colors.append(color)
    # The camera always looks down on z=0. Draw paving first: averaging depths
    # of wide brick faces with tall obstacle faces can otherwise paint bricks
    # over trunks and lower walls, especially in the zoomed follow view.
    ground = Poly3DCollection([], edgecolors="#9da1a5", linewidths=.3 if true_scale else .45, zorder=0)
    ax.add_collection3d(ground, autolim=False)
    obstacles = []
    for index, (x, y, width, depth) in enumerate(rects):
        meshes = []
        _building(meshes, x, y, width, depth, height, index)
        obstacles.append(((x, y, x+width, y+depth), meshes))
    for index, (x, y, radius) in enumerate(circles):
        meshes = []
        (_grove if true_scale else _tree)(meshes, x, y, radius, height, index)
        obstacles.append(((x-radius, y-radius, x+radius, y+radius), meshes))
    # Smooth moving window edges; matching edge strokes seal adjacent tiles.
    scenery = Poly3DCollection([], zorder=1, antialiaseds=True, snap=False)
    ax.add_collection3d(scenery, autolim=False)
    # Added after the scenery (collection order is relied on elsewhere); zorder
    # still draws them first. The street lies on the pavement, below every hidden overlay (HIDDEN_LAYER).
    # Asphalt and markings are separate layers: one depth-sorted collection
    # would paint asphalt over the far edge line. Matching edge strokes seal
    # the joints between asphalt quads.
    road = Poly3DCollection([], linewidths=.5, zorder=.1)
    ax.add_collection3d(road, autolim=False)
    marks = Poly3DCollection([], edgecolors="none", linewidths=0, zorder=.15)
    ax.add_collection3d(marks, autolim=False)
    asphalt = np.array([c != "#e5e7eb" for c in street_colors], bool)
    street_boxes = np.array([[np.min(np.asarray(f)[:, 0]), np.min(np.asarray(f)[:, 1]),
                              np.max(np.asarray(f)[:, 0]), np.max(np.asarray(f)[:, 1])]
                             for f in street_faces]).reshape(-1, 4)

    def clip_view(window):
        """Show only scenery inside window; mplot3d does not clip polygons.

        Pavers are cut exactly at the window edge. Obstacles are shown whole
        or not at all, so none overhangs the axes box. Returns the paving.
        """
        x0, y0, x1, y1 = window
        kept, fills = [], []
        for face, color in zip(faces, colors):
            (left, bottom, _), _, (right, top, _), _ = face
            left, right = max(left, x0), min(right, x1)
            bottom, top = max(bottom, y0), min(top, y1)
            if left < right and bottom < top:
                kept.append([(left, bottom, 0), (right, bottom, 0),
                             (right, top, 0), (left, top, 0)])
                fills.append(color)
        ground.set_verts(kept)
        ground.set_facecolor(fills)
        near = ((street_boxes[:, 2] >= x0) & (street_boxes[:, 0] <= x1) &
                (street_boxes[:, 3] >= y0) & (street_boxes[:, 1] <= y1))
        for layer, mask in ((road, near & asphalt), (marks, near & ~asphalt)):
            chosen = np.flatnonzero(mask)
            layer.set_verts([street_faces[k] for k in chosen])
            layer.set_facecolor([street_colors[k] for k in chosen])
            if layer is road:
                layer.set_edgecolor([street_colors[k] for k in chosen])
        visible = [mesh for (left, bottom, right, top), meshes in obstacles
                   if x0 <= left and right <= x1 and y0 <= bottom and top <= y1
                   for mesh in meshes]
        # Batch faces so mplot3d sorts individual faces, not whole objects.
        polys, face_colors, edges, widths = zip(*visible) if visible else ([],)*4
        scenery.set_verts(list(polys))
        scenery.set_facecolor(list(face_colors))
        scenery.set_edgecolor(list(edges))
        scenery.set_linewidth([max(width, .2) for width in widths])
        return kept

    clip_view(ground_bounds)
    return clip_view


def occluder_bounds(rects, circles, height):
    """Axis-aligned volumes enclosing every decorative obstacle mesh.

    Used only by replay layering: a volume encloses its mesh (parapets and the
    rooftop plant reach 1.09 × height; tree crowns stay inside their disk and
    below ``height``), so a view ray that misses the volume misses the mesh.
    """
    boxes = [((x, y, 0.), (x+width, y+depth, height*1.09))
             for x, y, width, depth in rects]
    boxes += [((x-radius, y-radius, 0.), (x+radius, y+radius, height))
              for x, y, radius in circles]
    return np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
