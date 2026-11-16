"""Offline pose rendering using only experiment artifacts."""
import json
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class TargetFollowAxes(Axes3D):
    """Fill a rectangular viewport and pan without changing world geometry."""

    camera_target = None
    camera_zoom = 1.

    def apply_aspect(self, position=None):
        # Axes3D normally shrinks the viewport to a square, leaving side bars.
        # Use the entire allocated rectangle; get_proj compensates its aspect.
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, which="active")

    def get_proj(self):
        projection = super().get_proj()
        if self.camera_target is not None:
            target = projection @ np.r_[self.camera_target, 1.]
            # Translate the camera in its screen plane, preserving heading.
            # Project the fixed world axes together with the scenery.
            for axis in (0, 1):
                projection[axis] = self.camera_zoom * (
                    projection[axis] - target[axis]/target[3]*projection[3])
        # Keep aircraft, trees and paving proportional in a wide viewport.
        projection[0] *= self.bbox.height / self.bbox.width
        return projection


def follow_camera(ax, target, bounds, max_z, zoom, exaggerate=True):
    """Move the camera to the target, keeping world bounds and heading fixed.

    Returns the vertical display scale and the nominal horizontal view window.
    Scenery stays in world coordinates and must not be rebuilt from this window.
    """
    xmin, ymin, xmax, ymax = bounds
    full_span = max(xmax-xmin, ymax-ymin)
    span = full_span/zoom
    # Exaggerate heights on a field-scale map; true proportions otherwise.
    display_z = max(max_z, .11*full_span) if exaggerate else max_z
    scale = display_z/max_z
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(0, max_z))
    ax.set_box_aspect((xmax-xmin, ymax-ymin, display_z))
    ax.set_zticks(np.linspace(0, max_z, 3))
    ax.camera_target = np.asarray(target, dtype=float).copy()
    ax.camera_zoom = zoom
    x, y = target[:2]
    return scale, (x-span/2, y-span/2, x+span/2, y+span/2)


# Layering with ``computed_zorder=False``: paving (0) < hidden overlays
# (HIDDEN_LAYER) < depth-sorted scenery (1) < visible overlays (>= 2).
# Overlay parts whose view ray is blocked by an obstacle volume are drawn
# beneath the scenery, so the obstacle covers them where its mesh is solid
# and they still show through gaps (e.g. between tree crown lobes).
HIDDEN_LAYER = .5
TARGET_COLOR = "#14b8a6"     # teal: not among the UAV colours (matplotlib Set1)


def view_directions(projection, points):
    """Unit world-space directions from each point toward the camera."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.c_[points, np.ones(len(points))]
    screen = homogeneous @ projection.T
    screen = screen/screen[:, 3:]
    # Smaller projected depth is nearer the camera (ortho and perspective).
    screen[:, 2] -= 1e-3
    nearer = screen @ np.linalg.inv(projection).T
    direction = nearer[:, :3]/nearer[:, 3:] - points
    return direction/np.linalg.norm(direction, axis=1, keepdims=True)


def occluded(projection, points, boxes):
    """True where the ray from a point to the camera crosses an obstacle box."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0 or len(boxes) == 0:
        return np.zeros(len(points), dtype=bool)
    direction = view_directions(projection, points)
    direction[np.abs(direction) < 1e-12] = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        lower = (boxes[None, :, 0] - points[:, None])/direction[:, None]
        upper = (boxes[None, :, 1] - points[:, None])/direction[:, None]
    near = np.minimum(lower, upper).max(axis=2)
    far = np.maximum(lower, upper).min(axis=2)
    return (far >= np.maximum(near, 0.)).any(axis=1)


class DepthSplitLine:
    """A 3D polyline drawn above the scenery only where nothing occludes it."""

    def __init__(self, ax, zorder=2., segment=None, **style):
        self.segment = segment  # optional maximum segment length for testing
        self.front = ax.plot([], [], [], zorder=zorder, **style)[0]
        style.pop("label", None)
        self.back = ax.plot([], [], [], zorder=HIDDEN_LAYER+zorder/100, **style)[0]

    def _subdivide(self, points):
        if self.segment is None or len(points) < 2:
            return points
        pieces = [points[:1]]
        for start, end in zip(points[:-1], points[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end-start)/self.segment)))
            pieces.append(start + np.outer(np.arange(1, count+1)/count, end-start))
        return np.concatenate(pieces)

    def update(self, points, projection, boxes):
        points = self._subdivide(np.asarray(points, dtype=float).reshape(-1, 3))
        if len(points) < 2:
            hidden = occluded(projection, points, boxes)
            for artist, mask in ((self.front, ~hidden), (self.back, hidden)):
                p = points[mask]
                artist.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            return
        hidden = occluded(projection, (points[:-1]+points[1:])/2, boxes)
        # Consecutive segments of one layer stay one path (so dash patterns
        # continue); a NaN row breaks the path where the layer changes.
        breaks = np.flatnonzero(np.diff(hidden)) + 1
        runs = zip(np.r_[0, breaks], np.r_[breaks, len(hidden)])
        paths = {False: [], True: []}
        gap = np.full((1, 3), np.nan)
        for start, stop in runs:
            paths[bool(hidden[start])] += [points[start:stop+1], gap]
        for artist, key in ((self.front, False), (self.back, True)):
            p = np.concatenate(paths[key]) if paths[key] else np.empty((0, 3))
            artist.set_data_3d(p[:, 0], p[:, 1], p[:, 2])


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.patches import Rectangle
    from .scene import draw_city, occluder_bounds
    from matplotlib.patches import Patch
    from .aircraft import QuadcopterGlyph, VehicleGlyph, headings
    directory = Path(args.directory)
    cfg = json.loads((directory/"config.json").read_text())
    with np.load(directory/"trajectory.npz") as archive:
        times,states,targets = (archive[k].copy() for k in ("time","aircraft","target"))
    output = Path(args.out or directory/"replay.png")
    if output.suffix.lower() not in (".png",".mp4",".gif"):
        raise ValueError("Replay output must end in .png, .mp4 or .gif")
    start = 0. if getattr(args,"start",None) is None else float(args.start)
    end = float(times[-1]) if getattr(args,"end",None) is None else min(float(args.end),float(times[-1]))
    if output.suffix.lower()!=".png" and not 0 <= start < end:
        raise ValueError(f"Video range must satisfy 0 <= start < end <= {times[-1]:.2f} s")
    output.parent.mkdir(parents=True,exist_ok=True)
    fig = plt.figure(figsize=(12,8),facecolor="#f3f5f9")
    # Explicit layers: ground, depth-sorted obstacles, then aircraft overlays.
    # Automatic collection ordering can put the ground above whole buildings.
    ax = fig.add_axes((0, 0, 1, 1), axes_class=TargetFollowAxes,
                      facecolor="#f3f5f9", computed_zorder=False)
    xmin,ymin,xmax,ymax = cfg["params"]["WORLD_BOUNDS"]
    max_z = max(10.,float(states[:,:,2].max())*1.7)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(0,max_z),
           xlabel="East [m]",ylabel="North [m]",zlabel="Altitude [m]")
    true_scale = getattr(args, "true_scale", False)
    display_z = max_z if true_scale else max(max_z,.22*max(xmax-xmin,ymax-ymin))
    z_scale = display_z/max_z
    ax.set_box_aspect((xmax-xmin,ymax-ymin,display_z))
    ax.set_zticks(np.linspace(0,max_z,3))
    ax.view_init(elev=35,azim=-65)
    camera = getattr(args, "camera", "follow")
    following = camera in ("follow", "team")
    follow_uav = getattr(args, "follow_uav", None)
    if follow_uav is not None and not 0 <= follow_uav < states.shape[1]:
        raise ValueError(f"--follow-uav must be in 0..{states.shape[1]-1}")
    following = following or follow_uav is not None
    # Defaults: a team view (target-follow) or a close-up on one aircraft.
    zoom = getattr(args, "zoom", None) or (
        (100. if true_scale else 8.) if follow_uav is not None else (20. if true_scale else 3.))
    # Close-ups look down steeply: at 55 deg a facade taller than the flight
    # altitude hides an aircraft flying right beside it.
    elevation = getattr(args, "elevation", None)
    if following:
        ax.view_init(elev=elevation or (80. if follow_uav is not None else 55.), azim=-65)
        ax.set_proj_type("ortho")
    elif elevation:
        ax.view_init(elev=elevation, azim=-65)
    ax.set_axis_off()
    annotate = getattr(args, "annotate", False)
    title = None
    if annotate:
        # Figure overlays do not reserve space or shrink the scenery viewport.
        fig.add_artist(Rectangle((0, .925), 1, .075, transform=fig.transFigure,
                                 facecolor="white", edgecolor="none", zorder=10))
        title = fig.text(.5, .988, "", ha="center", va="top", fontsize=11,
                         color="#111111", zorder=11)
    height = max(5.,float(states[:,:,2].max())*1.3)
    clip_view = draw_city(ax, (xmin,ymin,xmax,ymax), cfg["scenario_definition"]["rects"],
                          cfg["scenario_definition"]["circles"], height,
                          true_scale=getattr(args, "true_scale", False))
    # Largest obstacle footprint: a padded view window must contain any obstacle
    # that can reach into the frame, so obstacles never pop in or out.
    obstacle_extent = max([max(w, d) for _, _, w, d in cfg["scenario_definition"]["rects"]] +
                          [2*r for _, _, r in cfg["scenario_definition"]["circles"]] + [0.])
    boxes = occluder_bounds(cfg["scenario_definition"]["rects"],
                            cfg["scenario_definition"]["circles"], height)
    n = states.shape[1]
    colors = plt.cm.Set1(np.arange(n)%9)
    if true_scale:
        # Physical size: the glyph's outer radius equals the configured ROBOT_RADIUS
        # (the collision/safety sphere); the target is a ~4.6 m car.
        glyph = float(cfg["params"]["ROBOT_RADIUS"])/QuadcopterGlyph.OUTER_RADIUS
        vehicle_scale = 1.
    else:
        # Enlarge only the illustrations for field-scale readability.
        glyph = max(1.,(xmax-xmin)*.008)
        vehicle_scale = .25*glyph
    # Optional size multipliers (illustration only; positions are unchanged).
    glyph *= float(getattr(args, "uav_scale", None) or 1.)
    vehicle_scale *= float(getattr(args, "target_scale", None) or 1.)
    half = cfg["params"]["VIEWING_RADIUS"]
    trails,aircraft,footprints = [],[],[]
    for i in range(n):
        trails.append(DepthSplitLine(ax,color=colors[i],lw=1.5,label=f"UAV {i}"))
        aircraft.append(QuadcopterGlyph(ax, colors[i], glyph))
        # Long square edges are split so an obstacle can hide part of one.
        footprints.append(DepthSplitLine(ax,segment=half/8,color=colors[i],lw=.8,alpha=.6))
    target_line = DepthSplitLine(ax,color="#f1f5f9",ls="--",lw=1)
    # The target is drawn as a ground vehicle facing its direction of travel, in
    # a colour outside the UAV palette (matplotlib Set1).
    vehicle = VehicleGlyph(ax, TARGET_COLOR, vehicle_scale)
    target_heading = headings(targets, times)
    if annotate:
        ax.legend(handles=[trail.front for trail in trails]+[Patch(color=TARGET_COLOR,label="Target")],
                  loc="upper left", bbox_to_anchor=(.008, .915), fontsize=8)
    square = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0],[-half,-half,0]])
    def update(t):
        index = max(0,min(len(times)-1,np.searchsorted(times,t,side="right")-1))
        s = states[index]
        # Set the camera first: occlusion is tested against this frame's view.
        if following:
            if follow_uav is not None:
                centre = s[follow_uav,:3]
            elif camera == "team":
                # Midway between the aircraft (at altitude) and the target, so
                # both the formation and the target stay in frame.
                centre = (s[:,:3].mean(axis=0)+targets[index])/2
            else:
                centre = targets[index]
            scale, window = follow_camera(ax, centre, (xmin,ymin,xmax,ymax), max_z, zoom,
                                          exaggerate=not true_scale)
            # Draw only scenery near the view: the full city every frame is slow.
            # Pad by the view span (wide frames, oblique camera, tall facades) and
            # the largest footprint, since clip_view keeps whole obstacles only.
            pad = (window[2]-window[0]) + obstacle_extent + height
            clip_view((window[0]-pad, window[1]-pad, window[2]+pad, window[3]+pad))
        else:
            scale = z_scale
        projection = ax.get_proj()
        stride = max(1,index//1000)
        hidden = occluded(projection, s[:,:3], boxes)
        if follow_uav is not None:
            # The close-up's subject is always drawn: next to a facade taller than
            # its flight altitude it would otherwise vanish behind the roof edge.
            hidden[follow_uav] = False
        for i in range(n):
            # Keep the newest sample so the trail reaches the aircraft.
            trails[i].update(np.r_[states[:index:stride,i,:3], s[None,i,:3]],
                             projection, boxes if i != follow_uav else np.empty((0,2,3)))
            aircraft[i].update(s[i,:3], s[i,6:10], times[index])
            aircraft[i].set_layer(*((HIDDEN_LAYER, .01) if hidden[i] else (0., 1.)))
            footprints[i].update(square+np.r_[s[i,:2],.05], projection, boxes)
        target_line.update(np.r_[targets[:index:stride], targets[None,index]],
                           projection, boxes)
        vehicle.update(np.r_[targets[index,:2],0.], target_heading[index])
        vehicle.set_layer(*((HIDDEN_LAYER, .01) if occluded(projection, np.r_[targets[index,:2],1.],
                                                          boxes)[0] else (0., 1.)))
        camera_label = (f"UAV {follow_uav} close-up · zoom ×{zoom:g}" if follow_uav is not None else
                        f"Team camera · zoom ×{zoom:g}" if camera == "team" else
                        f"Target-follow camera · zoom ×{zoom:g}" if following else "Full-map camera")
        size_label = "true-scale aircraft" if true_scale else "enlarged aircraft glyphs"
        if title is not None:
            title.set_text(f"{cfg['scenario']} · {cfg['controller']} · {n} UAVs · t = {times[index]:.3f} s\n"
                           f"{camera_label} · {size_label} · vertical display scale ×{scale:.1f}")
        return []
    if output.suffix.lower()==".png":
        update(times[-1] if args.time is None else args.time)
        fig.savefig(output,dpi=180)
    else:
        frames = np.r_[np.arange(start,end,args.speed/args.fps),end]
        animation = FuncAnimation(fig,update,frames=frames,interval=1000/args.fps,blit=False)
    draft = getattr(args,"draft",False)
    if output.suffix.lower()==".gif":
        from matplotlib.animation import PillowWriter
        # GIF previews: Pillow only, no FFmpeg; low resolution keeps files small.
        animation.save(output, dpi=60 if draft else 100, writer=PillowWriter(fps=args.fps))
    elif output.suffix.lower()==".mp4" and draft:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        animation.save(output, dpi=100, writer=FFMpegWriter(
            fps=args.fps, codec="libx264", extra_args=["-crf", "23", "-pix_fmt", "yuv420p"]))
    elif output.suffix.lower()==".mp4":
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        # Supersample smoothed mesh faces, then filter to the delivery resolution.
        # Direct rasterization makes subpixel window mullions jump between pixel
        # columns during camera pans. Matching edge strokes seal facade tiles;
        # downsampling the completed frame also stabilizes subpixel details.
        animation.save(output, dpi=480,
                       writer=FFMpegWriter(
                           fps=args.fps, codec="libx264",
                           extra_args=["-vf", "scale=iw/3:ih/3:flags=lanczos",
                                       "-crf", "18", "-pix_fmt", "yuv420p",
                                       "-movflags", "+faststart"]))
    plt.close(fig)
    print(output)
