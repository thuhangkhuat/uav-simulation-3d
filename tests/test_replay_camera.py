import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pytest

from uav3d.cli import main, parser
from uav3d.replay import TargetFollowAxes, follow_camera


@pytest.mark.parametrize("figsize", [(12, 8), (8, 12)])
def test_camera_fills_frame_without_stretching_geometry(figsize):
    fig = plt.figure(figsize=figsize, dpi=60)
    ax = fig.add_axes((0, 0, 1, 1), axes_class=TargetFollowAxes)
    ax.set_axis_off()
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    follow_camera(ax, [500, 500, 0], (0, 0, 1000, 1000), 50, 3)
    try:
        fig.canvas.draw()
        np.testing.assert_allclose(ax.bbox.bounds, fig.bbox.bounds)
        # Equal display-space steps along screen right/up must have equal pixel
        # lengths (after accounting for intentional altitude exaggeration).
        center = np.array([500., 500., 0.])
        world_scale = ax._box_aspect / [1000, 1000, 50]
        points = [center, center+.01*ax._view_u/world_scale,
                  center+.01*ax._view_v/world_scale]
        pixels = np.array([ax.transData.transform(
            proj3d.proj_transform(*p, ax.get_proj())[:2]) for p in points])
        assert np.linalg.norm(pixels[1]-pixels[0]) == pytest.approx(
            np.linalg.norm(pixels[2]-pixels[0]))
    finally:
        plt.close(fig)


def test_follow_keeps_target_projected_at_center_without_zoom_pumping():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    previous_offset = None
    try:
        for target in ([100, 100, 0], [0, 0, 0], [1000, 1000, 20]):
            _, window = follow_camera(ax, target, (0, 0, 1000, 1000), 50, 3)
            projected = proj3d.proj_transform(*target, ax.get_proj())
            np.testing.assert_allclose(projected[:2], (0, 0), atol=1e-12)
            assert ax.get_xlim() == (0, 1000)
            assert ax.get_ylim() == (0, 1000)
            assert window[2]-window[0] == pytest.approx(1000/3)
            assert ax.get_zlim() == (0, 50)
            nearby = np.asarray(target) + (10, 20, 5)
            offset = proj3d.proj_transform(*nearby, ax.get_proj())
            if previous_offset is not None:
                np.testing.assert_allclose(offset[:2], previous_offset, atol=1e-12)
            previous_offset = offset[:2]
            assert (ax.elev, ax.azim) == (55, -65)
    finally:
        plt.close(fig)


def test_replay_defaults_to_target_follow():
    args = parser().parse_args(["replay", "recording"])
    assert args.camera == "follow"
    # None = resolved in render: 3, or 20 (~50 m window) with --true-scale.
    assert args.zoom is None


@pytest.mark.parametrize("zoom", ["0", "-1", "nan", "inf"])
def test_replay_rejects_invalid_zoom(zoom):
    with pytest.raises(SystemExit) as error:
        main(["replay", "recording", "--zoom", zoom])
    assert error.value.code == 2


def test_follow_preserves_scenery_geometry_across_frames():
    from uav3d.scene import draw_city

    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes, computed_zorder=False)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    draw_city(ax, (0, 0, 1000, 1000), [(80, 80, 40, 35)],
              [(160, 100, 22)], 40)
    from unittest.mock import Mock

    setters = []
    for collection in ax.collections:
        collection.set_verts = Mock(wraps=collection.set_verts)
        setters.append(collection.set_verts)
    try:
        for target in ([100, 100, 0], [900, 950, 20]):
            follow_camera(ax, target, (0, 0, 1000, 1000), 50, 3)
            fig.canvas.draw()
            for setter in setters:
                setter.assert_not_called()
            projected = proj3d.proj_transform(*target, ax.get_proj())
            np.testing.assert_allclose(projected[:2], (0, 0), atol=1e-12)
    finally:
        plt.close(fig)


def test_replay_preview_options_parse():
    from uav3d.cli import parser
    args = parser().parse_args(["replay","runs/x","--out","p.gif","--start","232","--end","236","--draft"])
    assert (args.start,args.end,args.draft) == (232.,236.,True)
    args = parser().parse_args(["replay","runs/x"])
    assert args.start is None and args.end is None and not args.draft


def test_true_scale_camera_has_no_vertical_exaggeration():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from uav3d.replay import TargetFollowAxes, follow_camera
    from uav3d.cli import parser
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), axes_class=TargetFollowAxes)
    scale, _ = follow_camera(ax, [500, 500, 0], (0, 0, 1000, 1000), 51., 20., exaggerate=False)
    assert scale == 1.
    scale, _ = follow_camera(ax, [500, 500, 0], (0, 0, 1000, 1000), 51., 3.)
    assert scale > 2
    plt.close(fig)
    assert parser().parse_args(["replay","runs/x","--true-scale"]).zoom is None   # 20 applied in render


def test_close_up_options_parse_and_validate():
    import pytest
    from uav3d.cli import main, parser
    args = parser().parse_args(["replay","runs/x","--follow-uav","3","--elevation","80"])
    assert (args.follow_uav,args.elevation) == (3,80.)
    for bad in ("0","-5","91"):
        with pytest.raises(SystemExit):
            main(["replay","runs/x","--elevation",bad])


def test_team_camera_and_size_options():
    import pytest
    from uav3d.cli import main, parser
    args = parser().parse_args(["replay","runs/x","--camera","team","--uav-scale","3","--target-scale","0.6"])
    assert (args.camera,args.uav_scale,args.target_scale) == ("team",3.,.6)
    for flag in ("--uav-scale","--target-scale"):
        with pytest.raises(SystemExit):
            main(["replay","runs/x",flag,"0"])


def test_fit_zoom_keeps_points_in_frame():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from uav3d.replay import TargetFollowAxes, fit_zoom, follow_camera
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes((0, 0, 1, 1), axes_class=TargetFollowAxes)
    ax.view_init(elev=55, azim=-65); ax.set_proj_type("ortho")
    bounds, centre = (0, 0, 1000, 1000), np.array([500., 500., 0.])
    points = centre + np.array([[60., 0, 30], [-40, 70, 30], [0, -80, 30]])
    zoom = fit_zoom(ax, [centre], [points], bounds, 51., exaggerate=False)
    follow_camera(ax, centre, bounds, 51., zoom, exaggerate=False)
    fig.canvas.draw()
    M = ax.get_proj()
    pix = lambda p: ax.transData.transform(np.c_[proj3d.proj_transform(p[:, 0], p[:, 1], p[:, 2], M)[:2]])
    offset = np.abs(pix(points)-pix(centre[None]))
    # Every point inside the frame, the widest one at the 90 % margin.
    assert offset[:, 0].max() <= .9*ax.bbox.width/2+1 and offset[:, 1].max() <= .9*ax.bbox.height/2+1
    assert max(offset[:, 0].max()/(ax.bbox.width/2), offset[:, 1].max()/(ax.bbox.height/2)) > .89
    plt.close(fig)


def test_smooth_zoom_stays_within_required_and_cap():
    from uav3d.replay import smooth_zoom
    t = np.arange(0, 20, .25)
    required = 20/(1+np.where((t > 5) & (t < 9), 1.5, 0.)) + 3*np.sin(t)   # a sudden spread
    zoom = smooth_zoom(required, .25, 3., cap=18.)
    assert zoom.shape == required.shape
    assert np.all(zoom <= np.minimum(required, 18.)+1e-9)
    # Eases instead of jumping: the per-sample change is far smaller than the step.
    assert np.abs(np.diff(np.log(zoom))).max() < .5*np.abs(np.diff(np.log(required))).max()
    np.testing.assert_allclose(smooth_zoom([5., 5., 5.], .25, 3.), 5.)


def test_auto_zoom_option_parses_and_excludes_fit_team():
    args = parser().parse_args(["replay", "runs/x", "--auto-zoom"])
    assert args.auto_zoom and not args.fit_team
    with pytest.raises(SystemExit):
        parser().parse_args(["replay", "runs/x", "--auto-zoom", "--fit-team"])
