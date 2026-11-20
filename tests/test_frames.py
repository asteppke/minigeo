"""Frames and joints: kinematic chains that carry geometry."""

import numpy as np
import pytest

from minigeo import Angle, Box, Cone, Frame, Group, Line, Point, Scene, Transform


@pytest.fixture
def xyz():
    """A serial XYZ stage: X carries Y, Y carries Z."""
    base = Frame()
    x = base.prismatic("x", limits=(0, 0.5), name="X")
    y = x.prismatic("y")
    z = y.prismatic("z")
    return base, x, y, z


def test_root_frame_stands_in_the_world():
    frame = Frame([1, 2, 3])
    assert np.allclose(frame.origin, [1, 2, 3])
    assert np.allclose(frame.axes, np.eye(3))
    assert frame.kind == "fixed"
    assert frame.axis is None
    assert frame.root is frame


def test_serial_chain_carries_everything_below(xyz):
    base, x, y, z = xyz
    tool = z.attach(Point([0, 0, 0]))
    x.value = 0.3
    y.value = 0.1
    z.value = 0.05
    assert np.allclose(tool.position, [0.3, 0.1, 0.05])
    assert np.allclose(z.origin, [0.3, 0.1, 0.05])


def test_values_are_absolute(xyz):
    _, x, _, _ = xyz
    carriage = x.attach(Box([0, 0, 0], size=(1, 1, 1)))
    for value in [0.1, 0.4, 0.2, 0.0]:
        x.value = value
    assert np.allclose(carriage.center, [0, 0, 0])
    x.value = 0.25
    assert np.allclose(carriage.center, [0.25, 0, 0])


def test_many_small_moves_do_not_drift(xyz):
    _, x, _, _ = xyz
    carriage = x.attach(Box([0, 0, 0], size=(1, 1, 1)))
    revolute = Frame().revolute("z", at=[1, 0, 0])
    arm = revolute.attach(Box([2, 0, 0], size=(1, 0.1, 0.1)))
    for _ in range(1000):
        x.move(0.0005)
        revolute.move(0.36)
    assert np.allclose(carriage.center, [x.value, 0, 0])
    assert np.isclose(revolute.value, 360)
    assert np.allclose(arm.center, [2, 0, 0], atol=1e-9)
    assert np.allclose(arm.axes, np.eye(3), atol=1e-9)


def test_move_is_relative_and_chains(xyz):
    _, x, _, _ = xyz
    assert x.move(0.1).move(0.1) is x
    assert np.isclose(x.value, 0.2)


def test_limits_are_checked(xyz):
    _, x, _, _ = xyz
    with pytest.raises(ValueError, match="outside the limits"):
        x.value = 0.6
    with pytest.raises(ValueError, match="smaller value first"):
        Frame().prismatic("x", limits=(1, 0))


def test_fixed_frames_have_no_value():
    mount = Frame().fixed(at=[0, 0, 1])
    assert np.allclose(mount.origin, [0, 0, 1])
    with pytest.raises(ValueError, match="fixed"):
        mount.value = 1


def test_revolute_turns_about_its_axis_through_its_origin():
    hinge = Frame().revolute("z", at=[1, 0, 0])
    arm = hinge.attach(Box([2, 0, 0], size=(1, 0.1, 0.1)))
    hinge.value = 90
    assert np.allclose(arm.center, [1, 1, 0])
    assert np.allclose(arm.axes[0], [0, 1, 0])  # the arm's x' turned with it
    assert np.allclose(hinge.axis, [0, 0, 1])


def test_joint_axis_follows_the_frames_above():
    turntable = Frame().revolute("z")
    slide = turntable.prismatic("x'")
    turntable.value = 90
    assert np.allclose(slide.axis, [0, 1, 0])
    slide.value = 2
    assert np.allclose(slide.origin, [0, 2, 0])


def test_axis_names_world_and_parent():
    tilted = Frame().revolute("x", value=90)
    world_y = tilted.prismatic("y")
    own_y = tilted.prismatic("y'")
    assert np.allclose(world_y.axis, [0, 1, 0])
    assert np.allclose(own_y.axis, [0, 0, 1])
    with pytest.raises(ValueError, match="Unknown axis"):
        tilted.prismatic("w")
    with pytest.raises(ValueError, match="three numbers"):
        tilted.prismatic([0, 0, 0])


def test_vector_axis_is_normalised():
    slide = Frame().prismatic([0, 0, 5])
    slide.value = 1
    assert np.allclose(slide.origin, [0, 0, 1])


def test_child_starts_with_parent_axes_at_given_point():
    turntable = Frame().revolute("z", value=90)
    mount = turntable.fixed(at=[0, 0, 1])
    assert np.allclose(mount.origin, [0, 0, 1])
    assert np.allclose(mount.axes, turntable.axes)


def test_world_moves_of_attached_parts_are_kept(xyz):
    """Option B: a transform moves the part in world coordinates; the joint still carries it."""
    _, x, _, _ = xyz
    carriage = x.attach(Box([0, 0, 0], size=(1, 1, 1)))
    Transform().translate([0, 0, 1]) @ carriage
    x.value = 0.3
    assert np.allclose(carriage.center, [0.3, 0, 1])


def test_place_works_on_attached_parts(xyz):
    _, x, y, _ = xyz
    rail = Box([0, 0, 0], size=(1, 1, 0.1))
    x.value = 0.2
    carriage = x.attach(Box([0, 0, 0], size=(0.3, 1, 0.05)))
    carriage.place("bottom", at=rail.anchor("top"))
    x.value = 0.4
    assert np.isclose(carriage.anchor("bottom")[2], rail.anchor("top")[2])
    assert np.isclose(carriage.center[0], 0.2)


def test_in_frame_moves_along_the_frames_axes():
    hinge = Frame().revolute("z", at=[1, 0, 0])
    arm = hinge.attach(Box([2, 0, 0], size=(1, 0.1, 0.1)))
    hinge.value = 90
    Transform().translate([0.5, 0, 0]).in_frame(hinge) @ arm  # along the arm, now world y
    assert np.allclose(arm.center, [1, 1.5, 0])
    hinge.value = 0
    assert np.allclose(arm.center, [2.5, 0, 0])


def test_in_frame_rotates_about_the_frames_origin():
    box = Box([1, 0, 0], size=(1, 1, 1))
    box.rotate("z", 90, about="center")
    Transform().rotate("x", 90).in_frame(box) @ box  # about the box's own x', which is world y
    assert np.allclose(box.center, [1, 0, 0])
    assert np.allclose(box.axes[2], [1, 0, 0])


def test_in_frame_returns_self():
    transform = Transform()
    assert transform.in_frame(Frame([1, 0, 0])) is transform
    assert np.allclose(transform.matrix, np.eye(4))


def test_moving_a_frame_moves_its_mount_and_children(xyz):
    base, x, y, _ = xyz
    carriage = y.attach(Box([0, 0, 0], size=(1, 1, 1)))
    Transform().translate([0, 0, 2]) @ x
    assert np.allclose(carriage.center, [0, 0, 2])
    assert x.value == 0  # the joint value is unchanged
    x.value = 0.5
    assert np.allclose(carriage.center, [0.5, 0, 2])


def test_turning_a_frame_turns_its_joint_axis(xyz):
    _, x, _, _ = xyz
    x.rotate("z", 90)
    x.value = 0.5
    assert np.allclose(x.origin, [0, 0.5, 0])
    x.rotate("z'", -90, about="center")
    assert np.allclose(x.axis, [1, 0, 0])


def test_frames_cannot_be_scaled():
    with pytest.raises(ValueError, match="only be moved and turned"):
        Transform().scale([2, 1, 1]) @ Frame()


def test_attach_all_kinds_of_geometry():
    slide = Frame().prismatic("x")
    line = Line([0, 0, 0], [0, 0, 1])
    angle = Angle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    group = Group([Box([0, 0, 0], size=(1, 1, 1)), Point([0, 0, 1])])
    assert slide.attach(line, angle, group) == (line, angle, group)
    slide.value = 1
    assert np.allclose(line.start, [1, 0, 0])
    assert np.allclose(angle.vertex, [1, 0, 0])
    assert np.allclose(group.geometries[1].position, [1, 0, 1])
    assert slide.attached == [line, angle, group]


def test_attach_checks():
    base = Frame()
    slide = base.prismatic("x")
    box = Box([0, 0, 0], size=(1, 1, 1))
    slide.attach(box)
    with pytest.raises(ValueError, match="already attached"):
        base.attach(box)
    with pytest.raises(TypeError, match="at least one"):
        slide.attach()
    with pytest.raises(TypeError, match="cannot be moved"):
        slide.attach("box")


def test_detach_leaves_the_geometry_where_it_is():
    slide = Frame().prismatic("x")
    box = slide.attach(Box([0, 0, 0], size=(1, 1, 1)))
    slide.value = 1
    slide.detach(box)
    slide.value = 2
    assert np.allclose(box.center, [1, 0, 0])
    with pytest.raises(ValueError, match="not attached"):
        slide.detach(box)


def test_to_world_and_back(xyz):
    _, x, y, z = xyz
    x.value, y.value, z.value = 0.1, 0.2, 0.3
    assert np.allclose(z.to_world([0, 0, -0.2]), [0.1, 0.2, 0.1])
    assert np.allclose(z.to_local(z.to_world([[1, 2, 3], [4, 5, 6]])), [[1, 2, 3], [4, 5, 6]])


def test_walk_and_repr(xyz):
    base, x, y, z = xyz
    assert list(base.walk()) == [base, x, y, z]
    assert z.root is base
    assert repr(x) == "Frame 'X' (prismatic) at [0. 0. 0.], value 0"
    assert "fixed" in repr(base)


def test_scene_draws_frame_axes_that_follow_the_joint(xyz):
    _, x, _, z = xyz
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)))
    scene.add_frame(z, labels=("x", "y", "z"))
    x.value = 0.4
    origin, axes, length = scene.frames[0].origin_axes_length()
    assert np.allclose(origin, [0.4, 0, 0])
    assert length == 0.5


def test_animation_with_joints(xyz):
    _, x, _, _ = xyz
    scene = Scene()
    carriage = scene.add(x.attach(Box([0, 0, 0], size=(0.2, 0.2, 0.2))))
    snapshots = scene.snapshots(lambda i: x.move(0.1), frames=4)
    assert len(snapshots) == 4
    assert np.allclose(carriage.center, [0.3, 0, 0])


def test_cone_tool_tip_position(xyz):
    _, x, y, z = xyz
    tool = z.attach(Cone([0, 0, 0], radius=0.05, height=0.2, anchor="top"))  # tip down at z origin
    x.value = 0.25
    assert np.allclose(tool.anchor("bottom")[:2], [0.25, 0])


def test_rounding_at_a_limit_is_clamped(xyz):
    _, x, _, _ = xyz
    x.value = 0.5 + 1e-12
    assert x.value == 0.5
