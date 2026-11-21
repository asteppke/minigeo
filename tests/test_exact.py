"""The shared hook for exact spheres and cylinders."""

import numpy as np
import pytest

from minigeo import Box, Cone, Cylinder, Scene, Sphere, Style, Transform
from minigeo.exact import exact_shape


def exact_of(geometry, **add):
    scene = Scene()
    scene.add(geometry, **add)
    item = scene.items[0]
    from minigeo.render import to_primitives
    from minigeo.scene import _restyle_faces

    primitives = to_primitives(geometry, scene.resolved_style(item))
    if item.face_styles:
        primitives = _restyle_faces(geometry, primitives, item.face_styles)
    return exact_shape(geometry, primitives)


def test_spheres_and_cylinders_qualify_others_not():
    assert exact_of(Sphere([0, 0, 0], 1)).kind == "sphere"
    cylinder = exact_of(Cylinder([0, 0, 0], radius=1, height=2), fill="C1")
    assert cylinder.kind == "cylinder" and len(cylinder.styles) == 3
    assert exact_of(Cone([0, 0, 0], radius=1, height=1)) is None
    assert exact_of(Box([0, 0, 0], size=1)) is None


def test_styles_that_exact_shapes_cannot_show():
    assert exact_of(Sphere([0, 0, 0], 1), fill="none") is None
    assert exact_of(Cylinder([0, 0, 0], radius=1, height=1), face_styles={0: Style(fill="red")}) is None
    top = exact_of(Cylinder([0, 0, 0], radius=1, height=1), fill="blue", face_styles={17: Style(fill="red")})
    assert top.styles[0].fill == "blue" and top.styles[2].fill == "red"


@pytest.mark.parametrize(
    "make",
    [
        lambda: Sphere([1, 2, 3], 0.5, segments=6),
        lambda: Cylinder([1, 2, 3], radius=0.5, height=2, segments=5),
    ],
)
def test_to_world_maps_the_unit_shape_onto_the_geometry(make):
    geometry = make()
    geometry.rotate([1, 1, 0], 37, about="center")
    Transform().scale([1, 0.5, 2]).shear([0.3, 0, 0, 0]) @ geometry
    exact = exact_of(geometry)
    unit = (
        Sphere([0, 0, 0], 1, segments=6) if exact.kind == "sphere" else Cylinder([0, 0, 0], 1, 1, segments=5)
    )
    mapped = np.column_stack([unit.local_vertices, np.ones(len(unit.local_vertices))]) @ exact.to_world.T
    assert np.allclose(mapped[:, :3], geometry.vertices)
    assert np.allclose(exact.from_world @ exact.to_world, np.eye(4))
    corners = exact.box_corners()
    assert corners.shape == (8, 3)


def test_reach_is_the_exact_outline():
    exact = exact_of(Sphere([1, 0, 0], radius=2))
    assert exact.reach([1, 0, 0]) == pytest.approx(2)
    assert np.allclose(exact.center, [1, 0, 0])
    cylinder = exact_of(Cylinder([0, 0, 0], radius=1, height=4, anchor="center"))
    assert cylinder.reach([0, 0, 1]) == pytest.approx(2)
    assert cylinder.reach([1, 0, 0]) == pytest.approx(1)
    diagonal = np.array([1, 0, 1]) / np.sqrt(2)
    assert cylinder.reach(diagonal) == pytest.approx((1 + 2) / np.sqrt(2))


def test_rims_lie_on_the_cylinder():
    exact = exact_of(Cylinder([0, 0, 0], radius=2, height=1))
    rim = exact.rim(0.5, points=12)
    assert rim.shape == (13, 3) and np.allclose(rim[0], rim[-1])
    assert np.allclose(np.linalg.norm(rim[:, :2], axis=1), 2) and np.allclose(rim[:, 2], 1)


def test_mirrored_and_flattened():
    mirrored = Sphere([0, 0, 0], 1)
    Transform().scale([-1, 1, 1]) @ mirrored
    assert np.linalg.det(exact_of(mirrored).to_world[:3, :3]) < 0
    flat = Sphere([0, 0, 0], 1)
    Transform().scale([1, 1, 0]) @ flat
    assert exact_of(flat) is None
