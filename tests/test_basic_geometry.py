import numpy as np
import pytest

from minigeo.geometry import (
    BasePoint,
    BaseLine,
    BaseAxis,
    BaseRectangle,
    BaseCircle,
    BaseBox,
    BaseCylinder,
    BaseCone,
    BasePolygon,  # though not used much
    Transform,
)


def test_base_point():
    # Create a point at (1, 1, 1)
    pt = BasePoint(np.array([1, 1, 1]))
    # Verify vertex shape is (1,3)
    assert pt.vertices.shape == (1, 3)
    np.testing.assert_allclose(pt.vertices, np.array([[1, 1, 1]]))

    # Apply translation transformation
    trans = Transform().translate([2, 0, 0])
    trans.apply(pt)
    np.testing.assert_allclose(pt.vertices, np.array([[3, 1, 1]]))


def test_base_line():
    line = BaseLine(np.array([1, 1, 1]), np.array([2, 2, 2]))
    # Verify vertices shape is (2,3)
    assert line.vertices.shape == (2, 3)
    np.testing.assert_allclose(line.vertices[0], np.array([1, 1, 1]))
    np.testing.assert_allclose(line.vertices[1], np.array([2, 2, 2]))

    # Translate line
    Transform().translate([1, 0, 0]).apply(line)
    np.testing.assert_allclose(line.vertices, np.array([[2, 1, 1], [3, 2, 2]]))


def test_base_axis():
    axis = BaseAxis(np.array([1, 0, 0]), np.array([0, 0, 0]))
    # Axis should produce two vertices (origin and one unit along normalized direction)
    assert axis.vertices.shape == (2, 3)
    # Rotate axis by 90 degrees around z-axis, should point along y axis
    axis.rotate("z", 90)
    # New direction should be approximately [0, 1, 0]
    np.testing.assert_allclose(axis.direction, np.array([0, 1, 0]), atol=1e-6)


def test_base_rectangle():
    center = np.array([0, 0, 0])
    dimensions = np.array([2, 4])  # width and length
    rect = BaseRectangle(center, dimensions=dimensions)
    # Check that vertices are computed (4 vertices)
    assert rect.vertices.shape == (4, 3)
    # Expected vertices: centered rectangle
    expected = np.array(
        [
            [-1, -2, 0],
            [1, -2, 0],
            [1, 2, 0],
            [-1, 2, 0],
        ]
    )
    np.testing.assert_allclose(rect.vertices, expected)


def test_base_circle():
    center = np.array([0, 0, 0])
    radius = 3
    circle = BaseCircle(center, dimensions=np.array([radius]))
    # Default number of vertices is 16
    assert circle.vertices.shape[0] == 16
    # Check that all vertices are at distance ~radius from center (in xy-plane)
    distances = np.linalg.norm(circle.vertices[:, :2] - center[:2], axis=1)
    np.testing.assert_allclose(distances, np.full(16, radius), atol=1e-6)


def test_base_box():
    center = np.array([0, 0, 0])
    dimensions = np.array([2, 4, 6])
    box = BaseBox(center, dimensions=dimensions)
    # Check vertices shape: 8 vertices of 3 coordinates each
    assert box.vertices.shape == (8, 3)
    # Check one vertex value: the first vertex should be at (-1, -2, -3)
    np.testing.assert_allclose(box.vertices[0], np.array([-1, -2, -3]))


def test_box_with_center_shift():
    center = np.array([1, 2, 3])
    dimensions = np.array([2, 4, 6])
    box = BaseBox(center, dimensions=dimensions)
    # Check vertices shape: 8 vertices of 3 coordinates each
    assert box.vertices.shape == (8, 3)
    # Check one vertex value: the first vertex should be at (0, 0, 0)
    np.testing.assert_allclose(box.vertices[0], np.array([0, 0, 0]))
    # Check that the center of the box is at (1, 2, 3)
    box_center = np.mean(box.vertices, axis=0)
    np.testing.assert_allclose(box_center, center, atol=1e-6)


def test_base_cylinder():
    center = np.array([0, 0, 0])
    radius = 2
    height = 5
    cyl = BaseCylinder(center, dimensions=np.array([radius, height]))
    # Cylinder calculates vertices with n=16 for bottom, then stacked top vertices => 32 vertices
    assert cyl.vertices.shape == (32, 3)
    # Check that bottom vertices are at z = 0 and top vertices at z = 5
    bottom_z = cyl.vertices[:16, 2]
    top_z = cyl.vertices[16:, 2]
    np.testing.assert_allclose(bottom_z, np.zeros(16), atol=1e-6)
    np.testing.assert_allclose(top_z, np.full(16, 5), atol=1e-6)


def test_base_cone():
    center = np.array([0, 0, 0])
    lower_radius = 2
    upper_radius = 1
    height = 4
    cone = BaseCone(center, dimensions=np.array([lower_radius, upper_radius, height]))
    # Cone calculates vertices similar to cylinder: 32 vertices
    assert cone.vertices.shape == (32, 3)
    # Bottom vertices at z = 0, top vertices at z = height
    bottom_z = cone.vertices[:16, 2]
    top_z = cone.vertices[16:, 2]
    np.testing.assert_allclose(bottom_z, np.zeros(16), atol=1e-6)
    np.testing.assert_allclose(top_z, np.full(16, 4), atol=1e-6)


def test_transform_rotation():
    # Create a point and rotate it 90 deg about the z-axis
    pt = BasePoint(np.array([1, 0, 0]))
    Transform().rotate("z", 90).apply(pt)
    # Expect point to be at approximately (0,1,0)
    np.testing.assert_allclose(pt.vertices, np.array([[0, 1, 0]]), atol=1e-6)

    # Test rotation of a line: rotate line from (1,0,0)-(2,0,0)
    line = BaseLine(np.array([1, 0, 0]), np.array([2, 0, 0]))
    Transform().rotate("z", 90).apply(line)
    expected_start = np.array([0, 1, 0])
    expected_end = np.array([0, 2, 0])
    np.testing.assert_allclose(line.vertices[0], expected_start, atol=1e-6)
    np.testing.assert_allclose(line.vertices[1], expected_end, atol=1e-6)
