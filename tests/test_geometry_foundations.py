"""
The geometric facts minigeo is built on, checked through its public API.

The identities are taken from the test suite of `galgebra <https://github.com/pygae/galgebra>`_
(BSD-3-Clause, (c) 2014-2019 Alan Bromborsky and the GAlgebra team), mainly
``test/test_cheatsheet.py``, ``test/test_test.py``, ``test/test_mv.py``, ``test/test_lt.py``
and ``test/test_versors.py``. galgebra checks them symbolically for multivectors; here they
are checked numerically, for random vectors, in the vector algebra of 3D space:

* the outer product ``a ^ b`` corresponds to the cross product ``a x b`` (its dual),
* the contraction ``a | (b ^ c)`` to ``-(a x (b x c))``,
* a rotor sandwich ``R x ~R`` to a rotation matrix, and a versor of two vectors to two
  reflections,
* an outermorphism ``f(x ^ y) = f(x) ^ f(y)`` to the rule for how normals transform.

Each test names the galgebra test it comes from. The last part checks minigeo's own
promises that follow from these facts: transforms move vertices exactly, shapes keep
perpendicular axes, and faces keep pointing outward.
"""

import numpy as np
import pytest

from minigeo import Angle, Box, Cone, Cylinder, Dimension, Line, Plane, Sphere, Transform, Vector
from minigeo.geometry import face_normal

SEEDS = [0, 1, 2, 3]
X, Y, Z = np.eye(3)


@pytest.fixture(params=SEEDS)
def rng(request):
    return np.random.default_rng(request.param)


@pytest.fixture
def abcd(rng):
    """Four random vectors."""
    return [Vector(rng.normal(size=3)) for _ in range(4)]


def random_unit(rng) -> np.ndarray:
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def random_matrix(rng, *, mirrored: bool = False) -> np.ndarray:
    """A random, well-conditioned invertible 3x3 matrix (with determinant < 0 if ``mirrored``)."""
    while True:
        matrix = np.eye(3) + 0.6 * rng.normal(size=(3, 3))
        determinant = np.linalg.det(matrix)
        if abs(determinant) > 0.2 and np.linalg.cond(matrix) < 20 and (determinant < 0) == mirrored:
            return matrix


def linear(matrix: np.ndarray, offset=(0, 0, 0)) -> Transform:
    transform = Transform()
    transform.matrix[:3, :3] = matrix
    transform.matrix[:3, 3] = offset
    return transform


def reflection(normal: np.ndarray) -> Transform:
    """
    The mirror in the plane through the origin with this unit normal, built from minigeo's
    own operations: turn the normal onto x, mirror x, turn back.
    """
    axis = np.cross(X, normal)
    angle = np.degrees(np.arctan2(np.linalg.norm(axis), normal @ X))
    if np.linalg.norm(axis) < 1e-12:
        axis = Z
    return Transform().rotate(axis, -angle).scale([-1, 1, 1]).rotate(axis, angle)


def volume(shape) -> float:
    """
    The volume enclosed by a shape's faces (divergence theorem), positive if every face
    winds counter-clockwise seen from outside.
    """
    total = 0.0
    for face in shape.faces:
        area_vector = 0.5 * np.cross(face, np.roll(face, -1, axis=0)).sum(axis=0)
        total += face[0] @ area_vector / 3
    return total


# ---------------------------------------------------------------- products of vectors


def test_dot_is_symmetric_and_cross_is_antisymmetric(abcd):
    """test_cheatsheet::test_geometric_product_decomposition: a.b and a^b are the symmetric and
    antisymmetric parts of the geometric product."""
    a, b, _, _ = abcd
    assert a.dot(b) == pytest.approx(b.dot(a))
    assert np.allclose(a.cross(b).components, -b.cross(a).components)
    assert np.allclose(a.cross(a).components, 0)


def test_lagrange_identity(abcd):
    """|ab|^2 = |a|^2 |b|^2 with ab = a.b + a^b: the squared parts add up."""
    a, b, _, _ = abcd
    assert a.cross(b).length ** 2 + a.dot(b) ** 2 == pytest.approx(a.length**2 * b.length**2)


def test_bac_cab(abcd):
    """test_test::test_check_generalized_BAC_CAB_formulas: a|(b^c) = (a.b)c - (a.c)b."""
    a, b, c, _ = abcd
    left = a.cross(b.cross(c)).components
    right = b.components * a.dot(c) - c.components * a.dot(b)
    assert np.allclose(left, right)


def test_jacobi_identity(abcd):
    """test_test: (a|(b^c)) + (c|(a^b)) + (b|(c^a)) = 0; test_cheatsheet::test_commutator_product."""
    a, b, c, _ = abcd
    total = a.cross(b.cross(c)) + b.cross(c.cross(a)) + c.cross(a.cross(b))
    assert np.allclose(total.components, 0)


def test_binet_cauchy(abcd):
    """test_cheatsheet::test_bivector_identities: (a^b).(c^d) = (b.c)(a.d) - (a.c)(b.d)."""
    a, b, c, d = abcd
    left = a.cross(b).dot(c.cross(d))
    assert left == pytest.approx(a.dot(c) * b.dot(d) - a.dot(d) * b.dot(c))


def test_triple_product_is_a_cyclic_volume(abcd):
    """test_test: a(b^c) - b(a^c) + c(a^b) = 3 a^b^c: the trivector is the oriented volume."""
    a, b, c, _ = abcd
    volume_abc = a.dot(b.cross(c))
    assert volume_abc == pytest.approx(b.dot(c.cross(a)))
    assert volume_abc == pytest.approx(c.dot(a.cross(b)))
    assert volume_abc == pytest.approx(-a.dot(c.cross(b)))
    assert volume_abc == pytest.approx(np.linalg.det([a.components, b.components, c.components]))


def test_the_axes_are_right_handed():
    """test_cheatsheet::test_pseudoscalar: e1 I^-1 = -(e2^e3), i.e. x, y, z are right-handed."""
    x, y, z = Vector(X), Vector(Y), Vector(Z)
    assert np.allclose(x.cross(y).components, Z)
    assert np.allclose(y.cross(z).components, X)
    assert np.allclose(z.cross(x).components, Y)
    assert x.dot(y.cross(z)) == pytest.approx(1)


def test_lengths_are_never_negative(abcd):
    """test_mv::test_norm_nonneg and test_norm2: |a|^2 = a.a >= 0."""
    for vector in abcd:
        assert vector.length >= 0
        assert vector.length**2 == pytest.approx(vector.dot(vector))
        assert vector.direction @ vector.direction == pytest.approx(1)
    assert Vector([0, 0, 0]).length == 0


# ---------------------------------------------------------------- projections


def test_projection_is_linear_and_idempotent(abcd):
    """test_mv::test_proj: proj(u, v) = (v.u)/u and proj(w, v) + proj(w, u) = proj(w, u + v)."""
    u, v, w, _ = abcd
    on_u = v.projected_onto(u)
    assert np.allclose(on_u.components, v.dot(u) / u.dot(u) * u.components)
    assert np.allclose(
        (v.projected_onto(w) + u.projected_onto(w)).components, (u + v).projected_onto(w).components
    )
    assert np.allclose(on_u.projected_onto(u).components, on_u.components)  # projecting twice
    assert np.allclose(v.projected_onto(3 * u).components, on_u.components)  # only u's direction counts
    assert (v - on_u).dot(u) == pytest.approx(0, abs=1e-12)  # the rest is perpendicular


def test_projection_map_kills_the_other_axes():
    """test_lt::test_lt_callable_zero: the projection onto e1 maps e2 and e3 to zero."""
    onto_x = Transform().scale([1, 0, 0])
    assert np.allclose(onto_x.apply_to_vertex(X), X)
    assert np.allclose(onto_x.apply_to_vertex(Y), 0)
    assert np.allclose(onto_x.apply_to_vertex(Z), 0)


def test_projection_onto_a_plane(rng):
    """The rejection from the normal: what is left after removing the part along it."""
    plane = Plane(rng.normal(size=3), rng.normal(size=3))
    point = rng.normal(size=3) * 3
    foot = plane.project(point)
    assert plane.contains(foot)
    assert np.allclose(plane.project(foot), foot)
    offset = Vector(point - foot)
    assert np.allclose(np.cross(offset.components, plane.normal), 0)  # straight along the normal
    assert offset.length == pytest.approx(abs(plane.distance_to(point)))
    assert plane.distance_to(foot + 2 * plane.normal) == pytest.approx(2)  # positive on the normal's side


@pytest.mark.parametrize("normal", [Z, -Z, X, -Y, [1, 1, 1], [0, 0, -1e-3]])
def test_plane_keeps_the_direction_of_its_normal(normal):
    """Including the special cases straight up and straight down."""
    plane = Plane([0, 0, 0], normal)
    expected = np.asarray(normal, float) / np.linalg.norm(normal)
    assert np.allclose(plane.normal, expected)
    assert np.allclose(plane.axes @ plane.axes.T, np.eye(3))
    assert np.linalg.det(plane.axes) == pytest.approx(1)


def test_plane_through_three_points(rng):
    """test_test::test_properties_of_geometric_objects: (p2-p1)^(p3-p1) = p1^p2 - p1^p3 + p2^p3."""
    p1, p2, p3 = rng.normal(size=(3, 3))
    plane = Plane.through(p1, p2, p3)
    for point in (p1, p2, p3):
        assert plane.contains(point, tolerance=1e-9)
    area_vector = np.cross(p2 - p1, p3 - p1)
    assert np.allclose(area_vector, np.cross(p1, p2) + np.cross(p2, p3) + np.cross(p3, p1))
    assert np.allclose(plane.normal, area_vector / np.linalg.norm(area_vector))  # right-hand rule
    with pytest.raises(ValueError):
        Plane.through(p1, p2, p1 + 2 * (p2 - p1))  # on one line


def test_line_meets_plane(rng):
    """A line and a plane meet in one point, unless the line runs parallel to the plane."""
    plane = Plane(rng.normal(size=3), rng.normal(size=3))
    start, end = rng.normal(size=(2, 3))
    (meeting,) = plane.intersect(Line(start, end))
    assert plane.contains(meeting, tolerance=1e-9)
    assert np.allclose(np.cross(meeting - start, end - start), 0)  # on the line
    sideways = np.cross(plane.normal, random_unit(rng))
    assert plane.intersect((start, start + sideways)).shape == (0, 3)


# ---------------------------------------------------------------- rotations


@pytest.mark.parametrize("angle", [0, 30, 90, 135, -60, 270])
def test_rotation_turns_x_toward_y(angle):
    """test_cheatsheet::test_outermorphism_rotation: R e1 ~R = cos(t) e1 + sin(t) e2 for
    R = cos(t/2) - sin(t/2) e1^e2: a positive angle turns counter-clockwise, seen from +z."""
    expected = np.cos(np.radians(angle)) * X + np.sin(np.radians(angle)) * Y
    for transform in (
        Transform().rotate("z", angle),
        Transform().rotate_z(angle),
        Transform().rotate(Z, angle),
        Transform().rotate_arbitrary(Z, angle),
        Transform().rotate(5 * Z, np.radians(angle), in_degrees=False),  # only the axis direction counts
    ):
        assert np.allclose(transform.apply_to_vertex(X), expected)
    # the same right-hand rule about x and y
    assert np.allclose(Transform().rotate("x", 90).apply_to_vertex(Y), Z)
    assert np.allclose(Transform().rotate("y", 90).apply_to_vertex(Z), X)


def test_rotations_are_proper_and_keep_products(rng, abcd):
    """test_cheatsheet::test_outermorphism_rotation: R ~R = 1 and det = 1; rotors are versors."""
    a, b, _, _ = abcd
    axis, angle = random_unit(rng), rng.uniform(-360, 360)
    rotation = Transform().rotate(axis, angle).matrix[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1)
    ra, rb = Vector(rotation @ a.components), Vector(rotation @ b.components)
    assert ra.dot(rb) == pytest.approx(a.dot(b))
    assert np.allclose(ra.cross(rb).components, rotation @ a.cross(b).components)  # keeps handedness
    assert np.allclose(rotation @ axis, axis)  # the axis stays


def test_rotation_follows_rodrigues(rng):
    """test_versors::test_bivector_exponential: exp(-B t/2) rotates by t in the plane of B."""
    axis, angle, v = random_unit(rng), rng.uniform(-180, 180), rng.normal(size=3)
    t = np.radians(angle)
    expected = v * np.cos(t) + np.cross(axis, v) * np.sin(t) + axis * (axis @ v) * (1 - np.cos(t))
    assert np.allclose(Transform().rotate(axis, angle).apply_to_vertex(v), expected)


def test_rotations_compose_and_undo(rng):
    """test_cheatsheet::test_adjoint: det(FG) = det(F) det(G); and rotations about one axis add up."""
    axis, alpha, beta = random_unit(rng), rng.uniform(-180, 180), rng.uniform(-180, 180)
    both = Transform().rotate(axis, alpha).rotate(axis, beta)
    assert np.allclose(both.matrix, Transform().rotate(axis, alpha + beta).matrix)
    assert np.allclose(Transform().rotate(axis, alpha).rotate(axis, -alpha).matrix, np.eye(4))
    assert np.allclose(Transform().rotate(axis, 360).matrix, np.eye(4))
    other = Transform().rotate(random_unit(rng), beta)
    composed = Transform().rotate(axis, alpha) @ other
    assert np.linalg.det(composed.matrix) == pytest.approx(1)


def test_reflections_through_minigeo_are_householder_mirrors(rng):
    """test_versors::test_basis_vectors_are_versors: a unit vector n mirrors x to -n x n."""
    normal = random_unit(rng)
    mirror = reflection(normal).matrix[:3, :3]
    assert np.allclose(mirror, np.eye(3) - 2 * np.outer(normal, normal))
    assert np.linalg.det(mirror) == pytest.approx(-1)


def test_two_reflections_make_a_rotation(rng):
    """test_versors::test_product_of_vectors: the product of two vectors is a rotor; mirroring
    in two planes turns about their common line by twice the angle between them."""
    n1, n2 = random_unit(rng), random_unit(rng)
    both = reflection(n1).matrix[:3, :3] @ reflection(n2).matrix[:3, :3]  # n2 first, then n1
    axis = np.cross(n2, n1)
    angle = Vector(n2).angle_to(Vector(n1))
    assert np.allclose(both, Transform().rotate(axis, 2 * angle).matrix[:3, :3])


def test_rotation_about_a_point_keeps_it(rng):
    """Rotating about p is moving p to the origin, rotating, and moving back."""
    pivot, axis, angle = rng.normal(size=3), random_unit(rng), rng.uniform(-180, 180)
    about = Transform().rotate(axis, angle, about=pivot)
    assert np.allclose(about.apply_to_vertex(pivot), pivot)
    manual = Transform().translate(-pivot).rotate(axis, angle).translate(pivot)
    assert np.allclose(about.matrix, manual.matrix)
    point = rng.normal(size=3)
    moved = about.apply_to_vertex(point)
    assert np.linalg.norm(moved - pivot) == pytest.approx(np.linalg.norm(point - pivot))


# ---------------------------------------------------------------- linear maps


def test_matrix_columns_are_images_of_the_axes():
    """test_lt::test_lt_matrix: Lt(a) = a + b, Lt(b) = 2a - b has the matrix [[1, 2], [1, -1]]."""
    transform = linear([[1, 2, 0], [1, -1, 0], [0, 0, 1]])
    assert np.allclose(transform.apply_to_vertex(X), [1, 1, 0])
    assert np.allclose(transform.apply_to_vertex(Y), [2, -1, 0])


def test_chained_steps_apply_in_order():
    """Chained methods apply left to right; t1 @ t2 applies t2 first."""
    move_then_turn = Transform().translate([1, 0, 0]).rotate("z", 90)
    assert np.allclose(move_then_turn.apply_to_vertex([0, 0, 0]), [0, 1, 0])
    turn, move = Transform().rotate("z", 90), Transform().translate([1, 0, 0])
    assert np.allclose((turn @ move).matrix, move_then_turn.matrix)
    assert np.allclose((move @ turn).apply_to_vertex([0, 0, 0]), [1, 0, 0])


def test_linear_maps_are_linear(rng):
    """test_lt::test_lt_function: a linear map keeps sums and multiples (affine: plus an offset)."""
    matrix, offset = random_matrix(rng), rng.normal(size=3)
    transform = linear(matrix, offset)
    x, y = rng.normal(size=(2, 3))
    alpha, beta = rng.normal(size=2)
    image = transform.apply_to_vertex
    assert np.allclose(
        image(alpha * x + beta * y) - offset, alpha * (image(x) - offset) + beta * (image(y) - offset)
    )
    assert np.allclose(transform.apply_to_vertices(np.array([x, y])), [image(x), image(y)])


@pytest.mark.parametrize("mirrored", [False, True])
def test_cross_products_transform_with_the_inverse_transpose(rng, mirrored):
    """test_lt::test_lt_function: f(x ^ y) = f(x) ^ f(y). For cross products (the duals) this
    reads (Mx) x (My) = det(M) M^-T (x x y): the reason normals transform differently."""
    matrix = random_matrix(rng, mirrored=mirrored)
    x, y = rng.normal(size=(2, 3))
    left = np.cross(matrix @ x, matrix @ y)
    right = np.linalg.det(matrix) * np.linalg.inv(matrix).T @ np.cross(x, y)
    assert np.allclose(left, right)


def test_reciprocal_frame(rng):
    """test_test::test_ReciprocalFrame: for a basis e_i, e^i = (e_j x e_k) / V gives e^i . e_j = delta_ij."""
    frame = random_matrix(rng).T  # rows e_1, e_2, e_3
    volume_123 = frame[0] @ np.cross(frame[1], frame[2])
    reciprocal = np.array([np.cross(frame[(i + 1) % 3], frame[(i + 2) % 3]) for i in range(3)]) / volume_123
    assert np.allclose(reciprocal @ frame.T, np.eye(3))


# ---------------------------------------------------------------- what minigeo promises


@pytest.mark.parametrize("mirrored", [False, True])
def test_transforms_move_every_vertex_exactly(rng, mirrored):
    """However a transform is split into pose and stretch, the world vertices follow it."""
    for shape in (
        Box([0.3, 0, 0], size=(1, 2, 3)),
        Cylinder([0, 1, 0], radius=0.5, height=2),
        Sphere([0, 0, 1], 1),
    ):
        before = shape.vertices
        transform = linear(random_matrix(rng, mirrored=mirrored), rng.normal(size=3))
        transform @ shape
        assert np.allclose(shape.vertices, transform.apply_to_vertices(before))


@pytest.mark.parametrize("mirrored", [False, True])
def test_axes_stay_perpendicular_and_right_handed(rng, mirrored):
    """The pose is a rotation: scaling, shear and mirroring go into the vertices."""
    box = Box([0, 0, 0], size=(1, 2, 3))
    linear(random_matrix(rng, mirrored=mirrored)) @ box
    box.rotate(random_unit(rng), 40, about="center")
    assert np.allclose(box.axes @ box.axes.T, np.eye(3))
    assert np.linalg.det(box.axes) == pytest.approx(1)


@pytest.mark.parametrize("mirrored", [False, True])
def test_faces_point_outward_after_any_transform(rng, mirrored):
    """Faces wind counter-clockwise from outside, so the enclosed volume is positive and
    scales with |det M|. Outward normals follow M^-T, also for mirrors: the cross product of
    the edges gains the sign of det M (see above), and the reversed winding cancels it."""
    box = Box([0, 0, 0], size=(1, 2, 3))
    normals_before = [face_normal(face) for face in box.faces]
    assert volume(box) == pytest.approx(6)
    matrix = random_matrix(rng, mirrored=mirrored)
    linear(matrix, rng.normal(size=3)) @ box
    assert volume(box) == pytest.approx(6 * abs(np.linalg.det(matrix)))
    for before, face in zip(normals_before, box.faces, strict=True):
        expected = np.linalg.inv(matrix).T @ before
        assert np.allclose(face_normal(face), expected / np.linalg.norm(expected))


def test_box_normals_are_the_reciprocal_frame_of_its_edges(rng):
    """The faces of a sheared box are perpendicular to the reciprocal frame of its edges."""
    box = Box([0, 0, 0], size=(1, 1, 1))
    matrix = random_matrix(rng)
    linear(matrix) @ box
    edges = matrix.T  # the images of the unit edges, as rows
    reciprocal = np.linalg.inv(edges).T  # rows e^1, e^2, e^3 with e^i . e_j = delta_ij
    directions = {tuple(np.round(r / np.linalg.norm(r), 9)) for r in np.vstack([reciprocal, -reciprocal])}
    normals = {tuple(np.round(face_normal(face), 9)) for face in box.faces}
    assert normals == directions


@pytest.mark.parametrize("segments", [3, 8, 16])
def test_closed_shapes_enclose_the_right_volume(segments):
    """Polyhedral cylinders and cones: an n-gon times the height (a third of it for cones)."""
    radius, height = 0.7, 1.3
    ngon = segments / 2 * radius**2 * np.sin(2 * np.pi / segments)
    assert volume(Cylinder([0, 0, 0], radius, height, segments=segments)) == pytest.approx(ngon * height)
    assert volume(Cone([0, 0, 0], radius, height, segments=segments)) == pytest.approx(ngon * height / 3)
    sphere = volume(Sphere([1, 2, 3], radius, segments=max(segments, 4)))
    assert 0 < sphere < 4 / 3 * np.pi * radius**3


def test_angles_are_kept_by_similarities(rng):
    """Angles between vectors survive rotation, translation and uniform scaling, not shear."""
    vertex, start, end = rng.normal(size=(3, 3))
    angle = Angle(vertex, start, end)
    assert angle.degrees == pytest.approx(Vector(start - vertex).angle_to(Vector(end - vertex)))
    similarity = Transform().scale([2.5, 2.5, 2.5]).rotate(random_unit(rng), 70).translate(rng.normal(size=3))
    moved = Angle(*(similarity.apply_to_vertex(p) for p in (vertex, start, end)))
    assert moved.degrees == pytest.approx(angle.degrees)
    assert Angle([0, 0, 0], [2, 0, 0], [0, 0, 3]).is_right


def test_distances_are_kept_by_rigid_motions(rng):
    """Dimension lines measure the Euclidean distance, which rotations and moves keep."""
    start, end = rng.normal(size=(2, 3))
    dimension = Dimension(start, end)
    assert dimension.length == pytest.approx(np.linalg.norm(end - start))
    motion = Transform().rotate(random_unit(rng), 123).translate(rng.normal(size=3))
    moved = Dimension(motion.apply_to_vertex(start), motion.apply_to_vertex(end))
    assert moved.length == pytest.approx(dimension.length)
