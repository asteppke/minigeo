"""Coordinate axes with ticks, and grids."""

import numpy as np
import pytest

from minigeo import Axes, Grid, Scene, Transform
from minigeo.geometry.axes import tick_values
from minigeo.render import Label, Path, to_primitives, type_style
from minigeo.style import DEFAULT_STYLE


def drawn(geometry):
    return to_primitives(geometry, DEFAULT_STYLE | type_style(geometry))


def test_tick_values_are_multiples_of_the_step():
    assert np.allclose(tick_values(-1, 2, 0.5), [-1, -0.5, 0, 0.5, 1, 1.5, 2])
    assert np.allclose(tick_values(0.3, 1.2, 0.5), [0.5, 1])
    assert np.allclose(tick_values(0, 0.3, 0.1), [0, 0.1, 0.2, 0.3])  # no rounding loss at the end


def test_axes_cross_at_the_origin_and_label_their_ticks():
    axes = Axes(low=[-1, -1, 0], high=[2, 2, 1], step=1)
    assert np.allclose(axes.axis_ends(0), [[-1, 0, 0], [2, 0, 0]])
    assert np.allclose(axes.axis_ends(2), [[0, 0, 0], [0, 0, 1]])
    texts = [text for axis, _, text in axes.ticks if axis == 0]
    assert texts == ["-1", "1", "2"]  # none at the crossing
    labels = [p.text for p in drawn(axes) if isinstance(p, Label)]
    assert {"x", "y", "z", "-1", "1", "2"} <= set(labels)


def test_crossing_outside_the_range_moves_into_it():
    axes = Axes(low=[1, 1, 1], high=[3, 2, 2])
    assert np.allclose(axes.vertices[6], [1, 1, 1])


def test_axes_with_equal_ends_are_left_out():
    axes = Axes(high=[4, 3, 0], step=1)
    assert axes.present == [0, 1]
    arrows = [p for p in drawn(axes) if isinstance(p, Path) and p.style.arrow]
    assert len(arrows) == 2
    assert np.allclose(axes.tick_direction(0), [0, 1, 0]) and np.allclose(axes.tick_direction(1), [1, 0, 0])


def test_ticks_and_numbers_sit_beside_the_axis():
    axes = Axes(high=[2, 1, 1], step=1, tick_length=0.1)
    ticks = [p for p in drawn(axes) if isinstance(p, Path) and not p.style.arrow]
    assert any(np.allclose(t.points, [[1, -0.05, 0], [1, 0.05, 0]]) for t in ticks)
    numbers = {p.text: p.position for p in drawn(axes) if isinstance(p, Label) and p.align == "center"}
    assert np.allclose(numbers["2"], [2, -0.25, 0])


def test_axes_turn_with_their_ticks():
    axes = Axes(high=[1, 1, 1], tick_length=0.1)
    Transform().rotate("z", 90) @ axes
    assert np.allclose(axes.axis_ends(0)[1], [0, 1, 0])
    assert np.allclose(axes.tick_direction(0), [-1, 0, 0])


def test_axes_checks():
    with pytest.raises(ValueError, match="low <= high"):
        Axes(low=[1, 0, 0], high=[0, 1, 1])
    with pytest.raises(ValueError, match="positive"):
        Axes(step=0)


def test_flat_grids_have_lines_at_multiples_of_the_step():
    grid = Grid.flat(low=(-1, -1), high=(1, 2), step=1)
    assert len(grid.lines) == 3 + 4
    assert any(np.allclose(line, [[0, -1, 0], [0, 2, 0]]) for line in grid.lines)
    offset = Grid.flat(low=(-1.25, 0), high=(1, 1), step=0.5, plane="xz", at=3)
    xs = sorted({round(float(line[0, 0]), 9) for line in offset.lines if np.isclose(line[0, 0], line[1, 0])})
    assert xs == [-1.25, -1, -0.5, 0, 0.5, 1]  # border, then multiples of the step
    assert np.allclose([line[:, 1] for line in offset.lines], 3)


def test_grid_of_axes_uses_their_ticks_but_not_their_lines():
    axes = Axes(low=[-1, -1, 0], high=[2, 1, 1], step=1)
    grid = axes.grid("xy")
    xs = sorted(float(line[0, 0]) for line in grid.lines if np.isclose(line[0, 0], line[1, 0]))
    assert xs == [-1, 1, 2]  # x = 0 is the y axis itself
    assert all(np.allclose(line[:, 2], 0) for line in grid.lines)
    with pytest.raises(ValueError, match="Unknown plane"):
        axes.grid("xw")


def test_axes_and_grids_draw_everywhere_and_are_not_picked():
    from minigeo.measure import pickable_points

    scene = Scene()
    axes = scene.add(Axes(high=[2, 1, 1], step=0.5))
    scene.add(axes.grid("xy"))
    assert "1.5" in str(scene.render("svg")) and "1.5" in scene.render("tikz").code
    assert pickable_points(scene) == []
    pytest.importorskip("PIL")
    assert scene.render("raytrace", width=60).width == 60


def test_grids_are_drawn_cell_by_cell():
    grid = Grid.flat(low=(0, 0), high=(2, 1), step=1)
    assert len(grid.lines) == 5 and len(grid.pieces) == 3 + 4
    assert sum(np.linalg.norm(p[1] - p[0]) for p in grid.pieces) == pytest.approx(3 * 1 + 2 * 2)
    axes_grid = Axes(low=[-1, -1, 0], high=[2, 1, 1]).grid("xy")  # no border lines
    total = sum(np.linalg.norm(p[1] - p[0]) for p in axes_grid.pieces)
    assert total == pytest.approx(sum(np.linalg.norm(line[1] - line[0]) for line in axes_grid.lines))
