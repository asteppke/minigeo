"""
A serial XYZ stage built with frames and joints.

The X carriage slides along the base and carries the Y carriage, which carries the Z column
with its slider and tool. Each stage is a joint (a :class:`~minigeo.Frame`); every part is
attached to the joint that carries it, so setting a joint's ``value`` moves the part and
everything after it in the chain. The parts themselves are built in world coordinates with
anchors, exactly as without joints.

Run it with ``python examples/xyz_stage.py`` (needs ``pip install ".[all]"``): it prints where
the tool tip is, saves ``xyz_stage.svg`` and an animation ``xyz_stage.html`` (plotly), and
shows the stage with matplotlib.
"""

from dataclasses import dataclass

import numpy as np

from minigeo import Box, Cone, Frame, Scene, Style

METAL = Style(fill="lightsteelblue", stroke="slategray")


@dataclass
class Stage:
    """The scene and the joints that move it."""

    scene: Scene
    base: Frame  # stands in the world; turn or move it to move the whole machine
    x: Frame
    y: Frame
    z: Frame
    tip: Frame  # a fixed frame at the tool tip, carried by z


def build_stage() -> Stage:
    """Build the XYZ stage with all joints at 0."""
    scene = Scene()

    # The chain: X carries Y, Y carries Z. Limits keep each carriage on its rail.
    base = Frame()
    x = base.prismatic("x", limits=(0, 0.8), name="X")
    y = x.prismatic("y", limits=(0, 0.6), name="Y")
    z = y.prismatic("z", limits=(-0.25, 0), name="Z")

    # The plate and rails belong to the base frame.
    plate = scene.add(base.attach(Box([0, 0, 0], size=(1.2, 1.0, 0.06), anchor="top")), fill="dimgray")
    for side in ("front", "back"):
        rail = Box(plate.anchor(f"top {side} left"), size=(1.2, 0.05, 0.03), anchor=f"bottom {side} left")
        scene.add(base.attach(rail), METAL)

    # X carriage: at value 0 it sits at the left end of the rails.
    x_carriage = Box(plate.anchor("top left") + [0, 0, 0.03], size=(0.3, 1.0, 0.04), anchor="bottom left")
    scene.add(x.attach(x_carriage), fill="C0", label="X", label_at="top front right")

    # Y carriage: at value 0 it sits at the front of the X carriage.
    y_carriage = Box(x_carriage.anchor("top front"), size=(0.3, 0.3, 0.04), anchor="bottom front")
    scene.add(y.attach(y_carriage), fill="C1", label="Y", label_at="top front right")

    # The Z column stands on the Y carriage and moves with it (attached to y, not z).
    column = Box(y_carriage.anchor("top back"), size=(0.3, 0.06, 0.5), anchor="bottom back")
    scene.add(y.attach(column), METAL)

    # Z slider: at value 0 it is at the top of the column's front face; it moves down from there.
    slider = Box(column.anchor("top front"), size=(0.2, 0.08, 0.12), anchor="top back")
    scene.add(z.attach(slider), fill="C2", label="Z", label_at="front")

    # The tool hangs from the slider, tip down.
    tool = Cone([0, 0, 0], radius=0.04, height=0.12)
    tool.rotate("x", 180, about="center")  # now the cone's own "bottom" (its base) faces up
    tool.place("bottom", at=slider.anchor("bottom"))
    scene.add(z.attach(tool), fill="C3")

    tip = z.fixed(at=tool.anchor("top"), name="tip")  # the cone's own "top" is the tip, now at the bottom
    scene.add_frame(tip, length=0.15, labels=("", "", ""))
    return Stage(scene, base, x, y, z, tip)


def trace(stage: Stage, frames: int = 60):
    """An update function for ``scene.animate``: the tip draws a circle and dips twice per turn."""

    def update(i: int) -> None:
        turn = 2 * np.pi * i / frames
        stage.x.value = 0.25 + 0.2 * np.cos(turn)  # absolute values: no drift, no bookkeeping
        stage.y.value = 0.3 + 0.2 * np.sin(turn)
        stage.z.value = -0.05 * (1 - np.cos(2 * turn))

    return update


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    stage = build_stage()
    stage.x.value, stage.y.value, stage.z.value = 0.4, 0.2, -0.1
    print("tool tip at", np.round(stage.tip.origin, 3))
    stage.scene.save("xyz_stage.svg")

    animation_stage = build_stage()
    figure = animation_stage.scene.animate(trace(animation_stage), frames=60)
    figure.write_html("xyz_stage.html")

    stage.scene.render(backend="matplotlib")
    plt.show()
