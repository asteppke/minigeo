"""
A small room scene built entirely with anchors, in the spirit of TikZ.

Nothing here uses hand-computed coordinates: every object is placed relative to another
one through named points such as "top", "bottom front left" or "near end". The geometry
is built first; the scene only says how each piece looks.

Run it with ``python examples/table_scene.py`` (needs ``pip install ".[mpl]"``),
or copy ``build_scene`` into a notebook and show the returned scene.
"""

import matplotlib.pyplot as plt

from minigeo import Scene, Style
from minigeo.geometry import BaseBox, BaseCone, BaseCylinder, BaseLine
from minigeo.groups import GeometryGroup

TABLE_SIZE = (2.0, 1.0, 0.1)  # width (x), depth (y), thickness (z)
LEG_HEIGHT = 1.0
LEG_SIZE = (0.08, 0.08, LEG_HEIGHT)
CORNERS = ("front left", "front right", "back left", "back right")

WOOD = Style(fill="peru", stroke="saddlebrown")


def build_scene() -> Scene:
    """Build the scene; every object is placed through anchors of another one."""
    scene = Scene()

    # The table top rests at leg height; its own "bottom" anchor sits on that point.
    table = scene.add(BaseBox([0, 0, LEG_HEIGHT], TABLE_SIZE, anchor="bottom"), WOOD)

    # Each leg hangs from a bottom corner of the table, flush with the table's edges.
    for corner in CORNERS:
        leg = BaseBox(table.anchor(f"bottom {corner}"), LEG_SIZE, anchor=f"top {corner}")
        scene.add(leg, WOOD, fill="sienna")

    # A cup stands near the front left corner of the tabletop (cylinders stand on their position).
    cup_spot = table.anchor("top front left") + [0.3, 0.3, 0]
    scene.add(BaseCylinder(cup_spot, (0.08, 0.2)), fill="C0", label="cup", label_at="top")

    # A book lies in the middle, turned a little around its own vertical axis.
    book = scene.add(BaseBox([0, 0, 0], (0.5, 0.35, 0.06)), fill="C3")
    book.rotate("z'", 20, about="center")
    book.place("bottom", at=table.anchor("top"))

    # A lamp is a group: a stand with a shade on top. The group is placed as a whole.
    stand = scene.add(BaseCylinder([0, 0, 0], (0.03, 0.45)), fill="dimgray")
    shade = scene.add(BaseCone(stand.anchor("top"), (0.2, 0.08, 0.25)), fill="gold")
    lamp = GeometryGroup([stand, shade])
    lamp.place("bottom", at=table.anchor("top back right") - [0.3, 0.25, 0])

    # A dashed line from the lamp shade to the book, labelled near its end.
    beam = BaseLine(shade.anchor("bottom"), book.anchor("top"))
    scene.add(beam, stroke="orange", dash="dashed", label="light", label_at="near end", text_color="darkorange")

    return scene


if __name__ == "__main__":
    build_scene().render()
    plt.show()
