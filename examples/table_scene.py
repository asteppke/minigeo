"""
A small room scene built entirely with anchors, in the spirit of TikZ.

Nothing here uses hand-computed coordinates: every object is placed relative to another
one through named points such as "top", "bottom front left" or "near end". The geometry
is built first; the scene only says how each piece looks.

Run it with ``python examples/table_scene.py`` (needs ``pip install ".[all]"``): it saves
the scene as ``table_scene.svg`` and ``table_scene.html`` (interactive, plotly) and shows it
with matplotlib. In a notebook, ``build_scene()`` on the last line of a cell shows the scene.
"""

import matplotlib.pyplot as plt

from minigeo import Box, Cone, Cylinder, Group, Line, Scene, Style

TABLE_SIZE = (2.0, 1.0, 0.1)  # width (x), depth (y), thickness (z)
LEG_HEIGHT = 1.0
LEG_SIZE = (0.08, 0.08, LEG_HEIGHT)
CORNERS = ("front left", "front right", "back left", "back right")

WOOD = Style(fill="peru", stroke="saddlebrown")


def build_scene() -> Scene:
    """Build the scene; every object is placed through anchors of another one."""
    scene = Scene()

    # The table top rests at leg height; its own "bottom" anchor sits on that point.
    table = scene.add(Box([0, 0, LEG_HEIGHT], size=TABLE_SIZE, anchor="bottom"), WOOD)

    # Each leg hangs from a bottom corner of the table, flush with the table's edges.
    for corner in CORNERS:
        leg = Box(table.anchor(f"bottom {corner}"), size=LEG_SIZE, anchor=f"top {corner}")
        scene.add(leg, WOOD, fill="sienna")

    # A cup stands near the front left corner of the tabletop (cylinders stand on their position).
    cup_spot = table.anchor("top front left") + [0.3, 0.3, 0]
    scene.add(Cylinder(cup_spot, radius=0.08, height=0.2), fill="C0", label="cup", label_at="top")

    # A book lies in the middle, turned a little around its own vertical axis.
    book = scene.add(Box([0, 0, 0], size=(0.5, 0.35, 0.06)), fill="C3")
    book.rotate("z'", 20, about="center")
    book.place("bottom", at=table.anchor("top"))

    # A lamp is a group: a stand with a shade on top. The group is placed as a whole.
    stand = scene.add(Cylinder([0, 0, 0], radius=0.03, height=0.45), fill="dimgray")
    shade = scene.add(Cone(stand.anchor("top"), radius=0.2, height=0.25, top_radius=0.08), fill="gold")
    lamp = Group([stand, shade])
    lamp.place("bottom", at=table.anchor("top back right") - [0.3, 0.25, 0])

    # A dashed line from the lamp shade to the book, labelled near its end.
    beam = Line(shade.anchor("bottom"), book.anchor("top"))
    scene.add(
        beam, stroke="orange", dash="dashed", label="light", label_at="near end", text_color="darkorange"
    )

    return scene


if __name__ == "__main__":
    scene = build_scene()
    scene.render("svg").save("table_scene.svg")
    scene.render("plotly").write_html("table_scene.html")
    scene.render("matplotlib")
    plt.show()
