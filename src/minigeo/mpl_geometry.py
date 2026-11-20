from typing import Protocol

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import BaseBox, BaseCone, BaseCylinder, BaseLine, BasePolygon, BaseRectangle
from .groups import GeometryGroup
from .mpl_3d_arrow import Arrow3D


class MplDrawable(Protocol):
    """
    Protocol for drawable objects using Matplotlib.
    """

    def _draw(self) -> None: ...


class MplPoly(BasePolygon):
    def __init__(self, ax, center, vertices, color="C0", alpha=1, draw=True):
        super().__init__(center, vertices=vertices)

        self.ax = ax
        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(self.faces, facecolors=self.color, edgecolors="grey", alpha=self.alpha)
        self.ax.add_collection3d(self.poly)

    def update_geometry(self):
        self.poly.set_verts(self.faces)


class MplSurface(BaseRectangle):
    def __init__(self, ax, center, vertices, color="C0", alpha=1, draw=True):
        super().__init__(center, vertices=vertices)

        self.ax = ax
        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(self.faces, facecolors=self.color, edgecolors="grey", alpha=self.alpha)
        self.ax.add_collection3d(self.poly)

    def update_geometry(self):
        self.poly.set_verts(self.faces)


class MplBox(BaseBox):
    def __init__(self, ax, center, dimensions, color="C0", alpha=1, draw=True):
        super().__init__(center, dimensions)

        self.ax = ax

        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(self.faces, facecolors=self.color, edgecolors="grey", alpha=self.alpha)
        self.ax.add_collection3d(self.poly)

    def update_geometry(self):
        self.poly.set_verts(self.faces)


class MplCube(MplBox):
    def __init__(
        self,
        ax,
        center,
        size,
        face_colors=("red", "red", "blue", "blue", "orange", "orange"),
        alpha=0.75,
    ):
        super().__init__(ax, center, (size, size, size), color="white", alpha=alpha, draw=False)
        self.face_colors = list(face_colors)
        self.edge_colors = ["gray"] * len(face_colors)
        self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(
            self.faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=False,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly)

    def add_face_labels(self):
        # Add text labels to cube faces
        face_labels = ["Front", "ab", "Bottom", "ac", "Left", "bc"]
        for i, face in enumerate(self.faces):
            if face_labels[i] in ["Front", "Bottom", "Left"]:
                continue
            center = np.mean(face, axis=0)
            self.ax.text(
                center[0], center[1], center[2], face_labels[i], color="black", ha="center", va="center"
            )


class MplLine(BaseLine):
    def __init__(self, ax, start, end, color="C0", alpha=1, draw=True, arrow_tip=False):
        super().__init__(start, end)

        self.ax = ax
        self.color = color
        self.alpha = alpha
        self.arrow_tip = arrow_tip

        if draw:
            self._draw()

    def _draw(self) -> None:
        # Draw arrow tip if specified
        if self.arrow_tip:
            self.arrow = Arrow3D(
                [self.start[0], self.end[0]],
                [self.start[1], self.end[1]],
                [self.start[2], self.end[2]],
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color=self.color,
            )
            self.ax.add_artist(self.arrow)
        else:
            (self.line,) = self.ax.plot(
                [self.start[0], self.end[0]],
                [self.start[1], self.end[1]],
                [self.start[2], self.end[2]],
                color=self.color,
                alpha=self.alpha,
            )

    def update_geometry(self):
        if self.arrow_tip:
            self.arrow._verts3d = (
                [self.start[0], self.end[0]],
                [self.start[1], self.end[1]],
                [self.start[2], self.end[2]],
            )
            self.arrow.stale = True  # similar to Line3D.set_data_3d
        else:
            self.line.set_data_3d(
                [self.start[0], self.end[0]], [self.start[1], self.end[1]], [self.start[2], self.end[2]]
            )


class MplCylinder(BaseCylinder):
    def __init__(self, ax, center, dimensions: tuple, color="C0", alpha=1, draw=True):
        radius, height = dimensions

        super().__init__(center, (radius, height))

        self.ax = ax
        self.color = color
        self.face_colors = color
        self.edge_colors = "grey"
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(
            self.side_faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=True,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly)

        self.poly2 = Poly3DCollection(
            self.top_faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=True,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly2)

        self.poly3 = Poly3DCollection(
            self.bottom_faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=True,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly3)

    def update_geometry(self):
        self.poly.set_verts(self.side_faces)
        self.poly2.set_verts(self.top_faces)
        self.poly3.set_verts(self.bottom_faces)


class MplCone(BaseCone, MplCylinder):
    def __init__(self, ax, center, lower_radius, upper_radius, height, color="C0", alpha=1, draw=True):
        BaseCone.__init__(self, center, (lower_radius, upper_radius, height))

        self.ax = ax
        self.color = color
        self.face_colors = color
        self.edge_colors = "grey"
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()


class MplGroup(GeometryGroup):
    def __init__(self, ax, geometries: list[MplDrawable] | None = None, draw: bool = False):
        """
        Groups Mpl geometries together and keeps their relative positions.

        Members are usually drawn already when they are created, so the group does not
        draw them again by default (that would create duplicate, stale artists).
        Pass ``draw=True`` only if the members were created with ``draw=False``.
        """

        super().__init__(geometries)
        # TODO: think if we need to keep a reference to the ax
        # this is at the moment only to have a similar interface
        self.ax = ax
        if draw:
            self._draw()

    def _draw(self):
        # Call each geometry’s draw method
        for geom in self.geometries:
            if hasattr(geom, "_draw"):
                geom._draw()
