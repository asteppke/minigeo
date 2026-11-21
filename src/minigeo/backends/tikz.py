r"""
Draw a scene as TikZ code, for LaTeX documents.

The scene is projected, sorted and shaded exactly as for the SVG backend
(:func:`minigeo.flat.flatten`), and written as plain 2D TikZ paths: the code needs no 3D
library, compiles quickly and can be edited by hand. Faces are shaded by mixing their color
with black (``mgperu!72!black``).

Sizes follow the SVG backend: the picture looks like the SVG image shrunk to ``width``, so
lines, arrow tips and points get thinner in a narrower picture. Text is the exception: labels
are set in ``font_size`` points, like the text around the picture.

Labels are LaTeX: text between ``$`` signs is math (``"$2\theta$"``). Outside math, the
characters ``# $ % & _ { } ~ ^ \`` are printed as they are, and a few common symbols
(′ ° × · and Greek letters) are turned into math, so labels such as ``"x′"`` work with
pdflatex too.

The code needs ``\usepackage{tikz}`` and ``\usetikzlibrary{arrows.meta}``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path as FilePath

import numpy as np

from ..camera import Camera
from ..colors import to_rgb
from ..flat import Placed, flatten
from ..render import Face, Label, Marker, Path
from ..style import Style

#: Dash and gap lengths, in multiples of the line width (as in the SVG backend).
DASH_PATTERNS = {"solid": None, "dashed": (6.0, 4.0), "dotted": (1.5, 3.0)}

#: Sizes in the scene (line widths, point sizes) are given for an image this many pixels wide.
REFERENCE_WIDTH = 500

POINTS_PER_CM = 72.27 / 2.54

PREAMBLE = r"\usepackage{tikz}" + "\n" + r"\usetikzlibrary{arrows.meta}"

GREEK = [
    *("alpha", "beta", "gamma", "delta", "varepsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda"),
    *("mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "varphi", "chi", "psi", "omega"),
    *("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega"),
]

#: Symbols that pdflatex cannot print directly, as math.
SYMBOLS = {
    "′": "$'$",
    "″": "$''$",
    "°": r"$^\circ$",
    "×": r"$\times$",
    "·": r"$\cdot$",
    "±": r"$\pm$",
    "−": "$-$",
    "Å": r"\AA{}",
    "µ": r"$\mu$",
    **{
        letter: f"$\\{name}$" for letter, name in zip("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ", GREEK, strict=True)
    },
}

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass
class TikzPicture:
    """
    The result of the TikZ backend.

    Attributes:
        code: The ``tikzpicture`` environment, ready for ``\\input`` or pasting into a document.
    """

    code: str

    def __str__(self) -> str:
        return self.code

    def _repr_markdown_(self) -> str:  # shown as code in Jupyter
        return f"```latex\n{self.code}\n```"

    def document(self) -> str:
        """A complete LaTeX document (class ``standalone``) that contains only the picture."""
        return (
            "\\documentclass[tikz, border=2pt]{standalone}\n"
            "\\usetikzlibrary{arrows.meta}\n"
            "\\begin{document}\n"
            f"{self.code}\n"
            "\\end{document}\n"
        )

    def save(self, path: str | FilePath, *, standalone: bool = False, engine: str = "pdflatex") -> None:
        """
        Write the picture to a file.

        * ``.tex``: the ``tikzpicture`` (for ``\\input``), or with ``standalone=True`` a
          complete document.
        * ``.pdf``: compiles the complete document with ``engine`` (``pdflatex``,
          ``lualatex`` or ``xelatex``), which must be installed.
        """
        path = FilePath(path)
        if path.suffix.lower() == ".pdf":
            path.write_bytes(self.to_pdf(engine))
        else:
            path.write_text(self.document() if standalone else self.code + "\n", encoding="utf-8")

    def to_pdf(self, engine: str = "pdflatex") -> bytes:
        """Compile the picture as a standalone document; return the PDF file's contents."""
        if shutil.which(engine) is None:
            raise RuntimeError(f"{engine} is not installed; save as .tex and compile it yourself.")
        with tempfile.TemporaryDirectory() as folder:
            source = FilePath(folder) / "picture.tex"
            source.write_text(self.document(), encoding="utf-8")
            run = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", source.name],
                cwd=folder,
                capture_output=True,
                text=True,
                errors="replace",
            )
            pdf = source.with_suffix(".pdf")
            if run.returncode != 0 or not pdf.exists():
                errors = [line for line in run.stdout.splitlines() if line.startswith("!")]
                raise RuntimeError(f"{engine} failed: {' '.join(errors) or run.stdout[-500:]}")
            return pdf.read_bytes()


def render(
    scene,
    *,
    width: float = 8.0,
    elev: float = 30.0,
    azim: float = -60.0,
    vertical_axis: str = "z",
    shading: bool = True,
    digits: int = 3,
) -> TikzPicture:
    r"""
    Draw ``scene`` as a ``tikzpicture``.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        width: Width of the drawing in centimetres; the height follows from the scene's shape.
        elev, azim, vertical_axis: Where the viewer looks from, as in the other backends.
        shading: Darken faces that turn away from the light, for a 3D impression.
        digits: Decimals of the coordinates (in cm).

    Example:
        >>> picture = scene.render("tikz", width=6)
        >>> picture.save("scene.tex")   # then \input{scene.tex} in the document
    """
    camera = Camera(elev, azim, vertical_axis)
    picture = flatten(scene.primitives(), camera, width=width, margin=0, y_down=False, shading=shading)
    writer = _Writer(points_per_unit=width * POINTS_PER_CM / REFERENCE_WIDTH, digits=digits)
    body = [writer.write(item) for item in picture.items]
    lines = [
        "% made with minigeo; needs \\usepackage{tikz} and \\usetikzlibrary{arrows.meta}",
        "\\begin{tikzpicture}[line join=round, line cap=round]",
        *writer.color_definitions(),
        *[line for line in body if line],
        "\\end{tikzpicture}",
    ]
    return TikzPicture("\n".join(lines))


def latex_text(text: str) -> str:
    r"""
    Prepare a label for LaTeX: math between ``$`` signs stays as it is; elsewhere special
    characters are escaped and some symbols become math.

    Example:
        >>> latex_text("x′ = 50% of $2\theta$")
        "x$'$ = 50\\% of $2\\theta$"
    """
    parts = re.split(r"(\$[^$]*\$)", text)
    result = []
    for part in parts:
        if len(part) >= 2 and part.startswith("$") and part.endswith("$"):
            result.append(part)
        else:
            result.append("".join(_SPECIAL.get(c, SYMBOLS.get(c, c)) for c in part))
    return "".join(result)


class _Writer:
    """Writes primitives as TikZ commands and collects the colors they use."""

    def __init__(self, points_per_unit: float, digits: int):
        self.points_per_unit = points_per_unit  # TikZ points per pixel of the reference image
        self.digits = digits
        self.colors: dict[str, str] = {}  # name -> RGB values

    def write(self, item: Placed) -> str:
        primitive = item.primitive
        if isinstance(primitive, Face):
            return self.face(primitive, item.points, item.darkness)
        if isinstance(primitive, Path):
            return self.path(primitive, item.points)
        if isinstance(primitive, Marker):
            return self.marker(primitive, item.points[0])
        if isinstance(primitive, Label):
            return self.label(primitive, item.points[0])
        raise TypeError(f"Cannot draw {type(primitive).__name__}.")

    # ------------------------------------------------------------ primitives

    def face(self, face: Face, corners: np.ndarray, darkness: float) -> str:
        style = face.style
        outline = " -- ".join(self.point(p) for p in corners) + " -- cycle"
        commands = []
        filled = style.fill != "none" and style.opacity > 0
        if filled:
            fill = self.color(style.fill)
            if darkness > 0.005:
                fill = f"{fill}!{round(100 * (1 - darkness))}!black"
            options = [f"fill={fill}"]
            if style.opacity < 1:
                options.append(f"fill opacity={self.number(style.opacity)}")
            if face.fully_outlined and style.stroke != "none":
                options += self.stroke(style)
                commands.append(f"\\filldraw[{', '.join(options)}] {outline};")
            elif style.opacity == 1:  # a hairline in the fill color covers seams between faces
                options += [f"draw={fill}", "line width=0.2pt"]
                commands.append(f"\\filldraw[{', '.join(options)}] {outline};")
            else:
                commands.append(f"\\fill[{', '.join(options)}] {outline};")
        if style.stroke != "none" and (face.outline and not (filled and face.fully_outlined)):
            if face.fully_outlined:
                commands.append(f"\\draw[{', '.join(self.stroke(style))}] {outline};")
            else:
                segments = " ".join(
                    f"{self.point(corners[i])} -- {self.point(corners[j])}" for i, j in face.outline
                )
                commands.append(f"\\draw[{', '.join(self.stroke(style))}] {segments};")
        return "\n".join(commands)

    def path(self, path: Path, points: np.ndarray) -> str:
        style = path.style
        if style.stroke == "none":
            return ""
        options = self.stroke(style)
        if style.arrow:
            size = 6 * style.line_width * self.points_per_unit
            options.append(f"-{{Stealth[length={self.number(size)}pt, width={self.number(size)}pt]}}")
        return f"\\draw[{', '.join(options)}] {' -- '.join(self.point(p) for p in points)};"

    def marker(self, marker: Marker, position: np.ndarray) -> str:
        style = marker.style
        if style.fill == "none":
            return ""
        radius = np.sqrt(style.marker_size) / 2 * self.points_per_unit
        options = [self.color(style.fill)]
        if style.opacity < 1:
            options.append(f"opacity={self.number(style.opacity)}")
        return f"\\fill[{', '.join(options)}] {self.point(position)} circle[radius={self.number(radius)}pt];"

    def label(self, label: Label, position: np.ndarray) -> str:
        if not label.text:
            return ""
        style = label.style
        size = self.number(style.font_size)
        anchor = "center" if label.align == "center" else "south west"
        options = [
            f"anchor={anchor}",
            "inner sep=1.5pt",
            f"text={self.color(style.text_color)}",
            f"font=\\fontsize{{{size}}}{{{self.number(1.2 * style.font_size)}}}\\selectfont",
        ]
        return f"\\node[{', '.join(options)}] at {self.point(position)} {{{latex_text(label.text)}}};"

    # ------------------------------------------------------------ helpers

    def stroke(self, style: Style) -> list[str]:
        width = style.line_width * self.points_per_unit
        options = [f"draw={self.color(style.stroke)}", f"line width={self.number(width)}pt"]
        if style.opacity < 1:
            options.append(f"draw opacity={self.number(style.opacity)}")
        dashes = DASH_PATTERNS[style.dash]
        if dashes:
            on, off = (self.number(d * width) for d in dashes)
            options.append(f"dash pattern=on {on}pt off {off}pt")
        return options

    def color(self, color: str) -> str:
        """The name of an xcolor color for the given color, defined once per picture."""
        name = "mg" + re.sub(r"[^A-Za-z0-9]", "", color)
        if name not in self.colors:
            red, green, blue = (round(255 * c) for c in to_rgb(color))
            self.colors[name] = f"{red},{green},{blue}"
        return name

    def color_definitions(self) -> list[str]:
        return [f"\\definecolor{{{name}}}{{RGB}}{{{rgb}}}" for name, rgb in self.colors.items()]

    def number(self, value: float) -> str:
        text = f"{value:.{self.digits}f}".rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text

    def point(self, point: np.ndarray) -> str:
        return f"({self.number(point[0])},{self.number(point[1])})"
