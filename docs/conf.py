"""Sphinx configuration: the API reference is built from the docstrings (Google style)."""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))  # the manual's example directive

project = "minigeo"
author = "Alexander Steppke"
copyright = "Alexander Steppke"
try:
    release = version("minigeo")
except PackageNotFoundError:
    release = "unknown"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style "Args:", "Returns:", "Example:" sections
    "sphinx.ext.viewcode",  # links to the source code
    "minigeo_manual",  # examples with their picture next to the code
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"show-inheritance": True}  # members are listed explicitly in api.rst
autoclass_content = "both"  # class docstring and __init__ docstring together
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True  # "Attributes:" sections as field lists, not as separate entries

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "minigeo"
html_static_path = ["_static"]
