"""Sphinx configuration for OpenSWMM Gymnasium."""

from __future__ import annotations

import os as _os
import sys
from pathlib import Path

# Make the package importable for autodoc without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


project = "OpenSWMM Gymnasium"
copyright = "2026, Caleb Buahin"  # noqa: A001
author = "Caleb Buahin"
version = "0.1.0"
release = "0.1.0.dev0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",  # ```mermaid``` architecture/workflow diagrams
    "sphinx_epytext",  # epytext docstring rendering — plan §13
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "IMPLEMENTATION_PLAN.md",  # internal planning doc; not part of public docs
    "api/_autogen/**",         # stale autosummary output; ignore if present
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/HydroCouple/openswmm.gymnasium",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "use_edit_page_button": False,
    "logo": {"text": "OpenSWMM Gymnasium"},
}
html_title = f"{project} {release}"

# Project icon — used as the sidebar logo and the browser favicon.
html_static_path = ["_static"]
html_logo = "_static/hydrocouplecomposer.png"
html_favicon = "_static/hydrocouplecomposer.png"

# ---------------------------------------------------------------------------
# Autodoc / autosummary
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = False  # Use manual automodule pages; see api/*.rst
autodoc_typehints = "description"

# Don't crash docs build if optional deps are missing.
autodoc_mock_imports = [
    "openswmm",
    "gymnasium",
    "plotly",
    "platypus",
    "mo_gymnasium",
]

# ---------------------------------------------------------------------------
# MyST
# ---------------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 3

# Render ```mermaid fenced code blocks (the GitHub/PyPI-native syntax) through
# the sphinxcontrib-mermaid directive, so the README's diagrams render the same
# everywhere without a Sphinx-only ``.. mermaid::`` variant.
myst_fence_as_directive = ["mermaid"]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

if _os.environ.get("SPHINX_OFFLINE"):
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "gymnasium": ("https://gymnasium.farama.org/", None),
    }

# Suppress intersphinx fetch failures in offline mode.
intersphinx_disabled_reftypes = []

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

# Tighten the warning set used by sphinx-build -W:
#
# - "ref.python" / "myst.header" — autodoc-mocked deps emit spurious cross-refs
#   for symbols Sphinx can't resolve (gymnasium.*, openswmm.*).
# - "autosectionlabel.*" — many MD files share heading names.
# - "duplicate" — autodoc renders class once for the manual API page and once
#   via autosummary recursion; both are valid.
# - "docutils" — sphinx-epytext's field-list conversion sometimes emits
#   "list ends without blank line" warnings on perfectly valid epytext.
# - "epytext" — same category for the epytext extension itself.
suppress_warnings = [
    "ref.python",
    "myst.header",
    "autosectionlabel.*",
    "duplicate",
    "docutils",
    "epytext",
]
nitpicky = False

# Intersphinx is configured above; offline mode swap handled there.
