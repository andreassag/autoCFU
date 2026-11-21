"""Sphinx documentation configuration for AutoCFU."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

project = "AutoCFU"
copyright = "2024-2026, Andreas Sagen"
author = "Andreas Sagen"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autodoc_mock_imports: list[str] = ["onnxruntime"]
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
always_use_bars_union = True
typehints_defaults = "comma"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "AutoCFU Documentation"
