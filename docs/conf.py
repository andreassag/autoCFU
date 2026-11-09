"""Sphinx configuration for AutoCFU documentation."""

from __future__ import annotations

import os
import sys

# Ensure src/ directory and repo root are on sys.path for autodoc
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

# Gracefully handle pure-python fallback for doc builds in environments with C-extension ABI mismatches
try:
    import numpy  # noqa: F401
except ImportError:
    import tests  # noqa: F401

project = "autocfu"
author = "Andreas Sagen"
copyright = "2024-2026, Andreas Sagen"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx mapping (disabled by default for offline/sandboxed build reliability)
intersphinx_mapping: dict[str, tuple[str, None]] = {}

# HTML output configuration
html_theme = "furo"
html_title = "AutoCFU Documentation"
html_static_path: list[str] = []
