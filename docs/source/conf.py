import os
import sys

import quadrants as qd

# Make the option-schema Sphinx extension (tools/config_codegen/sphinx_ext.py)
# importable so the qd-config-options directive can render the qd.init options
# from the same schema that generates the C++ defaults.
sys.path.insert(0, os.path.abspath("../../tools/config_codegen"))

__version__ = ".".join([str(v) for v in qd.__version__])
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Quadrants"
copyright = "2025 Genesis AI Inc"
author = ""
release = __version__
version = __version__

autoapi_dirs = ["../../python/quadrants"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_python_use_implicit_namespaces = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_subfigure",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx_design",
    "autoapi.extension",
    "sphinx_ext",
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
# https://github.com/executablebooks/MyST-Parser/issues/519#issuecomment-1037239655
myst_heading_anchors = 4

templates_path = ["_templates"]
# exclude_patterns = ["user_guide/reference/_autosummary/*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 1,
}
html_css_files = [
    "css/custom.css",
]
html_static_path = ["_static"]

### Autodoc configurations ###
autodoc_typehints = "signature"
autodoc_typehints_description_target = "all"
autodoc_default_flags = ["members", "show-inheritance", "undoc-members"]
autodoc_member_order = "bysource"
autosummary_generate = True
