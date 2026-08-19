import quadrants as qd

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


# --- Surface CompileConfig option types and defaults in autodoc --------------
# nanobind knows each property's type (via __nb_signature__) and writes it into the .pyi stub, but Sphinx's autodoc
# only reads standard __annotations__, which nanobind leaves empty on its nb_method getters. So autodoc shows neither
# the type nor a default for CompileConfig's def_rw properties. We bridge that here: the type is read straight from
# nanobind's own signature metadata, and the default is read from a single default-constructed CompileConfig() (with a
# small override map for the few defaults that are machine-derived or set Python-side).
import re as _re  # noqa: E402

_QD_MODULE_PREFIXES = (
    "quadrants._lib.core.quadrants_python.",
    "quadrants._lib.core.",
)

# Defaults that a default-constructed CompileConfig() reports incorrectly for an end user: machine-derived values, or
# values the Python frontend overrides. A value of None means "do not emit an auto default" (it is covered in prose).
_CONFIG_DEFAULT_OVERRIDES = {
    "offline_cache": "``True``",  # frontend sets this; the C++ default is False
    "arch": None,  # machine-derived; documented in the description
    "cpu_max_num_threads": "the number of available CPU cores",
    # machine-derived (built from XDG_CACHE_HOME/HOME, and differs on Windows); avoid leaking the doc builder's path
    "offline_cache_file_path": "the per-user Quadrants cache directory (``~/.cache/quadrants/qdcache`` on Linux)",
    "default_fp": "``qd.f32``",
    "default_ip": "``qd.i32``",
}


def _shorten_qd_type(type_str):
    for prefix in _QD_MODULE_PREFIXES:
        type_str = type_str.replace(prefix, "")
    return type_str.strip()


def _nb_property_type(prop):
    """Return the getter's return type from nanobind's __nb_signature__, if any."""
    fget = getattr(prop, "fget", None)
    signature = getattr(fget, "__nb_signature__", None)
    if not signature:
        return None
    match = _re.search(r"->\s*(.+?)\s*$", signature[0][0])
    return _shorten_qd_type(match.group(1)) if match else None


def _config_default_text(attr):
    """Human-readable default for a CompileConfig attribute, or None to skip."""
    if attr in _CONFIG_DEFAULT_OVERRIDES:
        return _CONFIG_DEFAULT_OVERRIDES[attr]
    value = getattr(_default_config(), attr, None)
    # Only emit literal defaults for plain scalar types; enums/dtypes are covered either by an override above or by
    # their description.
    if type(value) in (bool, int, float, str):
        return f"``{value!r}``"
    return None


_DEFAULT_CONFIG_INSTANCE = None


def _default_config():
    global _DEFAULT_CONFIG_INSTANCE
    if _DEFAULT_CONFIG_INSTANCE is None:
        from quadrants._lib.core.quadrants_python import CompileConfig

        _DEFAULT_CONFIG_INSTANCE = CompileConfig()
    return _DEFAULT_CONFIG_INSTANCE


def _patch_property_documenter():
    from sphinx.ext.autodoc import PropertyDocumenter

    original_add_directive_header = PropertyDocumenter.add_directive_header

    def add_directive_header(self, sig):
        original_add_directive_header(self, sig)
        if self.config.autodoc_typehints == "none":
            return
        type_str = _nb_property_type(self.object)
        if type_str:
            self.add_line("   :type: " + type_str, self.get_sourcename())

    PropertyDocumenter.add_directive_header = add_directive_header


def _append_config_default(app, what, name, obj, options, lines):
    if what != "property":
        return
    parts = name.split(".")
    if len(parts) < 2 or parts[-2] != "CompileConfig":
        return
    default_text = _config_default_text(parts[-1])
    if default_text:
        lines.extend(["", f"Default: {default_text}."])


def _patch_autoclass_module_names():
    # Show CompileConfig by its short name rather than its full internal dotted path. Rather than flip add_module_names
    # globally -- which would restyle the entire autoapi API reference -- we scope the suppression to just this one
    # autoclass directive, toggling the config for its run() (which also covers the nested rendering of its members).
    from sphinx.ext.autodoc.directive import AutodocDirective

    original_run = AutodocDirective.run

    def run(self):
        target = self.arguments[0] if self.arguments else ""
        if "quadrants._lib.core" not in target:
            return original_run(self)
        saved = self.config.add_module_names
        self.config.add_module_names = False
        try:
            return original_run(self)
        finally:
            self.config.add_module_names = saved

    AutodocDirective.run = run


def setup(app):
    _patch_property_documenter()
    _patch_autoclass_module_names()
    app.connect("autodoc-process-docstring", _append_config_default)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
