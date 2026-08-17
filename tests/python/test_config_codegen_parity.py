"""Parity guard: C++ CompileConfig and the option schema must stay in sync.

tools/config_codegen/schema.py is the single source of truth; the C++ struct
fields, their defaults, and the nanobind bindings are all generated from it (see
the B1 design doc). These tests construct a fresh CompileConfig from the built
extension and check that:

  * every Python-settable schema option exists on the struct with the schema's
    default (literal defaults are compared exactly; computed ones are only
    checked to exist, since their value depends on the host), and
  * the struct exposes exactly the schema's set of bound options and nothing
    more -- catching a stale generated .inc, an accidental hand-edit, or a new
    binding added without a schema entry.
"""

import sys
from pathlib import Path

import pytest

from quadrants._lib import core as _qd_core

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "config_codegen"))

import generate  # noqa: E402

_BOUND = [o for o in generate.get_options() if o.bind]


def _fresh_config():
    _qd_core.reset_default_compile_config()
    return _qd_core.default_compile_config()


@pytest.mark.parametrize("opt", _BOUND, ids=lambda o: o.name)
def test_schema_default_matches_struct(opt):
    cfg = _fresh_config()
    assert hasattr(cfg, opt.name), f"CompileConfig is missing bound field {opt.name!r}"
    if opt.is_computed:
        # Value depends on the host (arch, thread count, cache dir, ...); the
        # constructor fragment is generated from the same schema, so just assert
        # the field is present and readable.
        getattr(cfg, opt.name)
        return
    actual = getattr(cfg, opt.name)
    assert actual == opt.literal_default, (
        f"{opt.name}: struct default {actual!r} != schema default {opt.literal_default!r}"
    )


def test_struct_has_no_unschemad_options():
    cfg = _fresh_config()
    runtime = {n for n in dir(cfg) if not n.startswith("_")}
    schema_bound = {o.name for o in _BOUND}
    extra = runtime - schema_bound
    missing = schema_bound - runtime
    assert not extra, f"CompileConfig exposes options with no schema entry: {sorted(extra)}"
    assert not missing, f"schema options missing from CompileConfig: {sorted(missing)}"
