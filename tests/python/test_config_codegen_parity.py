"""Parity guard: C++ CompileConfig defaults must match the option schema.

tools/config_codegen/schema.py is the single source of truth; the C++ struct
fields and their defaults are generated from it (see the B1 design doc). This
test constructs a fresh CompileConfig from the built extension and checks that
each schema option's default is what the struct actually holds -- catching a
stale generated .inc or an accidental hand-edit.
"""

import sys
from pathlib import Path

import pytest

from quadrants._lib import core as _qd_core

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "config_codegen"))

import generate  # noqa: E402


def _fresh_config():
    _qd_core.reset_default_compile_config()
    return _qd_core.default_compile_config()


@pytest.mark.parametrize("opt", generate.get_options(), ids=lambda o: o.name)
def test_schema_default_matches_struct(opt):
    cfg = _fresh_config()
    assert hasattr(cfg, opt.name), f"CompileConfig is missing field {opt.name!r}"
    actual = getattr(cfg, opt.name)
    if opt.is_computed:
        # The only computed default in the current schema is offline_cache_file_path.
        assert isinstance(actual, str) and actual.endswith("qdcache"), (
            f"{opt.name}: unexpected computed default {actual!r}"
        )
    else:
        assert actual == opt.literal_default, (
            f"{opt.name}: struct default {actual!r} != schema default {opt.literal_default!r}"
        )
