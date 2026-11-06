"""Kernel source capture must survive a transient empty linecache.getlines().

Guards against the OSError seen when linecache.getlines() transiently returns [] for quadrants/_kernels.py under load,
which aborted kernel compilation. CPU-only.
"""

import linecache

from quadrants._kernels import ext_arr_to_ndarray
from quadrants.lang import _wrap_inspect


def _kernel_fn():
    return getattr(ext_arr_to_ndarray, "fn", ext_arr_to_ndarray)


def test_source_capture_survives_transient_linecache_miss(monkeypatch):
    fn = _kernel_fn()
    real_getlines = linecache.getlines

    def flaky_getlines(filename, module_globals=None):
        if filename.endswith("quadrants/_kernels.py"):
            return []
        return real_getlines(filename, module_globals)

    monkeypatch.setattr(linecache, "getlines", flaky_getlines)

    info, src = _wrap_inspect.get_source_info_and_src(fn)
    assert info.function_name == "ext_arr_to_ndarray"
    assert "def ext_arr_to_ndarray" in "".join(src)


def test_direct_file_findsource_matches_stdlib():
    fn = _kernel_fn()
    _info, src_normal = _wrap_inspect.get_source_info_and_src(fn)
    lines, lineno = _wrap_inspect._direct_file_findsource(fn)
    assert lines[lineno].lstrip().startswith(("def ext_arr_to_ndarray", "@"))
    assert "".join(src_normal).strip() in "".join(lines)
