"""Regression test: kernel source capture must survive a transient linecache miss.

Reproduces the failure where stdlib ``linecache.getlines`` transiently returns an
empty list for ``quadrants/_kernels.py``. This was observed under heavy concurrent
load (dozens of xdist workers plus torch/CUDA and other jobs on the node): a
``MemoryError`` makes ``linecache.getlines`` call ``linecache.clearcache()`` and
return ``[]``, and transient ``os.stat``/``open`` failures in
``linecache.updatecache`` have the same effect.

Because Quadrants materializes kernels by re-reading their ``.py`` source at
runtime (``inspect.getsourcelines`` -> ``linecache``), and every ``_wrap_inspect``
fallback (stdlib, dill, bpy) is linecache/source based, a single transient empty
read turned a readable file into a fatal
``OSError: Cannot find source code for Object: <function ext_arr_to_ndarray ...>``
(``_wrap_inspect.py``). The direct-file fallback added in ``_custom_findsource``
recovers from that transient.

CPU-only; no device initialization required.
"""

import linecache

from quadrants._kernels import ext_arr_to_ndarray
from quadrants.lang import _wrap_inspect


def _kernel_fn():
    # ``@kernel`` wraps the function; the raw function is exposed as ``.fn``.
    return getattr(ext_arr_to_ndarray, "fn", ext_arr_to_ndarray)


def test_source_capture_survives_transient_linecache_miss(monkeypatch):
    fn = _kernel_fn()

    real_getlines = linecache.getlines

    def flaky_getlines(filename, module_globals=None):
        # Simulate the transient empty read for the kernel's own source file.
        if filename.endswith("quadrants/_kernels.py"):
            return []
        return real_getlines(filename, module_globals)

    monkeypatch.setattr(linecache, "getlines", flaky_getlines)

    # Before the fix this raised OSError; the direct-file fallback must recover.
    info, src = _wrap_inspect.get_source_info_and_src(fn)

    assert info.function_name == "ext_arr_to_ndarray"
    assert "def ext_arr_to_ndarray" in "".join(src)


def test_direct_file_findsource_matches_stdlib():
    # The direct-read fallback must return the same source as the normal path
    # when linecache is healthy.
    fn = _kernel_fn()
    _info, src_normal = _wrap_inspect.get_source_info_and_src(fn)
    lines, lineno = _wrap_inspect._direct_file_findsource(fn)
    assert lines[lineno].lstrip().startswith(("def ext_arr_to_ndarray", "@"))
    assert "".join(src_normal).strip() in "".join(lines)
