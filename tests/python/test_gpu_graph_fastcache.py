"""Test that fastcache correctly preserves graph_do_while_arg.

When a gpu_graph=True kernel using graph_do_while is loaded from the fastcache,
the graph_do_while_arg must be restored so the do-while counter argument is
correctly identified during kernel launch. Without this, the GPU graph replay
doesn't know which argument controls the loop, producing wrong results.
"""
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _gpu_graph_used():
    return impl.get_runtime().prog.get_gpu_graph_cache_used_on_last_call()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _skip_if_cuda_without_hopper():
    if _on_cuda() and qd.lang.impl.get_cuda_compute_capability() < 90:
        pytest.skip("graph_do_while requires SM 9.0+ (Hopper)")


def _import_kernel(filepath: Path, module_name: str, kernel_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, kernel_name)


KERNEL_SRC = """\
import quadrants as qd

@qd.kernel(gpu_graph=True, fastcache=True)
def k(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[()] = counter[()] - 1
"""


@test_utils.test()
def test_gpu_graph_do_while_fastcache_graph_do_while_arg_restored():
    """After fastcache restore, graph_do_while_arg should be set on the Kernel."""
    if not _on_cuda():
        pytest.skip("gpu_graph requires CUDA")
    _skip_if_cuda_without_hopper()

    N = 16
    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    mod_name = "_test_fastcache_do_while_mod"

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "k.py"
        filepath.write_text(KERNEL_SRC)

        # First import: compiles and populates the fastcache
        k1 = _import_kernel(filepath, mod_name, "k")
        x.from_numpy(np.zeros(N, dtype=np.int32))
        counter.from_numpy(np.array(5, dtype=np.int32))
        k1(x, counter)

        primal1 = k1._primal
        assert primal1.graph_do_while_arg == "counter"
        assert primal1.src_ll_cache_observations.cache_stored

        np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5))
        assert counter.to_numpy() == 0

        # Second import: loads from fastcache (same filepath = same cache key).
        # graph_do_while_arg must be restored from the cached metadata.
        k2 = _import_kernel(filepath, mod_name, "k")
        primal2 = k2._primal

        x.from_numpy(np.zeros(N, dtype=np.int32))
        counter.from_numpy(np.array(3, dtype=np.int32))
        k2(x, counter)

        assert primal2.src_ll_cache_observations.cache_loaded, "fastcache should have been loaded"
        assert primal2.graph_do_while_arg == "counter", (
            "graph_do_while_arg should be restored from fastcache"
        )

        np.testing.assert_array_equal(
            x.to_numpy(), np.full(N, 3),
            err_msg="graph_do_while counter not restored from fastcache",
        )
        assert counter.to_numpy() == 0

    if mod_name in sys.modules:
        del sys.modules[mod_name]
