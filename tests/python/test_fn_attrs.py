"""Tests for the @qd.kernel(fn_attrs=...) plumbing.

Three layers of coverage:

1. Validation: unknown backend / unknown attr name raise at decoration time.
   Cheap, runs on any backend.
2. Cache differentiation: identical kernels with different fn_attrs values
   produce different fastcache keys. Verifies the hash plumbing in
   `src_hasher` (Python) and `offline_cache_util` (C++).
3. End-to-end IR plumbing: registered attribute appears verbatim in the
   LLVM IR that jit_amdgpu sees. Gated on AMDGPU backend; runs the kernel
   in a subprocess with print_kernel_llvm_ir=True and greps the dump.
"""

import os
import pathlib
import subprocess
import sys

import pytest

import quadrants as qd
from quadrants.lang._fast_caching import src_hasher
from quadrants.lang._wrap_inspect import get_source_info_and_src
from quadrants.lang.exception import QuadrantsSyntaxError

from tests import test_utils

RET_SUCCESS = 42


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_fn_attrs_unknown_backend_raises():
    with pytest.raises(QuadrantsSyntaxError, match="unknown backend 'nvidia'"):

        @qd.kernel(fn_attrs={"nvidia": {"some-attr": "x"}})
        def k():
            pass


@test_utils.test(arch=[qd.cpu])
def test_fn_attrs_unknown_amdgpu_attr_raises():
    with pytest.raises(QuadrantsSyntaxError, match="unknown amdgpu attribute"):

        @qd.kernel(fn_attrs={"amdgpu": {"definitely-not-a-real-attr": "0,0"}})
        def k():
            pass


@test_utils.test(arch=[qd.cpu])
def test_fn_attrs_registered_attr_accepts():
    # Should not raise; uses an attribute we know is in the registry.
    @qd.kernel(fn_attrs={"amdgpu": {"amdgpu-max-num-workgroups": "1,1,1"}})
    def k():
        pass

    assert k._primal.fn_attrs == {"amdgpu": {"amdgpu-max-num-workgroups": "1,1,1"}}


# ---------------------------------------------------------------------------
# 2. Cache key differentiation (Python fastcache layer)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_fn_attrs_changes_fastcache_key():
    # Same source, same args, same config — only fn_attrs differs.
    # Cache keys must differ, otherwise a fn_attrs change would silently
    # serve a stale CompiledKernelData.
    def _kernel_source():
        @qd.kernel
        def k():
            pass

        return k

    k = _kernel_source()
    info, _ = get_source_info_and_src(k._primal.func)

    key_none = src_hasher.create_cache_key(False, info, (), (), None)
    key_a = src_hasher.create_cache_key(
        False, info, (), (), {"amdgpu": {"amdgpu-max-num-workgroups": "1,1,1"}}
    )
    key_b = src_hasher.create_cache_key(
        False, info, (), (), {"amdgpu": {"amdgpu-max-num-workgroups": "256,1,1"}}
    )

    assert key_none is not None and key_a is not None and key_b is not None
    assert key_none != key_a, "fn_attrs=None must hash differently from fn_attrs={...}"
    assert key_a != key_b, "different fn_attr values must produce different cache keys"


# ---------------------------------------------------------------------------
# 3. End-to-end IR plumbing — AMDGPU only
# ---------------------------------------------------------------------------


# Marker value — unique enough that grep finds it only on the kernel we set
# it on, not on any incidental codegen output.
_FN_ATTRS_MARKER = "1,7,42"


def _e2e_child(args: list[str]) -> None:
    dump_dir = args[0]
    os.chdir(dump_dir)  # print_kernel_llvm_ir writes to CWD
    qd.init(arch=qd.amdgpu, print_kernel_llvm_ir=True, offline_cache=False)

    # Use a non-trivial body so the kernel has a real offloaded task. An
    # empty kernel may produce zero offloaded tasks, in which case the
    # per-task addFnAttr loop in codegen_llvm.cpp's Arch::amdgpu branch
    # has nothing to attach the attribute to and the test would falsely
    # fail.
    buf = qd.ndarray(qd.f32, shape=(64,))

    @qd.kernel(fn_attrs={"amdgpu": {"amdgpu-max-num-workgroups": _FN_ATTRS_MARKER}})
    def with_attr(x: qd.types.ndarray(qd.f32, 1)):
        for i in x:
            x[i] = 1.0

    with_attr(buf)
    sys.exit(RET_SUCCESS)


@test_utils.test(arch=[qd.amdgpu])
def test_fn_attrs_reaches_amdgpu_jit_ir(tmp_path: pathlib.Path):
    # Run the kernel in a fresh subprocess so the LLVM IR dump lives in a
    # clean directory we can grep without interference from other tests.
    cmd = [sys.executable, __file__, _e2e_child.__name__, str(tmp_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + "."
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != RET_SUCCESS:
        print(proc.stdout)
        print("-" * 80)
        print(proc.stderr)
    assert proc.returncode == RET_SUCCESS

    ll_files = list(tmp_path.glob("quadrants_kernel_amdgpu_llvm_ir_*.ll"))
    assert ll_files, f"no LLVM IR dumps produced in {tmp_path}"

    # The marker should appear in an attribute group at the bottom of the IR
    # as `"amdgpu-max-num-workgroups"="1,7,42"`. Search loosely (just the key
    # and the value) so we tolerate any escaping/whitespace LLVM may apply.
    matches = []
    for p in ll_files:
        text = p.read_text()
        if "amdgpu-max-num-workgroups" in text and _FN_ATTRS_MARKER in text:
            matches.append(p)

    if not matches:
        # Dump every `attributes #N` group from each .ll so the failure is
        # actionable instead of opaque.
        debug = []
        for p in ll_files:
            attr_lines = [
                ln for ln in p.read_text().splitlines() if ln.startswith("attributes #")
            ]
            debug.append(f"--- {p.name} ---\n" + "\n".join(attr_lines[:20]))
        pytest.fail(
            f"expected 'amdgpu-max-num-workgroups' with value {_FN_ATTRS_MARKER!r} "
            f"in at least one of {[p.name for p in ll_files]}, but absent.\n"
            f"Attribute groups found:\n" + "\n\n".join(debug)
        )


# ---------------------------------------------------------------------------
# Subprocess dispatch (matches the pattern used in test_config.py).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    name, *rest = sys.argv[1:]
    {
        _e2e_child.__name__: _e2e_child,
    }[name](rest)
