"""Regression test for the committed CUDA fatbin headers' per-toolkit arch partition.

Each `scripts/build_*_fatbin.py` compiles its `SM_VERSIONS` as one fatbin *per build toolkit* (see
`scripts/_fatbin_common.py`) and bundles them into a generated `*_fatbin.h`. This test re-derives that partition and
asserts the committed header actually contains exactly those architectures, in exactly those blobs -- e.g. for the
condition kernel, `sm_90/sm_100/sm_120` in blob 0 (CUDA 12.8) and `sm_110` in blob 1 (CUDA 13.0+).

This is the check that would have caught the driver <-> cubin mismatch behind genesis-world#2942: the sm_120 (RTX 5090)
cubin must be built with the wide-compatibility toolkit and therefore live in a *separate* blob from sm_110 (Thor,
CUDA 13.0-only). If a future change collapses everything back into a single newer-toolkit fatbin, sm_110 lands next to
sm_120 in blob 0 and this test fails. It also drift-guards each `SM_VERSIONS` list against the arch set actually baked
into the checked-in header.

It shells out to `cuobjdump`, so it is skipped unless `cuobjdump` is on `PATH`. The toolkit providing it must also be
at least as new as the newest one used to build the fatbins (CUDA 13.0+, for sm_110); otherwise cuobjdump cannot parse
that blob and the affected header is skipped rather than failed.
"""

import importlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _fatbin_common import group_archs_by_toolkit  # noqa: E402

# Build-script module names; each exposes SM_VERSIONS + OUT_HEADER for one committed multi-toolkit fatbin header.
BUILD_SCRIPTS = [
    "build_condition_kernel_fatbin",
    "build_checkpoint_gate_fatbin",
    "build_checkpoint_yield_check_fatbin",
]

_BLOB_RE = re.compile(r"static const unsigned char (\w+_\d+)\[\] = \{(.*?)\};", re.DOTALL)
_HEX_RE = re.compile(r"0x([0-9a-fA-F]{2})")
_SM_RE = re.compile(r"sm_(\d+)")


def _fmt(sms):
    return ", ".join(f"sm_{s}" for s in sorted(sms))


def _blobs_from_header(text):
    """Return [(array_name, bytes), ...] for each `static const unsigned char <base>_<i>[]` blob, in file order."""
    return [(name, bytes(int(h, 16) for h in _HEX_RE.findall(body))) for name, body in _BLOB_RE.findall(text)]


def _fatbin_archs(data):
    """cuobjdump the fatbin bytes; return (returncode, sorted list of int SM targets, stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".fatbin") as f:
        f.write(data)
        f.flush()
        proc = subprocess.run(["cuobjdump", "-lelf", f.name], capture_output=True, text=True)
    return proc.returncode, sorted({int(a) for a in _SM_RE.findall(proc.stdout)}), proc.stderr


@pytest.mark.skipif(shutil.which("cuobjdump") is None, reason="cuobjdump (CUDA toolkit) not on PATH")
@pytest.mark.parametrize("script_name", BUILD_SCRIPTS)
def test_committed_fatbin_arch_partition(script_name):
    mod = importlib.import_module(script_name)
    sm_versions = mod.SM_VERSIONS
    header = Path(mod.OUT_HEADER)
    assert header.is_file(), f"{header} missing -- regenerate with scripts/{script_name}.py"

    # group_archs_by_toolkit returns the groups in the same order build_fatbin_header emits the blobs.
    groups = group_archs_by_toolkit(sm_versions)
    blobs = _blobs_from_header(header.read_text())
    assert len(blobs) == len(groups), (
        f"{header.name}: found {len(blobs)} fatbin blob(s) but SM_VERSIONS partitions into {len(groups)} toolkit "
        f"group(s) -- regenerate with scripts/{script_name}.py"
    )

    seen = set()
    for (name, data), (toolkit, expected_sms) in zip(blobs, groups):
        rc, archs, stderr = _fatbin_archs(data)
        if rc != 0:
            pytest.skip(f"cuobjdump could not parse {name} (likely older than CUDA {toolkit}): {stderr.strip()}")
        assert archs == sorted(expected_sms), (
            f"{header.name}:{name} contains [{_fmt(archs)}], expected [{_fmt(expected_sms)}] (CUDA {toolkit} group) "
            f"-- regenerate with scripts/{script_name}.py"
        )
        seen |= set(archs)

    assert seen == set(sm_versions), f"{header.name}: bundled archs [{_fmt(seen)}] != SM_VERSIONS [{_fmt(sm_versions)}]"
