"""Regression test for the committed CUDA fatbin headers' per-toolkit layout.

Each `scripts/build_*_fatbin.py` compiles its `SM_VERSIONS` as one fatbin *per build toolkit* (see
`scripts/_fatbin_common.py`) and bundles them into a generated `*_fatbin.h`. This test asserts the committed header
actually contains exactly the expected architectures, in exactly the expected blobs, each built by exactly the expected
CUDA toolkit -- e.g. for the condition kernel, `sm_90/sm_100/sm_120` built with CUDA 12.8 in blob 0 and `sm_110` built
with CUDA 13.0+ in blob 1.

The expected layout is hardcoded below (`EXPECTED_LAYOUT`) rather than re-derived from `scripts/_fatbin_common.py`, so
this test independently pins the mapping: a wrong `SM_TOOLKIT` that still produces a plausible-looking header is caught,
not rubber-stamped. It reads the build toolkit that cuobjdump reports for *every* embedded cubin, so it directly checks
the driver <-> cubin property behind genesis-world#2942 -- the sm_120 (RTX 5090) cubin must be built with the wide-
compatibility CUDA 12.8, not a newer toolkit that would reject 570-series drivers with CUDA_ERROR_INVALID_IMAGE. It also
drift-guards each `SM_VERSIONS` list against the arch set actually baked into the checked-in header.

It shells out to `cuobjdump`, so it is skipped unless `cuobjdump` is on `PATH`. The toolkit providing it must also be at
least as new as the newest one used to build the fatbins (CUDA 13.0+, for sm_110); otherwise cuobjdump cannot parse that
blob and the affected header is skipped rather than failed.
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

# Intended layout of each committed header, keyed by build-script module (each exposes SM_VERSIONS + OUT_HEADER). The
# value lists the header's blobs in file order; each blob is (expected build toolkit, [expected SASS archs]). The
# toolkit is checked against the version cuobjdump reports for every embedded cubin: "12.8" must match exactly, "13.0+"
# allows any CUDA 13.x. This is deliberately a hand-maintained copy of the SM->toolkit design in _fatbin_common.py so a
# regression in that mapping is caught here instead of silently regenerating a bad header.
EXPECTED_LAYOUT = {
    "build_condition_kernel_fatbin": [("12.8", [90, 100, 120]), ("13.0+", [110])],
    "build_checkpoint_gate_fatbin": [("12.8", [90, 100, 120]), ("13.0+", [110])],
    "build_checkpoint_yield_check_fatbin": [("12.8", [60, 70, 80, 90, 100, 120]), ("13.0+", [110])],
}

_BLOB_RE = re.compile(r"static const unsigned char (\w+_\d+)\[\] = \{(.*?)\};", re.DOTALL)
_HEX_RE = re.compile(r"0x([0-9a-fA-F]{2})")
# cuobjdump -elf prints one header line per embedded cubin, e.g. "... sm=90, toolkit=128, flags=..." (older, packed
# form) or "... sm=110, toolkit=13.1, flags=..." (dotted form).
_CUBIN_RE = re.compile(r"\bsm=(\d+),\s*toolkit=([0-9.]+)")


def _fmt(sms):
    return ", ".join(f"sm_{s}" for s in sorted(sms))


def _norm_toolkit(raw):
    """Normalize cuobjdump's build-toolkit field to "MAJOR.MINOR" (it prints dotted "13.1" or packed "128" == 12.8)."""
    if "." in raw:
        return raw
    packed = int(raw)
    return f"{packed // 10}.{packed % 10}"


def _ver(dotted):
    major, _, minor = dotted.partition(".")
    return (int(major), int(minor or 0))


def _toolkit_matches(actual, expected):
    """`expected` is an exact "MAJOR.MINOR", or a "MAJOR.MINOR+" lower bound *within the same CUDA major*.

    The `+` is major-bounded on purpose: a SASS cubin's minimum driver is its toolkit's major family floor, so a newer
    minor within the same major is fine but a newer *major* would silently raise the floor. "13.0+" therefore accepts
    13.1 but rejects 14.0, matching the `>=13.0` resolver bound in scripts/_fatbin_common.py.
    """
    if expected.endswith("+"):
        floor = _ver(expected[:-1])
        return floor <= _ver(actual) and _ver(actual)[0] == floor[0]
    return actual == expected


def _blobs_from_header(text):
    """Return [(array_name, bytes), ...] for each `static const unsigned char <base>_<i>[]` blob, in file order."""
    return [(name, bytes(int(h, 16) for h in _HEX_RE.findall(body))) for name, body in _BLOB_RE.findall(text)]


def _fatbin_cubins(data):
    """cuobjdump the fatbin bytes; return (returncode, [(sm:int, toolkit:"MAJOR.MINOR"), ...], stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".fatbin") as f:
        f.write(data)
        f.flush()
        proc = subprocess.run(["cuobjdump", "-elf", f.name], capture_output=True, text=True)
    cubins = [(int(sm), _norm_toolkit(tk)) for sm, tk in _CUBIN_RE.findall(proc.stdout)]
    return proc.returncode, cubins, proc.stderr


@pytest.mark.skipif(shutil.which("cuobjdump") is None, reason="cuobjdump (CUDA toolkit) not on PATH")
@pytest.mark.parametrize("script_name", EXPECTED_LAYOUT)
def test_committed_fatbin_layout(script_name):
    mod = importlib.import_module(script_name)
    header = Path(mod.OUT_HEADER)
    assert header.is_file(), f"{header} missing -- regenerate with scripts/{script_name}.py"

    expected_blobs = EXPECTED_LAYOUT[script_name]
    blobs = _blobs_from_header(header.read_text())
    assert len(blobs) == len(expected_blobs), (
        f"{header.name}: found {len(blobs)} fatbin blob(s) but expected {len(expected_blobs)} "
        f"-- regenerate with scripts/{script_name}.py"
    )

    seen = set()
    for (name, data), (expected_toolkit, expected_sms) in zip(blobs, expected_blobs):
        rc, cubins, stderr = _fatbin_cubins(data)
        if rc != 0 or not cubins:
            floor = expected_toolkit.rstrip("+")
            pytest.skip(f"cuobjdump could not parse {name} (needs CUDA >= {floor}): {stderr.strip()}")
        archs = sorted(sm for sm, _ in cubins)
        assert archs == sorted(expected_sms), (
            f"{header.name}:{name} contains [{_fmt(archs)}], expected [{_fmt(expected_sms)}] "
            f"-- regenerate with scripts/{script_name}.py"
        )
        for sm, toolkit in cubins:
            assert _toolkit_matches(toolkit, expected_toolkit), (
                f"{header.name}:{name} sm_{sm} was built with CUDA {toolkit}, expected {expected_toolkit} -- a SASS cubin "
                f"only loads on drivers whose CUDA version is >= its build toolkit, so this regresses genesis-world#2942"
            )
        seen |= set(archs)

    assert seen == set(mod.SM_VERSIONS), (
        f"{header.name}: bundled archs [{_fmt(seen)}] != SM_VERSIONS [{_fmt(mod.SM_VERSIONS)}] "
        f"-- regenerate with scripts/{script_name}.py"
    )
