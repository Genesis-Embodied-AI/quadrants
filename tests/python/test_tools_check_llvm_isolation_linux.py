"""Test the LLVM wheel-audit tool using synthetic libraries and wheels with known passing and failing properties."""

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from zipfile import ZipFile

import pytest

import quadrants as qd

from tests.helpers import test_utils

if sys.platform != "linux":
    pytest.skip("Linux ELF wheel audit", allow_module_level=True)

_tool_path = Path(__file__).parents[2] / "python/tools/check_llvm_isolation.py"
_spec = importlib.util.spec_from_file_location("check_llvm_isolation", _tool_path)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
LibraryBuilder = Callable[..., Path]


@pytest.fixture
def build_library(tmp_path: Path) -> LibraryBuilder:
    """Provide a compiler helper for small ELF fixtures, skipping when the required tools are unavailable."""
    for tool in ("c++", "nm", "readelf"):
        if not shutil.which(tool):
            pytest.skip(f"Requires {tool}")

    def build(source: str, name: str = "core.so", flags: Sequence[str] = ()) -> Path:
        """Compile C++ source into a shared library with optional linker or visibility flags."""
        cpp = tmp_path / f"{name}.cpp"
        cpp.write_text(source)
        library = tmp_path / name
        subprocess.run(
            ["c++", "-shared", "-fPIC", str(cpp), "-o", str(library), *flags], check=True, capture_output=True
        )
        return library

    return build


@pytest.mark.parametrize(
    "source,label",
    [
        ('extern "C" int LLVMProbe() { return 1; }', "exported"),
        ("namespace llvm { int probe() { return 1; } }", "exported"),
        ('extern "C" int llvm_orc_probe() { return 1; }', "exported"),
        ('extern "C" int LLVMProbe(); int use() { return LLVMProbe(); }', "undefined"),
        (
            'extern "C" int LLVMProbe() __attribute__((weak)); int use() { return LLVMProbe ? LLVMProbe() : 0; }',
            "undefined",
        ),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_detects_llvm_symbols(build_library: LibraryBuilder, source: str, label: str) -> None:
    """Reject exported and undefined LLVM symbols, including weak references and runtime entry points."""
    failures = audit.audit_library(build_library(source))
    assert len(failures) == 1
    assert failures[0].startswith(f"{label} LLVM symbol:")


@test_utils.test(arch=qd.cpu)
def test_detects_shared_llvm_without_llvm_symbols(build_library: LibraryBuilder) -> None:
    """Detect a shared LLVM dependency even when no LLVM-named symbols are visible."""
    dependency = build_library('extern "C" int ordinary() { return 1; }', name="libLLVM-test.so")
    library = build_library("int use() { return 0; }", flags=["-Wl,--no-as-needed", str(dependency)])
    assert audit.audit_library(library) == [f"shared LLVM dependency: {dependency}"]


@test_utils.test(arch=qd.cpu)
def test_wheel_audit_and_cli(build_library: LibraryBuilder, tmp_path: Path) -> None:
    """It describes an end-to-end test of our **command-line audit tool** (`CLI`).

    The test compiles a tiny library containing a function named `llvm::probe()`, then checks two versions:

    - **Hidden:** compiled with `-fvisibility=hidden`. The function exists inside the library but is not exported
      through its dynamic symbol table. The audit must report `PASS` and exit with code **0**.
    - **Exported:** compiled with default visibility. Other libraries could resolve that symbol against this library.
      The audit must report an exported LLVM symbol and exit with code **1**.

    Each version is placed in a minimal `.whl` ZIP at the path where Quadrants’ extension normally lives. The test then
    launches `check_llvm_isolation.py` as a separate process, exercising extraction, symbol inspection, reporting, and
    exit status together.

    It uses a **synthetic LLVM-named function**, not an actual LLVM build or importable Python extension. Here, “accepts
    hidden LLVM symbols” means private LLVM code is allowed; only exposing it is a failure for this check.
    """
    source = 'namespace llvm { int probe() { return 1; } } extern "C" int unrelated() { return 2; }'
    for hidden in (True, False):
        library = build_library(source, flags=["-fvisibility=hidden"] if hidden else [])
        wheel = tmp_path / "quadrants.whl"
        with ZipFile(wheel, "w") as archive:
            archive.write(library, "quadrants/_lib/core/quadrants_python.cpython-312-x86_64-linux-gnu.so")
        result = subprocess.run([sys.executable, str(_tool_path), str(wheel)], capture_output=True, text=True)
        assert result.returncode == (0 if hidden else 1), result.stdout + result.stderr
        assert ("PASS:" if hidden else "exported LLVM symbol:") in result.stdout


@test_utils.test(arch=qd.cpu)
def test_missing_core_fails(tmp_path: Path) -> None:
    """Reject a wheel without a core extension instead of reporting a successful audit."""
    wheel = tmp_path / "empty.whl"
    with ZipFile(wheel, "w"):
        pass
    with pytest.raises(ValueError, match="expected one Quadrants core extension"):
        audit.audit_wheel(wheel)
