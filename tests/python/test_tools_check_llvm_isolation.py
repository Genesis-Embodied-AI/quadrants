import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

_tool_path = Path(__file__).parents[2] / "python/tools/check_llvm_isolation.py"
_spec = importlib.util.spec_from_file_location("check_llvm_isolation", _tool_path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux ELF wheel audit")


@pytest.fixture
def build_library(tmp_path):
    for tool in ("c++", "nm", "readelf"):
        if not shutil.which(tool):
            pytest.skip(f"Requires {tool}")

    def build(source, name="core.so", flags=()):
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
def test_detects_llvm_symbols(build_library, source, label):
    failures = audit.audit_library(build_library(source))
    assert len(failures) == 1
    assert failures[0].startswith(f"{label} LLVM symbol:")


def test_detects_shared_llvm_without_llvm_symbols(build_library):
    dependency = build_library('extern "C" int ordinary() { return 1; }', name="libLLVM-test.so")
    library = build_library("int use() { return 0; }", flags=["-Wl,--no-as-needed", str(dependency)])
    assert audit.audit_library(library) == [f"shared LLVM dependency: {dependency}"]


def test_wheel_audit_and_cli(build_library, tmp_path):
    source = 'namespace llvm { int probe() { return 1; } } extern "C" int unrelated() { return 2; }'
    for hidden in (True, False):
        library = build_library(source, flags=["-fvisibility=hidden"] if hidden else [])
        wheel = tmp_path / "quadrants.whl"
        with ZipFile(wheel, "w") as archive:
            archive.write(library, "quadrants/_lib/core/quadrants_python.cpython-312-x86_64-linux-gnu.so")
        result = subprocess.run([sys.executable, str(_tool_path), str(wheel)], capture_output=True, text=True)
        assert result.returncode == (0 if hidden else 1), result.stdout + result.stderr
        assert ("PASS:" if hidden else "exported LLVM symbol:") in result.stdout


def test_missing_core_fails(tmp_path):
    wheel = tmp_path / "empty.whl"
    with ZipFile(wheel, "w"):
        pass
    with pytest.raises(ValueError, match="expected one Quadrants core extension"):
        audit.audit_wheel(wheel)
