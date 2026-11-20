"""Check final Linux wheels for LLVM exports, imports, and shared-library dependencies.

Usage: python python/tools/check_llvm_isolation.py wheelhouse/*.whl

Requires GNU nm and readelf. This checks ELF metadata, not runtime dlsym/JIT lookups.
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

# Include demangled C++ names, the C API, and LLVM runtime entry points. LLVM types in an exported signature also cross
# the intended private-library boundary, so check the complete demangled name.
LLVM_SYMBOL = re.compile(r"\b(?:llvm::|LLVM\w*|llvm_\w+|__llvm_\w+)")


def run_tool(*args: str) -> str:
    """Run a binary inspection tool and return its output, raising if the command fails."""
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def audit_library(library: Path) -> list[str]:
    """Report LLVM symbols and shared LLVM dependencies exposed by an ELF library."""
    failures = []
    for label, selection in (("exported", "--defined-only"), ("undefined", "--undefined-only")):
        symbols = run_tool("nm", "--dynamic", "--extern-only", "--demangle", selection, str(library))
        failures.extend(
            f"{label} LLVM symbol: {line.strip()}" for line in symbols.splitlines() if LLVM_SYMBOL.search(line)
        )
    dynamic = run_tool("readelf", "--dynamic", "--wide", str(library))
    for dependency in re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic):
        if Path(dependency).name.startswith("libLLVM"):
            failures.append(f"shared LLVM dependency: {dependency}")
    return failures


def audit_wheel(wheel: Path) -> list[str]:
    """Extract and audit the wheel's single Quadrants core extension, rejecting missing or ambiguous cores."""
    with ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as directory:
        cores = [
            name for name in archive.namelist() if re.fullmatch(r"quadrants/_lib/core/quadrants_python[^/]*\.so", name)
        ]
        if len(cores) != 1:
            raise ValueError(f"expected one Quadrants core extension, found {len(cores)}")
        library = Path(directory) / Path(cores[0]).name
        library.write_bytes(archive.read(cores[0]))
        return audit_library(library)


def main() -> int:
    """Audit each wheel passed on the command line and return a nonzero status if any audit fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for wheel in args.wheels:
        try:
            failures = audit_wheel(wheel)
        except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as error:
            failures = [str(error)]
            if isinstance(error, subprocess.CalledProcessError):
                failures.append(error.output.strip())
        failed |= bool(failures)
        print(f"{'FAIL' if failures else 'PASS'}: {wheel}")
        for failure in failures:
            print(f"  {failure}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
