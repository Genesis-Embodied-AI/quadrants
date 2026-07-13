#!/usr/bin/env python3
"""Shared machinery for building the CUDA graph conditional / checkpoint kernel fatbins.

Each kernel's SASS is bundled for several SM architectures. A SASS cubin can only be loaded by a driver whose CUDA
version is >= the toolkit that produced it, so building every architecture with a single (newest) toolkit forces even
old architectures to require a recent driver. Concretely, bundling sm_110 (Thor, CUDA 13.0-only) alongside sm_120
(RTX 5090) with one CUDA 13.0 toolkit makes the sm_120 cubin reject 570-series (CUDA 12.8) drivers with
`CUDA_ERROR_INVALID_IMAGE`.

To avoid that, each architecture is compiled with the toolkit given by the explicit per-SM `SM_TOOLKIT` mapping,
producing one fatbin per distinct toolkit. All the per-toolkit fatbins are emitted into a single generated header as
separate byte arrays plus a small lookup table; at runtime the loader tries each in turn and keeps the one whose SASS
matches the running GPU (`GraphManager::load_first_matching_fatbin`). This keeps e.g. the sm_120 image loadable on the
widest range of drivers while still shipping the sm_110 image that only CUDA 13.0+ can emit.

Running the build scripts raises `MissingToolkitError` unless every required toolkit is available.
"""

import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Explicit CUDA toolkit requirement for every SM architecture we bundle. The value is a version spec:
#   "==X.Y"  build this arch with EXACTLY CUDA X.Y. A SASS cubin only loads on drivers whose CUDA version is >= the
#            toolkit that produced it, so the wide-compatibility archs (pre-Hopper, Hopper sm_90, and the shipping
#            Blackwell parts sm_100 / sm_120) are pinned to the oldest toolkit that covers them -- building them with a
#            newer toolkit would needlessly raise their minimum driver (this is what made sm_120 reject 570-series
#            drivers with CUDA_ERROR_INVALID_IMAGE).
#   ">=X.Y"  build this arch with the oldest *available* toolkit >= CUDA X.Y. Used for brand-new archs that have no
#            old-driver constraint (their hardware only ever ships with X.Y+ drivers), so any newer toolkit is fine.
#            sm_110 (Thor) needs CUDA 13.0 -- earlier toolkits reject compute_110 -- but 13.1, 13.2, ... are all OK too.
SM_TOOLKIT = {
    75: "==12.8",
    80: "==12.8",
    86: "==12.8",
    89: "==12.8",
    90: "==12.8",
    100: "==12.8",
    110: ">=13.0",
    120: "==12.8",
}


class MissingToolkitError(RuntimeError):
    """Raised when one or more of the CUDA toolkits required to build the fatbins cannot be found."""


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def _parse_spec(spec: str) -> tuple[str, str, tuple[int, ...]]:
    """Split a toolkit spec like '==12.8' or '>=13.0' into (operator, version_str, version_tuple)."""
    op, version = spec[:2], spec[2:]
    if op not in ("==", ">=") or not version:
        raise ValueError(f"invalid toolkit spec {spec!r}; expected '==X.Y' or '>=X.Y'")
    return op, version, _parse_version(version)


def group_archs_by_toolkit(sm_versions: list[int]) -> list[tuple[str, list[int]]]:
    """Partition `sm_versions` into (toolkit-spec, [sm, ...]) groups, preserving arch order within each group.

    Each SM's toolkit-spec comes from `SM_TOOLKIT` (a `==X.Y` exact pin or a `>=X.Y` minimum). Groups are ordered by the
    CUDA version they require (lowest first, exact before minimum on a tie), so the loader tries the widest-compatibility
    fatbin -- the one holding the common Blackwell / Hopper parts -- first. (Ordering is only cosmetic: each SM maps to
    exactly one spec, so a given GPU's cubin lives in exactly one blob.)
    """
    groups: dict[str, list[int]] = {}
    for sm in sm_versions:
        if sm not in SM_TOOLKIT:
            raise KeyError(f"sm_{sm} has no entry in SM_TOOLKIT (scripts/_fatbin_common.py); add its toolkit spec")
        groups.setdefault(SM_TOOLKIT[sm], []).append(sm)

    def sort_key(spec: str) -> tuple:
        op, _version, version_tuple = _parse_spec(spec)
        return (version_tuple, op != "==")

    return [(spec, groups[spec]) for spec in sorted(groups, key=sort_key)]


def _nvcc_release(nvcc: str) -> str | None:
    """Return the "major.minor" release reported by `nvcc --version`, or None if it can't be determined."""
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", out)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def discover_toolkits() -> dict[str, str]:
    """Discover every reachable nvcc, returning {release "major.minor": nvcc_path} (first found per release wins).

    Locations probed, in order: `QUADRANTS_NVCC_CANDIDATES` (os.pathsep-separated nvcc binaries and/or CUDA root dirs),
    the versioned `/usr/local/cuda-*/bin/nvcc` installs, `nvcc` on PATH, `CUDA_HOME` / `CUDA_PATH`, and
    `/usr/local/cuda/bin/nvcc`.
    """
    candidates: list[Path] = []
    env_candidates = os.environ.get("QUADRANTS_NVCC_CANDIDATES")
    if env_candidates:
        for entry in env_candidates.split(os.pathsep):
            entry = entry.strip()
            if entry:
                path = Path(entry)
                candidates.append(path if path.name == "nvcc" else path / "bin" / "nvcc")
    candidates += [Path(p) for p in sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"))]
    which = shutil.which("nvcc")
    if which:
        candidates.append(Path(which))
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root:
            candidates.append(Path(root) / "bin" / "nvcc")
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))

    found: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        release = _nvcc_release(str(candidate))
        if release and release not in found:
            found[release] = str(candidate)
    return found


def _resolve_group_toolkit(spec: str, available: dict[str, str]) -> tuple[str, str] | None:
    """Resolve one group's toolkit-spec against discovered toolkits, returning (nvcc_path, actual_release) or None.

    A `==X.Y` spec is matched exactly; a `>=X.Y` spec resolves to the oldest available release >= X.Y.
    """
    op, version, version_tuple = _parse_spec(spec)
    if op == "==":
        path = available.get(version)
        return (path, version) if path else None
    satisfying = sorted((r for r in available if _parse_version(r) >= version_tuple), key=_parse_version)
    return (available[satisfying[0]], satisfying[0]) if satisfying else None


def resolve_toolkits(groups: list[tuple[str, list[int]]]) -> dict[str, tuple[str, str]]:
    """Resolve an nvcc for every group, or raise `MissingToolkitError` listing what's missing.

    Returns {toolkit-spec: (nvcc_path, actual_release)}. All groups are resolved before any compilation happens so the
    script fails fast instead of half-building.
    """
    available = discover_toolkits()
    resolved: dict[str, tuple[str, str]] = {}
    missing: list[tuple[str, list[int]]] = []
    for spec, sms in groups:
        if spec in resolved:
            continue
        result = _resolve_group_toolkit(spec, available)
        if result is None:
            missing.append((spec, sms))
        else:
            resolved[spec] = result
    if missing:
        found_desc = ", ".join(f"{rel} ({path})" for rel, path in sorted(available.items())) or "none"
        lines = ["Missing required CUDA toolkit(s) to build the fatbins:"]
        for spec, sms in missing:
            arch_list = ", ".join(f"sm_{v}" for v in sms)
            op, version, _ = _parse_spec(spec)
            requirement = f"exactly CUDA {version}" if op == "==" else f"CUDA >= {version}"
            lines.append(f"  - {requirement} (needed for {arch_list})")
        lines.append("")
        lines.append(f"Discovered toolkits: {found_desc}")
        lines.append("")
        lines.append(
            "Each architecture is built with the oldest suitable toolkit, so all of the above must be present."
        )
        lines.append("Make toolkits discoverable via any of:")
        lines.append("  - QUADRANTS_NVCC_CANDIDATES=/path/to/nvcc:/path/to/cuda-root  (os.pathsep-separated)")
        lines.append("  - /usr/local/cuda-X.Y/bin/nvcc                                (conventional install location)")
        lines.append("  - nvcc on PATH, or CUDA_HOME / CUDA_PATH")
        raise MissingToolkitError("\n".join(lines))
    return resolved


def _run(cmd: list[str]) -> None:
    print(f"  {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _compile_group(nvcc: str, src: Path, sms: list[int]) -> bytes:
    """Compile + device-link `src` for the given SM list with one toolkit, returning the resulting fatbin bytes."""
    targets = [f"-gencode=arch=compute_{v},code=sm_{v}" for v in sms]
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = Path(tmpdir) / "kernel.o"
        fatbin_path = Path(tmpdir) / "kernel.fatbin"
        print(f"Compiling {', '.join(f'sm_{v}' for v in sms)} with {nvcc} ...")
        _run([nvcc, "-dc", "-rdc=true", *targets, str(src), "-o", str(obj_path)])
        print("Device-linking with libcudadevrt ...")
        _run([nvcc, "-dlink", *targets, str(obj_path), "-lcudadevrt", "-fatbin", "-o", str(fatbin_path)])
        return fatbin_path.read_bytes()


def _bytes_array_lines(name: str, data: bytes) -> list[str]:
    lines = [f"static const unsigned char {name}[] = {{"]
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    return lines


def _write_header(
    out_path: Path,
    script_name: str,
    src: Path,
    base_name: str,
    blobs: list[tuple[str, list[int], bytes]],
) -> None:
    src_rel = src.relative_to(REPO_ROOT)
    lines = [
        f"// Auto-generated by scripts/{script_name}",
        "// Do not edit manually. Regenerate with that script.",
        "//",
        f"// Source: {src_rel}",
        "// Bundled fatbins (one per build toolkit, tried in order at load time):",
    ]
    for i, (toolkit, sms, data) in enumerate(blobs):
        arch_list = ", ".join(f"sm_{v}" for v in sms)
        lines.append(f"//   [{i}] CUDA {toolkit}: {arch_list} ({len(data)} bytes)")
    lines += ["", "#pragma once", "", "#include <cstddef>", ""]
    for i, (_toolkit, _sms, data) in enumerate(blobs):
        lines += _bytes_array_lines(f"{base_name}_{i}", data)
        lines.append("")
    lines.append(f"static const unsigned char *const {base_name}s[] = {{")
    for i in range(len(blobs)):
        lines.append(f"    {base_name}_{i},")
    lines.append("};")
    lines.append("")
    lines.append(f"static const std::size_t {base_name}Sizes[] = {{")
    for i in range(len(blobs)):
        lines.append(f"    sizeof({base_name}_{i}),")
    lines.append("};")
    lines.append("")
    lines.append(f"static const std::size_t {base_name}Count = {len(blobs)};")
    lines.append("")
    out_path.write_text("\n".join(lines), newline="\n")


def build_fatbin_header(
    *,
    script_name: str,
    src: Path,
    out_header: Path,
    sm_versions: list[int],
    base_name: str,
) -> None:
    """Build one fatbin per required toolkit and write them all into `out_header` as a C byte-array table.

    Raises `MissingToolkitError` if any required toolkit is unavailable (checked before any compilation).
    """
    groups = group_archs_by_toolkit(sm_versions)
    resolved = resolve_toolkits(groups)
    blobs: list[tuple[str, list[int], bytes]] = []
    for spec, sms in groups:
        nvcc, release = resolved[spec]
        blobs.append((release, sms, _compile_group(nvcc, src, sms)))
    _write_header(out_header, script_name, src, base_name, blobs)
    total = sum(len(data) for _, _, data in blobs)
    summary = "; ".join(f"CUDA {release}: {', '.join(f'sm_{v}' for v in sms)}" for release, sms, _ in blobs)
    print(f"Wrote {out_header} ({len(blobs)} fatbin(s), {total} bytes total) [{summary}]")
