# -*- coding: utf-8 -*-

# -- stdlib --
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from os.path import join
from pathlib import Path

# -- third party --
# -- own --
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, error, get_cache_home, warn
from .tinysh import powershell


def grep(value, target):
    for line in target.split("\n"):
        if value in line:
            return line


# -- code --
@banner("Setup Clang")
def setup_clang(as_compiler=True) -> None:
    """
    Setup Clang.
    """
    u = platform.uname()
    if u.system == "Linux":
        for v in ("-22", "-21", "-20", ""):
            clang = shutil.which(f"clang{v}")
            clangpp = shutil.which(f"clang++{v}")
            if clang is not None and clangpp is not None:
                break
        else:
            error("Could not find clang of any version")
            return
    elif u.system == "Darwin":
        brew_config = subprocess.check_output(["brew", "config"]).decode("utf-8")
        print("brew_config", brew_config)
        brew_prefix = grep("HOMEBREW_PREFIX", brew_config).split()[1]
        print("brew_prefix", brew_prefix)
        clang = join(brew_prefix, "opt", "llvm@22", "bin", "clang")
        clangpp = join(brew_prefix, "opt", "llvm@22", "bin", "clang++")
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        out = get_cache_home() / "clang-22-1-0"
        url = "https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/releases/download/llvm-22.1.0-202603120808/taichi-llvm-22.1.0-windows-amd64.zip"
        download_dep(url, out, force=True)
        clang = str(out / "bin" / "clang++.exe").replace("\\", "\\\\")
        clangpp = clang
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    cmake_args["CLANG_EXECUTABLE"] = clang
    cmake_args["CMAKE_CXX_COMPILER_WORKS"] = "1"

    if as_compiler:
        cc = os.environ.get("CC")
        cxx = os.environ.get("CXX")
        if cc:
            warn(f"Explicitly specified compiler via environment variable CC={cc}, not configuring clang.")
        else:
            cmake_args["CMAKE_C_COMPILER"] = clang

        if cxx:
            warn(f"Explicitly specified compiler via environment variable CXX={cxx}, not configuring clang++.")
        else:
            cmake_args["CMAKE_CXX_COMPILER"] = clangpp


ENV_EXTRACT_SCRIPT = """
param ([string]$DevShell, [string]$VsPath, [string]$OutFile, [string]$DevCmdArguments)
$WarningPreference = 'SilentlyContinue'
Import-Module $DevShell
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments $DevCmdArguments
Get-ChildItem env:* | ConvertTo-Json -Depth 1 | Out-File $OutFile
"""


def _vs_devshell(vs, dev_cmd_arguments="-arch=x64"):
    dll = vs / "Common7" / "Tools" / "Microsoft.VisualStudio.DevShell.dll"

    if not dll.exists():
        error("Could not find Visual Studio DevShell")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "extract.ps1"
        with script.open("w") as f:
            f.write(ENV_EXTRACT_SCRIPT)
        outfile = tmp / "env.json"
        powershell(
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-DevShell",
            str(dll),
            "-VsPath",
            str(vs),
            "-OutFile",
            str(outfile),
            "-DevCmdArguments",
            dev_cmd_arguments,
        )
        with outfile.open(encoding="utf-16") as f:
            envs = json.load(f)

    for v in envs:
        os.environ[v["Key"]] = v["Value"]


def _vswhere_latest_vs():
    """Locate the newest installed Visual Studio via vswhere, regardless of edition/version.

    Returns the installation Path, or None if vswhere is missing or finds nothing.
    """
    pf86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    try:
        out = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-prerelease",
                "-products",
                "*",
                "-property",
                "installationPath",
            ],
            encoding="utf-8",
            errors="ignore",
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if not out:
        return None
    vs = Path(out.splitlines()[0].strip())
    return vs if vs.exists() else None


def _pick_vcvars_ver(vs):
    """Pick a -vcvars_ver toolset prefix to pin, if available in this VS install.

    Defaults to v143 (14.4x) so we keep a known-good compiler even when the host VS ships a
    newer default (e.g. VS 2026 defaults to v145 / 14.51). Set QUADRANTS_VCVARS_VER='' to opt
    out (use the VS default), or to another prefix (e.g. '14.51') to pin something else.
    """
    want = os.environ.get("QUADRANTS_VCVARS_VER", "14.44")
    if not want:
        return None
    msvc_dir = vs / "VC" / "Tools" / "MSVC"
    if msvc_dir.exists():
        for d in sorted(msvc_dir.iterdir()):
            if d.name.startswith(want):
                return want
    return None


@banner("Setup MSVC")
def setup_msvc() -> None:
    assert platform.system() == "Windows"

    base = Path("C:\\Program Files (x86)\\Microsoft Visual Studio")

    # Prefer whatever Visual Studio is already installed (discovered via vswhere). This avoids a
    # ~15 min Build Tools download on CI images that ship VS but not under the legacy
    # "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022" path (e.g. windows-2025-vs2026,
    # which has VS 2026 Enterprise under "C:\\Program Files\\Microsoft Visual Studio\\18").
    vs = _vswhere_latest_vs()

    if vs is None:
        # Legacy fallback: probe the well-known VS 2022 locations directly.
        for ver in ("2022",):
            for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
                cand = base / ver / edition
                if cand.exists():
                    vs = cand
                    break
            if vs is not None:
                break

    if vs is not None:
        if os.environ.get("QUADRANTS_USE_MSBUILD"):
            # Caller explicitly requested the MSBuild / Visual Studio generator path.
            return

        # Ninja + MSVC (cl.exe) via the VS developer shell. This is generator-agnostic, so it
        # works regardless of whether the bundled CMake knows the host VS's generator, and gives
        # better sccache behaviour. Pin the toolset (default v143/14.44) for reproducibility.
        dev_cmd_arguments = "-arch=x64"
        vcvars_ver = _pick_vcvars_ver(vs)
        if vcvars_ver:
            dev_cmd_arguments += f" -vcvars_ver={vcvars_ver}"
        _vs_devshell(vs, dev_cmd_arguments=dev_cmd_arguments)
        cmake_args["CMAKE_C_COMPILER"] = "cl.exe"
        cmake_args["CMAKE_CXX_COMPILER"] = "cl.exe"
        return

    # No Visual Studio found anywhere: install Build Tools, then ask the user to re-run.
    url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
    out = base / "2022" / "BuildTools"
    download_dep(
        url,
        out,
        elevate=True,
        args=[
            "--passive",
            "--wait",
            "--norestart",
            "--includeRecommended",
            "--add",
            "Microsoft.VisualStudio.Workload.VCTools",
            # NOTE: We are using the custom built Clang++,
            #       so components below are not necessary anymore.
            # '--add',
            # 'Microsoft.VisualStudio.Component.VC.Llvm.Clang',
            # '--add',
            # 'Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Llvm.Clang',
            # '--add',
            # 'Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset',
        ],
    )
    warn("Please restart build.py after Visual Studio Build Tools is installed.")
    sys.exit(1)
