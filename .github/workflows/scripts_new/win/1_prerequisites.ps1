$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

pip install -U pip
pip install --group dev

# NOTE: We no longer install Visual Studio Build Tools here. build.py (run in 2_build.ps1)
# discovers the Visual Studio already present on the runner via vswhere and builds against it
# (Ninja + VS developer shell), which avoids the ~15 min Build Tools download. If a future image
# ships no Visual Studio at all, build.py will install Build Tools and exit non-zero in 2_build.
