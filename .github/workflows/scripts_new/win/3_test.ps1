$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

python -c 'import quadrants as qd; qd.init();'
$env:QD_LIB_DIR="python/quadrants/_lib/runtime"
Get-ChildItem -Path build -Recurse
pip install --group test

# Phase 1: run all tests except torch-dependent ones
python .\tests\run_tests.py -v -r 1 -m "not needs_torch"

# Phase 2: install torch, run only torch tests
pip install torch
python .\tests\run_tests.py -v -r 1 -m needs_torch
