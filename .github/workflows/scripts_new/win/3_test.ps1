$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

python -c 'import gstaichi as ti; ti.init();'
$env:QD_LIB_DIR="python/gstaichi/_lib/runtime"
Get-ChildItem -Path build -Recurse
pip install --group test
pip install -r requirements_test_xdist.txt
pip install torch
python .\tests\run_tests.py -v -r 3
