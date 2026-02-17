$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

$env:GSTAICHI_CMAKE_ARGS = "-DQD_WITH_VULKAN:BOOL=ON -DQD_WITH_AMDGPU:BOOL=ON -DQD_WITH_CUDA:BOOL=ON -DQD_BUILD_TESTS:BOOL=ON"
python build.py
