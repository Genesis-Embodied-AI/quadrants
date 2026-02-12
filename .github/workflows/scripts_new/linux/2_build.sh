#!/bin/bash

set -ex

# this was already downloaded in 1_prerequisites.sh, so this is just to set the env var
LLVM_DIR=$(python download_llvm.py | tail -n 1)
export PATH=${LLVM_DIR}/bin:$PATH
which clang
clang --version

export QUADRANTS_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_AMDGPU:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_BUILD_TESTS:BOOL=ON"
./build.py wheel
