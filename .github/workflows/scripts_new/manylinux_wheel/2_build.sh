#!/bin/bash

set -ex

PLATFORM=$(uname -m)
if [[ ! "$PLATFORM" =~ ^(x86_64|aarch64)$ ]]; then
    echo "Unsupported architecture: $PLATFORM"
    exit 1
fi

# this was already downloaded in 1_prerequisites.sh, so this is just to set the env var
LLVM_DIR=$(python download_llvm.py | tail -n 1)
export PATH=${LLVM_DIR}/bin:$PATH
which clang
clang --version

echo "Detected platform: $PLATFORM"
# Add Taichi LLVM toolchain to PATH
export PATH="$PWD/taichi-llvm-15.0.7-linux-${PLATFORM}/bin:$PATH"

# Taichi build options
export QUADRANTS_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_AMDGPU:BOOL=ON -DTI_BUILD_TESTS:BOOL=ON"

# GCC toolset include paths
inc_base="/opt/rh/gcc-toolset-14/root/usr/include/c++/14"
extra="$inc_base:$inc_base/${PLATFORM}-redhat-linux:$inc_base/backward"

export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:+$CPLUS_INCLUDE_PATH:}$extra"

./build.py wheel
