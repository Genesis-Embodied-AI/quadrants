#!/bin/bash
# Configure quadrants for the AMDGPU backend so that compile_commands.json
# captures the AMDGPU TUs, but do *not* compile anything. clang-tidy only
# needs the compile database; skipping the build is a large CI-time win.
#
# We deliberately disable the CUDA, Vulkan, Metal, and Python backends:
# the AMDGPU TUs (codegen/amdgpu, runtime/amdgpu, rhi/amdgpu, platform/amdgpu)
# transitively include LLVM headers but do not include CUDA/Vulkan/Metal/
# pybind headers, so this is sufficient to lint them. If a future AMDGPU
# source ever pulls in another backend, this workflow will fail loudly,
# which is the desired signal.

set -ex

export QUADRANTS_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DQD_WITH_LLVM=ON \
    -DQD_WITH_AMDGPU=ON \
    -DQD_WITH_CUDA=OFF \
    -DQD_WITH_VULKAN=OFF \
    -DQD_WITH_METAL=OFF \
    -DQD_WITH_PYTHON=OFF \
    -DQD_BUILD_TESTS=OFF"

./build.py configure
