#!/usr/bin/env bash
# Native incremental build on the SSH AMD host, using its already provisioned LLVM 22 SDK.
set -euxo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(dirname "$repo_root")"
export PATH="$HOME/.local/bin:$PATH"
export LLVM_DIR="$HOME/.cache/qd-build-cache/llvm-22.1.0-x86-202603120808"
export VULKAN_SDK="$HOME/.cache/qd-build-cache/vulkan-1.4.321.1/x86_64"
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$workspace/tmp"
export UV_CACHE_DIR="$workspace/tmp/uv-cache"
export UV_PYTHON_INSTALL_DIR="$workspace/tmp/python"
export PIP_CACHE_DIR="$workspace/tmp/pip-cache"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export CMAKE_ARGS="-DQD_WITH_AMDGPU=ON -DQD_WITH_CUDA=OFF -DQD_WITH_VULKAN=OFF -DQD_WITH_GGUI=OFF -DQD_BUILD_TESTS=ON -DCLANG_EXECUTABLE=$HOME/.local/bin/clang"
mkdir -p "$TMPDIR"
cd "$repo_root"
git rev-parse HEAD
"$HOME/.local/bin/clang" --version
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.10 --seed .venv
fi
source .venv/bin/activate
uv pip install --group dev --group test
python -m pip install -v --no-build-isolation -e . -Ceditable.rebuild=true
build/quadrants_cpp_tests --gtest_filter='RuntimeHsacoCache.*'
python tests/run_tests.py amdgpu_runtime_cache --arch amdgpu -t 1 -v
