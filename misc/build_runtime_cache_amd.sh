#!/usr/bin/env bash
# Native incremental build on the SSH AMD host, using its already provisioned LLVM 22 SDK.
set -euxo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(dirname "$repo_root")"
export PATH="$HOME/.local/bin:$PATH"
export LLVM_DIR="$HOME/.cache/qd-build-cache/llvm-22.1.0-x86-202603120808"
export VULKAN_SDK="$HOME/.cache/qd-build-cache/vulkan-1.4.321.1/x86_64"
export PATH="$VULKAN_SDK/bin:$PATH"
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$workspace/tmp"
export UV_CACHE_DIR="$workspace/tmp/uv-cache"
export UV_PYTHON_INSTALL_DIR="$workspace/tmp/python"
export PIP_CACHE_DIR="$workspace/tmp/pip-cache"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export CMAKE_ARGS="-DQD_WITH_AMDGPU=ON -DQD_WITH_CUDA=OFF -DQD_WITH_VULKAN=OFF -DQD_WITH_GGUI=OFF -DQD_BUILD_TESTS=OFF -DCLANG_EXECUTABLE=$HOME/.local/bin/clang"
mkdir -p "$TMPDIR"
cd "$repo_root"
git rev-parse HEAD
"$HOME/.local/bin/clang" --version
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.10 --seed .venv
fi
source .venv/bin/activate
if [ "${1:-}" != --test-only ]; then
  uv pip install --group dev --group test
  python -m pip install -v --no-build-isolation -e . -Ceditable.rebuild=true
fi
python tests/run_tests.py amdgpu_runtime_cache --arch amdgpu -t 1 -v
python misc/profile_runtime_hsaco_cache.py --output "$TMPDIR/cache-toggle" 2>&1 | tee "$TMPDIR/cache-toggle.log"
python misc/profile_ci_startup.py --arch amdgpu --output "$TMPDIR/startup" 2>&1 | tee "$TMPDIR/startup.log"
python tests/run_tests.py basics assign scalar_op amdgpu_context_cache --arch amdgpu -t 4 -v 2>&1 | tee "$TMPDIR/subset.log"
