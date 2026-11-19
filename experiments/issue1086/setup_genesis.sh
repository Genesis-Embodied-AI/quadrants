#!/bin/bash
set -euxo pipefail
workspace=/mnt/data/hugh/ais/handler_arch_1086_fast_math
export PATH=/mnt/data/hugh/.local/bin:$PATH
export TMPDIR=$workspace/tmp
export XDG_CACHE_HOME=$workspace/tmp/cache
source "$workspace/perso_hugh/ai/scripts/python_cache.sh"
cd "$workspace/quadrants"
source "$workspace/.venv/bin/activate"
export LLVM_DIR=/mnt/data/hugh/.cache/qd-build-cache/llvm-22.1.0-x86-202603120808
export CMAKE_ARGS="-DCLANG_EXECUTABLE=$LLVM_DIR/bin/clang -DQD_WITH_VULKAN=OFF -DQD_WITH_OPENGL=OFF -DQD_WITH_AMDGPU=OFF -DQD_WITH_CUDA=ON"
export CMAKE_BUILD_PARALLEL_LEVEL=16
python -m pip wheel --no-deps --no-build-isolation -w "$workspace/tmp/wheels" .
cd "$workspace/genesis"
# Keep CI's dependency recipe, relocating its venv into the assigned workspace.
sed "s|/venv|$workspace/.venv-genesis|g" .github/workflows/scripts/production_build.sh > "$workspace/tmp/production_build_workspace.sh"
bash "$workspace/tmp/production_build_workspace.sh"
source "$workspace/.venv-genesis/bin/activate"
uv pip install --no-deps -e .
uv pip install --no-deps --reinstall "$workspace"/tmp/wheels/quadrants-*.whl
uv pip freeze > "$workspace/tmp/genesis_packages.txt"
printf 'GENESIS_SETUP_COMPLETE\n'
