#!/bin/bash
set -euxo pipefail
workspace=/mnt/data/hugh/ais/handler_arch_1086_fast_math
export PATH=/mnt/data/hugh/.local/bin:$PATH
export TMPDIR=$workspace/tmp
export XDG_CACHE_HOME=$workspace/tmp/cache
source "$workspace/perso_hugh/ai/scripts/python_cache.sh"
cd "$workspace/quadrants"
git submodule update --init --recursive
if [ ! -d "$workspace/.venv" ]; then
    uv venv --python 3.12 "$workspace/.venv"
fi
source "$workspace/.venv/bin/activate"
uv pip install --group dev --group test pip
export LLVM_DIR=/mnt/data/hugh/.cache/qd-build-cache/llvm-22.1.0-x86-202603120808
# The shared SDK's clang currently lacks execute permission; use a private executable copy.
mkdir -p "$workspace/tmp/toolchain/bin" "$workspace/tmp/toolchain/lib"
cp -u "$LLVM_DIR/bin/clang" "$workspace/tmp/toolchain/bin/clang"
chmod +x "$workspace/tmp/toolchain/bin/clang"
ln -sfn "$LLVM_DIR/lib/clang" "$workspace/tmp/toolchain/lib/clang"
"$workspace/tmp/toolchain/bin/clang" --version
export CMAKE_ARGS="-DQD_GLSLANG_EXECUTABLE=/mnt/data/hugh/.cache/qd-build-cache/vulkan-1.4.321.1/x86_64/bin/glslang -DCLANG_EXECUTABLE=$workspace/tmp/toolchain/bin/clang -DQD_WITH_VULKAN=OFF -DQD_WITH_OPENGL=OFF -DQD_WITH_AMDGPU=OFF -DQD_WITH_CUDA=ON"
export CMAKE_BUILD_PARALLEL_LEVEL=16
python -m pip install --no-build-isolation -e . -v
printf 'BUILD_COMPLETE\n'
