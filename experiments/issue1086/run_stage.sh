#!/bin/bash
set -euxo pipefail
workspace=/mnt/data/hugh/ais/handler_arch_1086_fast_math
stage=$1
arch=$2
export TMPDIR=$workspace/tmp
export XDG_CACHE_HOME=$workspace/tmp/cache
export GS_CACHE_FILE_PATH=$workspace/tmp/genesis_cache
export QD_OFFLINE_CACHE=0
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=16
cd "$workspace/quadrants"
if [ "$stage" = reproduce ]; then
    source "$workspace/.venv/bin/activate"
    python experiments/issue1086/run_matrix.py --arch "$arch" --output "$workspace/tmp/reproduction_$arch"
else
    source "$workspace/.venv-genesis/bin/activate"
    if [ "$stage" = replay ]; then
        python experiments/issue1086/run_matrix.py --arch "$arch" --probe genesis_probe --output "$workspace/tmp/replay_$arch" --performance
    elif [ "$stage" = benchmark ]; then
        n_envs=0
        if [ "$arch" = cuda ]; then n_envs=16384; fi
        python experiments/issue1086/run_matrix.py --arch "$arch" --probe genesis_probe --output "$workspace/tmp/benchmark_$arch" --scene anymal --n-envs "$n_envs" --performance --benchmark
    else
        exit 2
    fi
fi
printf 'STAGE_COMPLETE %s %s\n' "$stage" "$arch"
