#!/bin/bash

set -ex

export QD_FILE_TIMING=1
export QD_FILE_TIMING_OUTPUT="${RUNNER_TEMP}/file_timing.md"

# Which backend(s) this job tests. The CI matrix runs one job per backend (metal / vulkan
# / cpu) in parallel so each job carries ~1/3 of the suite; a single combined run pins all
# three backends onto one runner and is far more prone to the teardown-time run-queue
# oversubscription that has been timing Mac CI out. Defaults to all three when unset (e.g.
# local runs). NB: deliberately NOT named QD_ARCH - that is a reserved quadrants env var
# that overrides qd.init()'s arch via arch_from_name(), which aborts on the alias "cpu".
MAC_TEST_ARCH="${MAC_TEST_ARCH:-metal,vulkan,cpu}"

pip install --prefer-binary --group test
find . -name '*.bc'
ls -lh build/
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
# The C++ tests are backend-agnostic; run them once (on the cpu leg, or on the combined
# default) rather than redundantly on every per-backend leg.
if [ "${MAC_TEST_ARCH}" = "cpu" ] || [ "${MAC_TEST_ARCH}" = "metal,vulkan,cpu" ]; then
  chmod +x ./build/quadrants_cpp_tests
  ./build/quadrants_cpp_tests
fi

# Phase 1: run all tests except torch-dependent ones
python tests/run_tests.py -v -r 1 --arch "${MAC_TEST_ARCH}" -m "not needs_torch"

# Phase 2: install torch, run only torch tests
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
python tests/run_tests.py -v -r 1 --arch "${MAC_TEST_ARCH}" -m needs_torch

if [ -f "$QD_FILE_TIMING_OUTPUT" ]; then
  cat "$QD_FILE_TIMING_OUTPUT" >> "$GITHUB_STEP_SUMMARY"
fi
