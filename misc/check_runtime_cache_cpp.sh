#!/usr/bin/env bash
# Standalone cache tests against the same LLVM SDK as the wheel, without building the full Python extension.
set -euxo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${LLVM_ROOT:?Set LLVM_ROOT to the LLVM 22 SDK directory}"
: "${CACHE_TEST_OUTPUT:?Set CACHE_TEST_OUTPUT to a writable directory}"
mkdir -p "$CACHE_TEST_OUTPUT"
cd "$repo_root"
git rev-parse HEAD
${CXX:-g++} $("$LLVM_ROOT/bin/llvm-config" --cxxflags) -fexceptions -pthread \
  -I "$repo_root" -I external/googletest/googletest/include -I external/googletest/googletest \
  tests/cpp/common/runtime_hsaco_cache_test.cpp tests/cpp/llvm/runtime_hsaco_cache_key_test.cpp \
  external/googletest/googletest/src/gtest-all.cc external/googletest/googletest/src/gtest_main.cc \
  $("$LLVM_ROOT/bin/llvm-config" --ldflags --libs bitwriter core support --system-libs) \
  -o "$CACHE_TEST_OUTPUT/runtime_cache_tests"
"$CACHE_TEST_OUTPUT/runtime_cache_tests" --gtest_filter='RuntimeHsacoCache.*'
