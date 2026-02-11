#!/bin/bash

set -ex

pip install -U pip
pip install --group dev

yum install -y git wget
# Note: following depends on the name of the repo:
git config --global --add safe.directory /__w/gstaichi/gstaichi
git submodule update --init --jobs 2

LLVM_DIR=$(python download_llvm.py | tail -n 1)
export PATH=${LLVM_DIR}/bin:$PATH
chmod +x ${LLVM_DIR}/bin/*
clang --version
which clang

# clang++ searches for libstd++.so, not libstdc++.so.6
# without this, then the compiler checks will fail
# eg:
# - check for working compiler itself
# - and also check for -Wno-unused-but-set-variable, in QuadrantsCXXFlags.cmake
#   which will cause obscure compile errors for external/Eigen
ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so

# since we are linking statically
# and looks like this installs the same version of libstdc++-static as libstdc++
yum install -y libstdc++-static
