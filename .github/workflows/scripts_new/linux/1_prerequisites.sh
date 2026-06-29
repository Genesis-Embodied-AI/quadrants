#!/bin/bash

set -ex

pwd
uname -a
git status
free -m
cat /etc/lsb-release
ls -la
python -V

pip install -U pip
pip install --group dev

LLVM_DIR=$(python download_llvm.py | tail -n 1)
export PATH=${LLVM_DIR}/bin:$PATH
chmod +x ${LLVM_DIR}/bin/*
clang --version
which clang

python -c 'import platform; u = platform.uname(); print("u.system", u.system, "u.machine", u.machine)'

git submodule
git submodule update --init --recursive
sudo apt update
sudo apt install -y \
    cmake \
    ninja-build
