#!/bin/bash
# Install host packages and Python dev dependencies needed to build quadrants
# with the AMDGPU backend enabled and to run clang-tidy on the resulting
# compile_commands.json.

set -ex

pip install -U pip
pip install --group dev
pip install scikit-build

sudo apt-get update
sudo apt-get install -y \
    clang-tidy-14 \
    pybind11-dev \
    libc++-15-dev \
    libc++abi-15-dev \
    clang-15 \
    libclang-common-15-dev \
    libclang-cpp15 \
    libclang1-15 \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip
