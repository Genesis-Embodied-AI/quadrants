#!/bin/bash

set -ex

set -ex

pip install -U pip
pip install --group dev

sudo apt-get update
sudo apt-get install -y clang-tidy-14
git submodule update --init --recursive

sudo apt install -y \
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
