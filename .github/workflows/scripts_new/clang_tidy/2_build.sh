#!/bin/bash

set -ex

export QUADRANTS_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
./build.py wheel
