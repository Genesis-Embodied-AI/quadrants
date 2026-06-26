#!/bin/bash

set -ex

export CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
./build.py wheel
