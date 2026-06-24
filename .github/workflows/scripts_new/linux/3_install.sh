#!/bin/bash

set -ex

pip3 install dist/*.whl
python -c "import quadrants as qd; qd.init(arch=qd.cpu)"
