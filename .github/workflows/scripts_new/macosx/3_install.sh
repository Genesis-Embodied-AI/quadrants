#!/bin/bash

set -ex

pip install dist/*.whl
python -c "import quadrants as qd; qd.init(arch=qd.cpu)"
python -c "import quadrants as qd; qd.init(arch=qd.metal)"
