#!/bin/bash

set -ex

pip install dist/*.whl
python -c 'import quadrants as qd; qd.init(arch=qd.cpu); print(qd.__version__)'
