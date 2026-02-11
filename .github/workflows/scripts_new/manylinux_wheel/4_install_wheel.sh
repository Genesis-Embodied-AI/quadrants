#!/bin/bash

set -ex

pip install dist/*.whl
python -c 'import quadrants as ti; ti.init(arch=ti.cpu); print(ti.__version__)'
