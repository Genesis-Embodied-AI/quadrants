#!/bin/bash

set -ex

pip install dist/*.whl
python -c "import quadrants as ti; ti.init(arch=ti.cpu)"
python -c "import quadrants as ti; ti.init(arch=ti.metal)"
