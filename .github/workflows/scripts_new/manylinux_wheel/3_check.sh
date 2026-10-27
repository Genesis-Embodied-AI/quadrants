#!/bin/bash

set -ex

pip install auditwheel
auditwheel show dist/*.whl
auditwheel repair --plat "manylinux_2_35_$(uname -m)" --only-plat dist/*.whl
pwd
ls -lh wheelhouse/
auditwheel show wheelhouse/*.whl
