#!/bin/bash

set -ex

pip install torch --index-url https://download.pytorch.org/whl/cu128

python tests/coverage_report.py --collect-only -v -r 1 --arch cuda --with-torch
