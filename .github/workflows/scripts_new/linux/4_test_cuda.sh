#!/bin/bash

set -ex

python tests/run_tests.py -v -r 1 --arch cuda --coverage -m "not needs_torch"

pip install torch --index-url https://download.pytorch.org/whl/cu128
python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append -m needs_torch

python tests/coverage_report.py --collect-only
