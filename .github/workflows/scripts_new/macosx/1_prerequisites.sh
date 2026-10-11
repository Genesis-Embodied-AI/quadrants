#!/bin/bash

set -ex

pwd

python --version
sw_vers
clang++ --version
uname -a
clang --version
ls -la
python -V

pip install -U pip
pip install --group dev

pip install numpy

brew install llvm@22

git submodule
git config --global credential.https://gitlab.com.helper \
  '!f() {
    if [ "$1" = get ] && [ -n "$GITLAB_PAT" ]; then
      printf "%s\n" "username=oauth2" "password=$GITLAB_PAT"
    fi
  }; f'
git submodule update --init --recursive

brew install pybind11
which python
which python3
which pip
