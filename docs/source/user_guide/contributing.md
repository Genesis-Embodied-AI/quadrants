# Contributing to quadrants

## Good practice reminder

* *testing*: Any new features or modified code should be tested. You have to run the test suite using `python tests/run_tests.py` which sets up the right test environment for `pytest`. CLI arguments are forwarded to `pytest`. Do not use `pytest` directly as it behaves differently.
* *format/linter*: Before pushing any commits, ensure you set up `pre-commit` and run it using `pre-commit run -a`
* No need to force push to keep a clean history as the merging is eventually done by squashing commits.

## Creating your build/dev environment

It is recommended to use a virtual env. Quadrants supports Python 3.10–3.13 for building and testing. However, `pre-commit` is configured with a pinned Python `3.10` to ensure consistent formatting, so you will need Python 3.10 available for running pre-commit hooks.

After cloning, make sure to initialize submodules:

```
git submodule update --init --recursive
```

`uv` could be handy when initializing such an environment:

```
# create the venv for development
uv venv --python 3.10

# activate it
source .venv/bin/activate

# install deps groups from pyproject.toml
uv pip install --group dev --group test
```

## `build.py`

`build.py` is a python script to automatically set up the build environment for you before invoking the build commands:

* `LLVM libraries`: downloads an archive for `LLVM` libraries, decompresses it and sets `LLVM_DIR`.
* `clang`: depending on the platform, download `clang` or just check if available with the right version.

`build.py` can be used at least two ways:

* `build.py wheel` to build the wheel currently using `setup.py bdist_wheel`
* `build.py --shell` to enter a shell with environment variables set up as with `build.py wheel` in order to let you invoke yourself the commands.

`python setup.py develop` provides incremental builds:

```
./build.py --shell # run a new shell with environment variables
python setup.py develop
```

To write the environment variables to a file, use `./build.py -w [filename]`. For example:

```
./build.py -w env.sh
source env.sh
python setup.py develop
```

## Building the package for release purposes

To build the release package:

```
./build.py wheel
```

We use `cmake` to build the C++ core. The build directory depends on the host architecture and the python version. For example: `_skbuild/linux-x86_64-3.10/cmake-build`.

You can modify the cmake options to your liking in order to enable or disable some features you need or don't need. To discover them, you can use `ccmake`:

```
ccmake _skbuild/linux-x86_64-3.10/cmake-build
```

You could then set the environment variable `QUADRANTS_CMAKE_ARGS` that will be appended to the `cmake` command used to configure the `cmake` build. For instance, to disable the CUDA and AMDGPU backends:

```
export QUADRANTS_CMAKE_ARGS="-DQD_WITH_CUDA=OFF -DQD_WITH_AMDGPU=OFF"
```

To direct `cmake` where to look at for some dependencies, for example `LLVM`, you could either use an environment variable `LLVM_DIR` or specify the cmake option `LLVM_ROOT`:

```
# using an env var
export LLVM_DIR="/path/to/llvm/"
# or with a cmake option
export QUADRANTS_CMAKE_ARGS="$QUADRANTS_CMAKE_ARGS -DLLVM_ROOT=/path/to/llvm"
```

## Advanced usage

### CI Convention about compilers/LLVM

Quadrants comprises at least three important parts:

1. `quadrants` host runtime: Made with a mix of Python and C++. The C++ core is compiled using the OS default C/C++ compiler.
2. `quadrants` device runtime (bitcode): C++ code compiled using `clang++` from the distribution/OS. Using `clang++` is required as it has to support the same targets as `LLVM`.
3. `LLVM` libraries used by host runtime: statically or dynamically linked, used to lower the kernel's final IR to machine code on the host. The CI uses an LLVM version compiled from source.

### Building LLVM for debugging it

Sometimes, it could be useful to have a `LLVM` version that allows to print intermediate passes or with debug symbols to find out where and why LLVM fails (for example, when Instruction Selection fails). To do so you would have to build LLVM by yourself. If so, you should take some inspiration from our [CI pipeline to build LLVM](https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/blob/main/.github/workflows/llvm-ci.yml) to tweak a little bit to your liking (and not enable/disable options that would create discrepancies).

You can then use `LLVM_DIR` to point to the `LLVM` build directory.
