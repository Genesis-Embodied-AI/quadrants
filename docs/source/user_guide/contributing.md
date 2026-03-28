# Contributing to quadrants

## Good practice reminder

* *testing*: Any new features or modified code should be tested. You have to run the test suite using `./tests/run_tests.py` which set up
  the right test environment for `pytest`. CLI arguments are forwarded to `pytest`. Do not use `pytest` directly as it behaves differently.
* *format/linter*: Before pushing any commits, ensure you set up `pre-commit` and run it using `pre-commit run -a`
* No need to force push to keep a clean history as the merging is eventually done by squashing commits.

## Getting the right python version with an isolated dependency tree

It is recommended to use a virtual env and the right python version (currently `3.10`). `pre-commit` for instance is configured with a pinned Python version.

To do so, you could use `uv`:

```
uv venv --python 3.10
source .venv/bin/activate

uv pip install --group dev test
```

## `build.py`

`build.py` is a python script to automatically set up the build environment for you before invoking the build commands:

* `LLVM libraries`: downloads an archive for `LLVM` libraires, decompress it and set `LLVM_DIR`.
* `clang`: depending on the platform, download `clang` or just check if available with right version.

`build.py` can be used at least two ways:

* `build.py wheel` to build the wheel currently using `setup.py bdist_wheel`
* `build.py --shell` to enter a shell with environment variables set up as with `build.py wheel` in order to let you invoke yourself the commands.

For instance, if you want to tinker the `quadrants` python runtime without rebuilding every time, you
can use `setup.py develop` by yourself.

```
./build.py --shell # run a new shell with environment variables
python setup.py develop
```

If you are interested about the environment variables set up by `build.py`
(or you need them for one of your script), you can write them down to a file
to source it by yourself:

```
./build.py -w env.sh
source env.sh
python setup.py develop
```

## Building the package for release purposes

As previously mentionned, to build the release package, you can invoke:

```
./build.py wheel
```

We use `cmake` to build the C++ core. The build directory depends on the host architecture and the python version: it could be for instance `_skbuild/linux-x86_64-3.10/cmake-build`.

You can modify the cmake variables to your liking in order to enable or disable some features you need or don't need. 

## Advanced usage

### CI Convention about compilers/LLVM

Quadrants comprises at least three important parts:

1. `quadrants` host runtime: Made with a mix of Python and C++. The C++ core is compiled using the OS default C/C++ compiler. 
2. `quadrants` device runtime (bitcode): C++ code compiled using `clang++` from the distribution/OS. Using `clang` is required as it has to support the same targets as `LLVM` (obviously!).
3. `LLVM` libraries used by host runtime: statically or dynamically linked, used to lower the kernel's final IR to machine code on the host. The CI uses an LLVM version compiled from source.

### Building LLVM for debugging it

Sometimes, it could be useful to have a LLVM version that allows to print intermediate passes or with debug symbols to find out where and why LLVM fails (for example, when Instruction Selection fails).
To do so you would have to build LLVM by yourself. If so, you should take some inspiration from our [CI pipeline to build LLVM](https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/blob/main/.github/workflows/llvm-ci.yml) to tweak a little bit to your liking (and not enable/disable options that would create discrepancies).

You can then use `LLVM_DIR` to point to the `LLVM` build directory.
