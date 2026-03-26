# Contributing to quadrants

## Fetching LLVM binaries

`quadrants` relies on `LLVM` to generate machine code. It is obviously a very important dependency which is kept as close as possible to the newest version to support new targets and enjoy new features (new intrinsics, better optimizations, ...).

Although using `LLVM` from your Linux distribution is possible, it is better to be sure you have the right version.
In order to do so, use `download_llvm.py` to get the right binary compiled by us.

Once it is done, you could use the env variable `LLVM_DIR` to specify where the root of the LLVM binaries are.

## Building the package

There are at least two ways to build the wheel:

* `setup.py bdist_wheel`: The method we currently use
* Using a package manager such as `pip` (`pip wheel`) or `uv` (`uv build`): Currently not supported, but possible if you disable build isolation.

The important environment variables are:

* `LLVM_DIR`: root directory for `LLVM` binaries
* `QUADRANTS_CMAKE_ARGS`: extra arguments to `cmake`

Concerning the extra cmake arguments, the notable ones are:

* `CMAKE_CXX_COMPILER`: it should be `clang++`
* `QD_WITH_CUDA` / `QD_WITH_AMDGPU`: If you want to enable/disable a target because you don't need it.

## Notes

* *testing*: Any new features or modified code should be tested. You have to run the test suite using `./tests/run_tests.py` which set up
  the right test environment for `pytest`. CLI arguments are forwarded to `pytest`. Do not use `pytest` directly as it behaves differently.
* *format/linter*: Before pushing any commits, ensure you set up `pre-commit` and run it using `pre-commit run -a`
* No need to force push to keep a clean history as the merging is eventually done by squashing commits.

## Advanced users

### Building LLVM for debugging it

Sometimes, it could be useful to have a LLVM version that allows to print intermediate passes or with debug symbols to find out where and why LLVM fails (for example, when Instruction Selection fails).
To do so you would have to build LLVM by yourself. If so, you should take some inspiration from our [CI pipeline to build LLVM](https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/blob/main/.github/workflows/llvm-ci.yml) to tweak a little bit to your liking (and not enable/disable options that would create discrepancies).
