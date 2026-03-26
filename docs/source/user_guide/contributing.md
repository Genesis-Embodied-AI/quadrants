# Contributing to quadrants

## Get the right LLVM

`quadrants` relies on `LLVM` to generate machine code. It is obviously a very important dependency which is kept as close as possible to the newest version to support new targets and enjoy new features (new intrinsics, better optimizations, ...).

Although using `LLVM` from your Linux distribution is possible, it is better to be sure you have the right version and rather do one of the following:

* use `download_llvm.py` to get the right binary compiled by us
* build `LLVM` yourself. You have to be cautious about the version (look at README.md or git history, take a look at CI scripts to get it). You also have to enable RTTI when building `LLVM`.

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