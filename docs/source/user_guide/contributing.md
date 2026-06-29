# Contributing to quadrants

## Good practice reminder

* *testing*: Any new features or modified code should be tested. see [unit_testing.md](unit_testing.md)
* *format/linter*: Before pushing any commits, ensure you set up `pre-commit` and run it using `pre-commit run -a`
* No need to force push to keep a clean history as the merging is eventually done by squashing commits.

## Running tests

Run the test suite with `python tests/run_tests.py`. CLI arguments are forwarded to pytest. For example, to run only Metal tests matching a keyword:

```
python tests/run_tests.py --arch metal -k "test_cholesky"
```

The target architecture can also be set via the `QD_WANTED_ARCHS` environment variable (comma-separated, e.g. `QD_WANTED_ARCHS=metal,vulkan`).

### Kernel compilation cache

During test runs, compiled kernels are cached to disk so that the same kernel is not recompiled after each `qd.reset()`/`qd.init()` cycle.

A fresh, empty cache directory is created for each test session by pytest's [`tmp_path_factory`](https://docs.pytest.org/en/stable/how-to/tmp_path.html) (typically under `/tmp/pytest-of-<user>/pytest-<N>/qdcache0/`). Old session directories are cleaned up automatically by pytest's retention policy. This cache is separate from the user-facing `~/.cache/quadrants/` cache.

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

* `build.py wheel` to build the wheel (via [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/), i.e. `pip wheel`)
* `build.py --shell` to enter a shell with environment variables set up as with `build.py wheel` in order to let you invoke yourself the commands.

For incremental development, do an editable install (scikit-build-core "redirect" mode: the compiled core is installed and rebuilt on demand, Python edits are live):

```
./build.py --shell # run a new shell with environment variables
pip install --no-build-isolation -e . -Ceditable.rebuild=true
```

To write the environment variables to a file, use `./build.py -w [filename]`. For example:

```
./build.py -w env.sh
source env.sh
pip install --no-build-isolation -e . -Ceditable.rebuild=true
```

`build.py` exports both the legacy `QUADRANTS_CMAKE_ARGS` and the `CMAKE_ARGS` that scikit-build-core actually reads, so sourcing `env.sh` (or using `--shell`) is enough -- no manual `export CMAKE_ARGS="$QUADRANTS_CMAKE_ARGS"` step is needed.

## Building the package for release purposes

To build the release package:

```
./build.py wheel
```

We use `cmake` to build the C++ core. scikit-build-core puts the CMake build tree under `build/{wheel_tag}`, where the wheel tag encodes the Python version and host platform. For example: `build/cp310-cp310-linux_x86_64`.

You can modify the cmake options to your liking in order to enable or disable some features you need or don't need. To discover them, you can use `ccmake`:

```
ccmake build/cp310-cp310-linux_x86_64
```

You could then set the environment variable `CMAKE_ARGS` (scikit-build-core's CMake-args passthrough) to configure the build. `build.py` also accepts the legacy `QUADRANTS_CMAKE_ARGS` and forwards it to `CMAKE_ARGS`. For instance, to disable the CUDA and AMDGPU backends:

```
export CMAKE_ARGS="-DQD_WITH_CUDA=OFF -DQD_WITH_AMDGPU=OFF"
```

To direct `cmake` where to look at for some dependencies, for example `LLVM`, you could either use an environment variable `LLVM_DIR` or specify the cmake option `LLVM_ROOT`:

```
# using an env var
export LLVM_DIR="/path/to/llvm/"
# or with a cmake option
export QUADRANTS_CMAKE_ARGS="$QUADRANTS_CMAKE_ARGS -DLLVM_ROOT=/path/to/llvm"
```

### Building with the AMD GPU backend (Linux)

The AMD GPU backend is Linux-only (it is force-disabled on macOS and Windows) and is off by default, so enable it explicitly through `CMAKE_ARGS`:

```
./build.py --shell
CMAKE_ARGS="-DQD_WITH_AMDGPU=ON -DQD_WITH_CUDA=OFF" pip install --no-build-isolation -e . -v
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

## CI checks

Pull requests are validated by several CI jobs. Most run automatically; a failing check blocks merge.

### Pre-commit / linters (`linters.yml`)

Runs `pre-commit run -a` which enforces:

- **black** — Python formatting with a 120-character line limit
- **clang-format** — C/C++ formatting
- **trailing-whitespace** and **end-of-file-fixer** — whitespace hygiene
- **ruff** — Python linting
- **pylint** — additional Python linting (scoped to `python/quadrants/`)

You can run these locally with `pre-commit run -a` after `pip install pre-commit`.

### Other CI jobs

- **pyright** (`pyright_linter.yml`) — Python type checking
- **clang-tidy** (`clang_tidy.yml`) — C++ static analysis
- **check-markup-links** (`check_markup_links.yml`) — validates links in documentation
- **linux / macosx / win** — build and test on each platform
- **test-gpu** — GPU-specific tests
- **coverage report** — a one-line diff coverage summary is posted as a PR comment on each push, linking to the full annotated report. This includes kernel-level branch coverage. See [Kernel code coverage](kernel_coverage.md) for details.

### Line wrapping check (`check_wrapping.yml`)

Uses an AI agent to check that lines in changed files follow wrapping conventions:

- **Markdown files (`.md`)**: lines should not be hard-wrapped. Each paragraph should be a single long line.
- **Code comments and docstrings**: lines should be wrapped at 120 characters, not at 80.

The check runs only on lines changed in the PR and reports up to 3 violations. This check is delayed by 30 minutes, to avoid running repeatedly if multiple commits pushed with a short delay between each.

### Deleted comments check (`check_deleted_comments.yml`)

Uses an AI agent to check that comments and docstrings have not been unnecessarily deleted. Reports up to 10 violations. This check is delayed by 30 minutes, to avoid running repeatedly if multiple commits pushed with a short delay between each.

### Test coverage check (`check_test_coverage.yml`)

Uses an AI agent to verify that new or modified source code in a PR has corresponding test coverage. The agent examines the diff of non-test source files and cross-references them against test files in the repo (existing or added in the PR). It flags up to 5 violations. This check is delayed by 30 minutes, to avoid running repeatedly if multiple commits pushed with a short delay between each.

### Feature factorization check (`check_feature_factorization.yml`)

Uses an AI agent to flag feature-specific code being piled into heavily-tracked core files when it could live in its own feature-specific file instead. The concern is not that the new code is in the "wrong" place semantically — it is usually topically related to the host file — but that the host file is already a hot, central, frequently-edited file, and adding more self-contained feature code to it makes review, merge conflicts, and future churn worse. The fix is almost always to extract the feature-specific block (top-level function, class, large block, or even a cluster of new methods on an existing class) into its own module, with the host file delegating to it via a narrow interface.

The agent reports up to 5 violations, each annotated with the host file's hotness numbers (commits / authors / size). This check is delayed by 30 minutes, to avoid running repeatedly if multiple commits pushed with a short delay between each.

### Doc quality check (`check_doc_quality.yml`)

Uses an AI agent to review documentation changes for an end-user audience (someone writing Quadrants kernels in Python, not a compiler engineer).


Let's first define an *advanced / internal section* as a clearly-marked section whose heading contains "Advanced", "Under the hood", "Internals", or "Implementation"; it targets a more advanced reader rather than the typical end user.


For each `docs/**/*.md` file added or modified in the PR, the CI agent reads the entire current file (not just the diff) and checks three things:
- (1) **undefined terms** — a term a typical user is unlikely to know (specialized or internal jargon, project-specific abbreviations) must be defined at its first use in that file, either inline or via a link to a doc that defines it, unless that first use is inside an advanced / internal section;
- (2) **end-user relevance** — internal / implementation / contributor-only material must be confined to an advanced / internal section;
- (3) **reading order** — information must be ordered so a first-time reader can follow the file top-to-bottom in one pass, without forward references (a passage that can only be understood by reading something introduced later in the same file).

The following do not count as violations: references to public APIs that the author links to their docs and/or labels as public; brief reader-directed pointers suggesting the reader could contribute upstream or file an issue; the core public API vocabulary a user already knows (e.g. `@qd.kernel`, `@qd.func`, `qd.Template`, fields, ndarrays); and, for the reading-order check, an overview/roadmap near the top, optional "see below" pointers (where the current passage still reads fine on its own), and backward references.

The agent reports up to 10 violations. This check is delayed by 30 minutes, to avoid running repeatedly if multiple commits pushed with a short delay between each.

### PR change report (`pr_change_report.yml`)

Posts a fresh PR comment on every push. The comment is a single line: the totals (file count, code lines added, code lines removed) formatted as a markdown link to a GitHub Check whose page contains the full per-file / per-function breakdown. "Code lines" exclude blank lines, comment-only lines, and (in Python) lines whose only token content is a string literal (i.e. docstrings and continuation lines of multi-line strings). C/C++ `/* … */` block comments are stripped before counting.

The number columns on the Check page (without a `+` or `-` sign) are code-line counts in the BASE (pre-PR) version: file size before this PR (0 for newly-added files), function body size before this PR (0 for new functions; original body size for deleted functions). `+<n>` / `-<n>` are code lines added / removed by this PR.

Files are sorted by added lines descending. Within each file, functions are split into a `New:` group (added by this PR), an `Existing:` group (modified by this PR), and a `Deleted:` group (removed by this PR), and within each group sorted by added lines descending, then removed lines descending. Files that are deleted in their entirety appear as a single per-file row (so the totals stay accurate) but skip the per-function breakdown. Sample shape:

```
quadrants/program/program_stream.cpp 0 +151
    New:
      StreamManager::create_event()             0     +18
      StreamManager::create_stream()            0     +18
      StreamManager::record_event()             0     +15
      StreamManager::destroy_event()            0     +13
      StreamManager::destroy_stream()           0     +13

python/quadrants/lang/stream.py 0 +111
    New:
      Event.destroy()             0      +9
      Stream.destroy()            0      +9
      Event._destroy_prog()       0      +8
      Stream._destroy_prog()      0      +8
      Event.__del__()             0      +7

quadrants/program/legacy_stream.cpp 42 -42
    # entire file deleted (per-function breakdown skipped)
```

The `0` in the LoC column for the two new files reflects that both files did not exist before this PR (their pre-PR code-line count is 0). The `42 -42` row for `legacy_stream.cpp` is a fully-deleted file: 42 code lines existed before this PR and all 42 were removed.

This check is delayed by 30 minutes, to avoid running repeatedly if multiple commits pushed with a short delay between each.
