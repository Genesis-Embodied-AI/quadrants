# Main versus PR #1076 source-formatting benchmark

This harness compares pinned `main` (`96f31f1083c7d45c2730b708198afb354ef33e16`) with PR #1076 (`2a5e06e69d9efefc1bb31e0434fc918eba717fc8`). The only source differences are three Python files. Both variants use the same compiled core from the main CI wheel.

## Setup

Use Python 3.12 in a dedicated workspace environment. Install the wheel and the repository's test dependency group:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python /path/to/quadrants-1.4.1.dev4+g96f31f108-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl --group pyproject.toml:test
```

The wheel is artifact `manylinux_wheel_cp312_ubuntu-22.04` from GitHub Actions run `42230972357`. Download it with:

```bash
gh run download 42230972357 --repo Genesis-Embodied-AI/quadrants --name manylinux_wheel_cp312_ubuntu-22.04 --dir /path/to/wheel
```

Both pinned commits must be present locally. For a partial clone, fetch their Python blobs before entering an environment without GitHub authentication:

```bash
git archive 96f31f1083c7d45c2730b708198afb354ef33e16 python > /dev/null
git archive 2a5e06e69d9efefc1bb31e0434fc918eba717fc8 python > /dev/null
```

## Run

Run on an allocated compute node with at least eight CPU cores:

```bash
set -o pipefail
.venv/bin/python -u misc/source_formatting_benchmark/run.py \
  --wheel /path/to/quadrants-1.4.1.dev4+g96f31f108-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  --output /path/to/results \
  --pairs 5 2>&1 | tee /path/to/output.log
```

The harness checks that the changed Python files inside the wheel exactly match pinned main. It extracts two package copies, replaces only those files with their pinned versions, and selects the package through `PYTHONPATH`. This retains identical native code and dependencies.

The workload is the existing `classfunc`, `class_method`, `matrix`, `loops`, and `fem99_headless` test files, run through `tests/run_tests.py`. The CPU backend uses eight threads; tests run sequentially. The harness forces both disk compilation caches off even when test fixtures configure them. Normal in-process reuse and source-location memoization remain enabled.

One warm-up pair is excluded. Each measured run uses a new Python process. Pair order alternates main/PR and PR/main. Results include process wall time, test-runner wall time, and time inside outermost `transform_tree` calls. Nested transformations are counted but not double-counted in the AST duration. Instrumentation wraps whole transformations, not individual nodes, and is identical in both variants.

The output directory contains metadata (including wheel SHA-256 and CPU details), full logs for each run, individual JSON measurements, and combined `results.json`. These measurements describe this CPU compilation/test workload. They do not measure Genesis scene-build time or GPU runtime performance. PR #1076 omits some diagnostic source context, so it provides an optimistic comparison for a future lazy formatter, not a universal mathematical upper bound.
