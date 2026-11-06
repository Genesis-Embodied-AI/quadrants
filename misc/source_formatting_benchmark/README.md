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

## Results: 2026-11-06

Ran on cluster node `h100-206-029` in `genesis-v1_25.sqsh`, with an eight-core CPU allocation and 16 GiB allocated RAM. No GPU was used. Python was 3.12.14. The harness revision was `c6c1bda47`; exact compared source revisions and wheel hash are in `results/metadata.json`. Both variants used the same main wheel and dependencies.

Five alternating measured pairs followed one excluded warm-up pair. Every run passed all 141 tests (19 warnings), with 448 outer and 678 total AST transformations. The initial pilot is excluded from these results.

| Metric | Main median | PR #1076 median | Reduction of medians |
| --- | ---: | ---: | ---: |
| Whole process | 40.188880 s | 39.946931 s | 0.602% |
| Test runner | 39.444314 s | 39.230723 s | 0.542% |
| Outermost AST transformation | 1.013755 s | 0.967273 s | 4.585% |

The median paired AST saving was 0.048088 s (4.736%), with savings of 0.035802 to 0.049547 s across all five pairs. Whole-process differences were less consistent: the PR was faster in three of five pairs, with paired savings ranging from -0.349625 s to +1.043239 s. These runs show a consistent reduction in AST time but do not establish a reliable whole-process speedup. The median paired AST saving is about 0.12% of the main process median.

This is a CPU compilation/test workload, not a Genesis scene-build benchmark. No conclusions about Genesis-wide speedups or GPU workloads follow from these results. The comparison is optimistic for a future lazy formatter because #1076 omits source context rather than preserving it lazily; it is not a strict upper bound on every possible implementation.

Raw measured timings, metadata, and dependency versions are committed under `results/`. Recompute the summary and CSV with:

```bash
.venv/bin/python misc/source_formatting_benchmark/summarize.py misc/source_formatting_benchmark/results/results.json
```

Full run logs are retained at `/mnt/data/hugh/ais/ext_prs_940/tmp/measurement/` on the cluster and `/home/hugh/ais/ext_prs_940/tmp/benchmark-results/` on the 5090 host. The combined log is `/home/hugh/ais/ext_prs_940/tmp/output.log`; an archive of the individual logs and JSON files is `/home/hugh/ais/ext_prs_940/tmp/benchmark-results.tar.gz`.
