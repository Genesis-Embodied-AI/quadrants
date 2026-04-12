# Kernel code coverage

Quadrants can measure which lines of your kernel code actually execute on the device (CPU or GPU). This goes beyond
standard Python coverage tools, which can only see host-side code — kernel coverage tracks execution *inside* compiled
kernels, including which branches of `if`/`else` blocks are taken at runtime.

The coverage data is written in the standard `coverage.py` format, so it integrates with familiar tools like
`coverage report`, `diff-cover`, and IDE coverage viewers.

## Quick start

### 1. Run tests with coverage

Use the built-in `run_tests.py` script with the `-C` flag:

```bash
python tests/run_tests.py -C -v
```

This sets `QD_KERNEL_COVERAGE=1` and enables `pytest-cov` with branch coverage automatically.

### 2. Generate a report

After the test run, combine the Python and kernel coverage data and produce a report:

```bash
python tests/coverage_report.py
```

By default this generates an HTML diff coverage report (`coverage-report.html`) comparing your current branch against
`origin/main`. Open it in a browser to see which changed lines are covered.

## How it works

When `QD_KERNEL_COVERAGE=1` is set, Quadrants rewrites the Python AST of each kernel and `@qd.func` before
compilation. It inserts lightweight probe statements (`field[probe_id] = 1`) at each source line. These probes compile
as ordinary field stores and execute on the device alongside your kernel code.

At process exit, the probe data is read back and written to a `.coverage` file. If `pytest-cov` also wrote Python
coverage data, the two are combined so that the final report includes both host-side and kernel-side coverage.

Key properties:
- **Zero overhead when disabled.** The module is never imported unless `QD_KERNEL_COVERAGE=1` is set.
- **Branch coverage.** Probes inside `if`/`else` bodies only fire when that branch is taken, giving true runtime
  branch coverage.
- **Works with pytest-xdist.** Each worker writes to a separate coverage file; these are merged during report
  generation.
- **Survives `qd.init()` resets.** Coverage data is accumulated across multiple `qd.init()` calls within the same
  process.

## Running coverage manually

You can enable kernel coverage for any script by setting the environment variable:

```bash
QD_KERNEL_COVERAGE=1 python my_script.py
```

This writes `_qd_kcov.<pid>` files in the working directory. To combine them with Python coverage (if applicable) and
produce a report, run:

```bash
python tests/coverage_report.py
```

## Report formats

The `coverage_report.py` script supports several output formats:

| Format | Flag | Description |
|--------|------|-------------|
| HTML (default) | `--format html` | Interactive file with collapsible per-file sections |
| Terminal | `--format terminal` | Compact summary printed to stdout |
| Annotated | `--format annotated` | Terminal summary + line-by-line hit/miss markers |
| Markdown | `--format markdown` | GitHub-flavored markdown (used in CI PR comments) |

### Examples

```bash
# HTML report (default), saved to a custom path
python tests/coverage_report.py -o my-report.html

# Terminal summary
python tests/coverage_report.py --format terminal

# Compare against a different base branch
python tests/coverage_report.py --compare-branch origin/release

# Report from existing coverage.xml files (skip combining step)
python tests/coverage_report.py --report-only --coverage-xml coverage.xml
```

## CI integration

In CI, kernel coverage is collected automatically during the test phases. The workflow:

1. Tests run with `QD_KERNEL_COVERAGE=1` and `pytest-cov`.
2. `coverage_report.py --collect-only` combines the data and generates `coverage.xml`.
3. `coverage_report.py --report-only --format markdown` produces a diff coverage report that is posted as a PR comment.

The PR comment includes:
- Overall project coverage percentage
- Diff coverage (only changed lines) with a per-file breakdown
- Collapsible annotated code sections showing which lines were hit or missed

## Prerequisites

Kernel coverage requires the `coverage` Python package:

```bash
pip install coverage
```

When using `run_tests.py -C`, `pytest-cov` is also needed:

```bash
pip install pytest-cov
```

## Limitations

- **Autodiff kernels are skipped.** Coverage probes are not inserted into kernels using autodiff (`AutodiffMode`),
  since the extra field stores would interfere with gradient computation.
- **Offline cache tests.** Some offline-cache tests are automatically skipped when `QD_KERNEL_COVERAGE=1` because the
  coverage probes change the compiled kernel, invalidating cache-related assertions.
- **Probe capacity.** There is a fixed limit of 100,000 coverage probes per process. This is sufficient for typical
  test suites but may need increasing for very large codebases.

## See also

- [Debug mode](./debug.md) — runtime bounds checking and assertions
- [Troubleshooting](./troubleshooting.md)
