# Kernel code coverage

Standard Python coverage tools only measure host-side code. Quadrants kernel coverage goes further — it tracks which lines actually execute *inside* compiled kernels on the device (CPU or GPU), including which branches of `if`/`else` blocks are taken at runtime.

The coverage data is written in the standard `coverage.py` format, so it works with `coverage report`, `pytest-cov`, `diff-cover`, and IDE coverage viewers out of the box.

## Prerequisites

Kernel coverage requires the `coverage` Python package:

```bash
pip install coverage
```

## Enabling kernel coverage

### Automatic with pytest-cov

If you use `pytest-cov`, kernel coverage is enabled automatically — no configuration needed. Quadrants ships a pytest plugin that detects `--cov` and sets `QD_KERNEL_COVERAGE=1` for you. Just run:

```bash
pytest --cov=my_package --cov-branch tests/
```

To disable kernel coverage while still collecting Python coverage, opt out explicitly:

```bash
QD_KERNEL_COVERAGE=0 pytest --cov=my_package --cov-branch tests/
```

### Manual with any script

For scripts outside pytest, set the `QD_KERNEL_COVERAGE` environment variable:

```bash
QD_KERNEL_COVERAGE=1 python my_simulation.py
```

This works with any script that uses quadrants kernels — no changes to your code are needed.

When the process exits, quadrants writes one or more `_qd_kcov.<pid>` files in the working directory containing the collected coverage data.

## Viewing results

### With coverage.py

Combine the kernel coverage files and produce a report using the standard `coverage` tool:

```bash
# Combine all kernel coverage files into .coverage
coverage combine _qd_kcov.*

# Terminal summary
coverage report --show-missing

# HTML report
coverage html
```

### With pytest-cov

When using `pytest-cov`, kernel coverage is enabled automatically (see above). The kernel coverage data is merged with Python coverage after the run:

```bash
coverage combine _qd_kcov.* .coverage
```

## Key properties

- **Zero overhead when disabled.** The coverage module is never imported unless `QD_KERNEL_COVERAGE=1` is set. There is no cost in normal operation.
- **Branch coverage.** Probes inside `if`/`else` bodies only fire when that branch is taken, giving true runtime branch coverage — not just kernel-level coverage, or static conditional coverage.
- **Works with pytest-xdist.** Each worker writes to a separate file; combine them afterward.
- **Survives `qd.init()` resets.** Coverage data is accumulated across multiple `qd.init()` calls within the same process.

## Advanced usage

### Probe capacity

There is a limit of 100,000 coverage probes per process (one probe per unique source line per kernel/func). If you hit the limit — for example in a very large codebase with many kernels — increase it via the environment variable:

```bash
QD_COVERAGE_MAX_PROBES=500000 QD_KERNEL_COVERAGE=1 python my_simulation.py
```

## Coverage and autodiff

The forward pass is covered. The backward pass is not, because instrumenting it would interfere with gradient computation. This is normally fine — the backward pass is auto-generated and replays the same control flow, so forward coverage is sufficient.

One edge case: kernel calls inside a `qd.ad.Tape` with `validation=True` will not be covered.

## Offline cache interaction

Coverage probes change the compiled kernel, so the offline cache will see them as new kernels and recompile. This is expected and does not affect correctness, but the first run with coverage enabled will be slower if you normally rely on cached kernels.

## Under the hood

When `QD_KERNEL_COVERAGE=1` is set, quadrants rewrites the Python AST of each `@qd.kernel` and `@qd.func` before compilation. It inserts lightweight probe statements (`field[probe_id] = 1`) at each source line. These probes compile as ordinary field stores and execute on the device alongside your kernel code.

At process exit, the probe data is read back from the device and written to a `.coverage`-compatible file.
