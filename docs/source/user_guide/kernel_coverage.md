# Kernel code coverage

Standard Python coverage tools only measure host-side code. Quadrants kernel coverage goes further — it tracks which
lines actually execute *inside* compiled kernels on the device (CPU or GPU), including which branches of `if`/`else`
blocks are taken at runtime.

The coverage data is written in the standard `coverage.py` format, so it works with `coverage report`, `diff-cover`,
and IDE coverage viewers out of the box.

## Prerequisites

Kernel coverage requires the `coverage` Python package:

```bash
pip install coverage
```

## Enabling kernel coverage

Set the `QD_KERNEL_COVERAGE` environment variable before running your program:

```bash
QD_KERNEL_COVERAGE=1 python my_simulation.py
```

This works with any script that uses quadrants kernels — no changes to your code are needed.

When the process exits, quadrants writes one or more `_qd_kcov.<pid>` files in the working directory containing the
collected coverage data.

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

If you run your tests with `pytest-cov`, kernel coverage data is automatically merged with Python coverage. Enable
both at once:

```bash
QD_KERNEL_COVERAGE=1 pytest --cov=my_package --cov-branch tests/
```

After the run, `coverage combine _qd_kcov.* .coverage` merges the kernel and Python data into a single report.

## Example

```python
import quadrants as qd

qd.init(arch=qd.gpu)

result = qd.field(dtype=qd.i32, shape=(1,))

@qd.kernel
def my_kernel():
    x = 10
    if x > 5:
        result[0] = 1    # this line will show as covered
    else:
        result[0] = 2    # this line will show as NOT covered

my_kernel()
```

Running with `QD_KERNEL_COVERAGE=1` and then inspecting the report will show that only the `if` branch was executed,
and the `else` branch was missed.

## Key properties

- **Zero overhead when disabled.** The coverage module is never imported unless `QD_KERNEL_COVERAGE=1` is set. There
  is no cost in normal operation.
- **Branch coverage.** Probes inside `if`/`else` bodies only fire when that branch is taken, giving true runtime
  branch coverage — not just line coverage.
- **Works with pytest-xdist.** Each worker writes to a separate file; combine them afterward.
- **Survives `qd.init()` resets.** Coverage data is accumulated across multiple `qd.init()` calls within the same
  process.

## Advanced usage

### Probe capacity

There is a limit of 100,000 coverage probes per process (one probe per unique source line per kernel/func). If you hit
the limit — for example in a very large codebase with many kernels — increase it via the environment variable:

```bash
QD_COVERAGE_MAX_PROBES=500000 QD_KERNEL_COVERAGE=1 python my_simulation.py
```

## Coverage and autodiff

Quadrants compiles each kernel multiple times when autodiff is used: once for the normal forward execution, and
again for the backward (or forward-mode AD) replay pass. Coverage probes are only inserted into the normal forward
compilation — they are excluded from the AD replay compilations because the extra field stores would interfere with
gradient computation.

### What is covered

When you call a kernel inside a `qd.ad.Tape` context, the forward pass runs first with coverage probes active. This
means every line of your kernel source code that executes during the forward pass is tracked normally, including
branch coverage.

```python
@qd.kernel
def compute_loss():
    for i in range(n):
        if x[i] > 0:          # covered: probe fires during forward pass
            loss[None] += x[i] # covered
        else:
            loss[None] += 0.0  # covered only if this branch is taken during forward

with qd.ad.Tape(loss):
    compute_loss()             # forward pass: probes active
                               # backward pass: runs automatically, no probes
```

### What is not covered

The backward pass is an automatically generated transformation of the same kernel — it is not separate source code
you wrote. Since it replays the same control flow as the forward pass, there are no user-written lines that would
only appear in the backward pass.

In short: as long as your test exercises the forward pass (which is always required before a backward pass), coverage
of your kernel source lines is accurate and complete.

### Edge case

If you have a kernel that is *only* ever called inside a `qd.ad.Tape` with `validation=True` and never called
outside one, it will be compiled exclusively in validation mode and will not receive coverage probes.

## Limitations

- **Offline cache interaction.** Coverage probes change the compiled kernel, so the offline cache will see them as
  new kernels and recompile. This is expected and does not affect correctness, but the first run with coverage enabled
  will be slower if you normally rely on cached kernels.

## Under the hood

When `QD_KERNEL_COVERAGE=1` is set, quadrants rewrites the Python AST of each `@qd.kernel` and `@qd.func` before
compilation. It inserts lightweight probe statements (`field[probe_id] = 1`) at each source line. These probes compile
as ordinary field stores and execute on the device alongside your kernel code.

At process exit, the probe data is read back from the device and written to a `.coverage`-compatible file.
