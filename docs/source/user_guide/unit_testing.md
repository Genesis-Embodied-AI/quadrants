# Unit testing

This page documents how to run, write, and tune the Quadrants Python unit test suite. For setup of the build / dev environment, see [contributing.md](contributing.md).

## Running the tests

The test suite is run via the project's launcher, **not** by invoking `pytest` directly:

```
python tests/run_tests.py
```

The launcher sets up the test-only env vars (kernel offline cache, watchdog, xdist worker count, etc.) and forwards any unrecognised flags to pytest. Calling `pytest` directly skips that setup and behaves differently.

Common one-liners:

```
# run one file
python tests/run_tests.py test_tile16

# run one test (any pytest -k expression)
python tests/run_tests.py -k test_tile16_cholesky

# run on a specific backend (or comma-separated list)
python tests/run_tests.py --arch cuda
python tests/run_tests.py --arch metal -k tile16

# same, via env var (handy for CI)
QD_WANTED_ARCHS=metal,vulkan python tests/run_tests.py

# rerun the last failing tests first
python tests/run_tests.py -f

# stop at the first failure
python tests/run_tests.py -x
```

The target architecture can also be set via `QD_WANTED_ARCHS` (comma-separated; supports `^arch` to exclude rather than include).

## Markers

Tests can opt into two project-specific markers, in addition to pytest's built-in ones (`skip`, `xfail`, etc.).

### `@pytest.mark.slow`

Marks a test as **slow**. `tests/run_tests.py` adds `-m "not slow"` to the pytest invocation by default; pass `--run-slow` to opt back in:

```
# default: skip slow
python tests/run_tests.py

# include slow
python tests/run_tests.py --run-slow

# slow ONLY (e.g. nightly job)
python tests/run_tests.py -m slow --run-slow
```

The marker is used in two patterns:

1. **Whole-test slow**: the whole test takes a long time.

   ```python
   @pytest.mark.slow
   def test_thing_that_is_always_slow():
       ...
   ```

2. **Slow-marked parametrize case**:

   ```python
   @pytest.mark.parametrize("n", [4, pytest.param(12, marks=pytest.mark.slow)])
   def test_sym_eig_general(n):
       ...
   ```

   In this specific example the default suite still exercises the code path; the slow lane just adds the larger-size variant for full coverage.

### `@pytest.mark.sample(...)`

Marks a single heavily-parametrized test as opting in to **per-run stochastic sub-selection** of its parametrize cases. Use when:

- the test's parametrize space is large (≥ ~16 cases),
- each parametrize case is roughly independent (covering an independent corner case rather than a single bug class),
- running every case every CI run is overkill, and
- asymptotic coverage over many runs is acceptable.

Apply it like any other marker:

```python
@pytest.mark.sample(n=6)                     # keep 6 of N cases per run
# OR
@pytest.mark.sample(fraction=0.25)           # keep 25% of cases per run, min 1
@pytest.mark.parametrize("size", [...])
@pytest.mark.parametrize("dtype", [...])
@pytest.mark.parametrize("layout", [...])
@test_utils.test(arch=qd.gpu)
def test_thing(size, dtype, layout):
    ...
```

**How to reproduce failing tests.** Three levels of reproducibility:

1. **One failing case** — paste the failing nodeid from the CI log. Pytest already prints the full nodeid on failure:

   ```
   FAILED tests/python/test_tile16.py::test_tile16_load_store[arch=cuda-qd_dtype0-ndarray-16-32-4-8-7-11]
   ```

   Just rerun it directly:

   ```
   python tests/run_tests.py -k "test_tile16_load_store and ndarray-16-32-4-8-7-11"
   # or, if you want the exact nodeid (bypasses -k matching):
   pytest "tests/python/test_tile16.py::test_tile16_load_store[arch=cuda-qd_dtype0-ndarray-16-32-4-8-7-11]"
   ```

   When pytest narrows collection to a single nodeid, the sampler's `len(group) <= 1` short-circuit keeps it. **No `--sample-seed` flag needed.**

2. **The exact subset of a failing run** — useful when several cases failed and you want to bisect or reproduce the whole sample locally. The report header of every run prints the seed used:

   ```
   sample-seed=1834729104  (reproduce the same sample: --sample-seed=1834729104; ...)
   ```

   Then locally:

   ```
   python tests/run_tests.py --sample-seed=1834729104
   ```

3. **Exhaustive run** — for release gates, coverage-debt audits, or a periodic "did anything regress in any branch of the parametrize space" sweep. Disables the sampler entirely; every `@sample`-marked test runs every parametrize case:

   ```
   python tests/run_tests.py --no-sample
   ```

**Per-test RNG independence.** Each `@sample`-marked test's subsample is seeded from `(global_seed, test_nodeid_prefix)`, so adding / renaming / tweaking the mark on `test_A` does NOT shift the sample of `test_B`. Routine refactors don't cause samples to migrate file-wide.

**Composition with `slow`.** Sampling runs **after** marker-based filtering. With `--run-slow` not passed (the default), slow-marked parametrize cases drop out first, then the sampler sub-selects from the remaining (fast) cases. The intersection is the right composition: `--no-sample --run-slow` is the truly-exhaustive combo.

## Writing new tests

The standard recipe combines `@test_utils.test(...)` (arch / option matrix) with `@pytest.mark.parametrize`:

```python
import pytest
import quadrants as qd
from tests import test_utils


@pytest.mark.parametrize("n", [4, pytest.param(12, marks=pytest.mark.slow)])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32)
def test_my_thing(n):
    ...
```

`@test_utils.test` is what wires the test into the per-backend matrix and applies platform exclusions (`exclude=`), extension requirements (`require=`, e.g. `qd.extension.data64` for f64 tests), and per-test options (`default_fp`, `fast_math`, etc.). See `tests/test_utils.py` for the full surface.

Common helpers in `tests/test_utils.py`:

- `test_utils.skip_if_f64_unsupported(dtype)` — skip the current test at runtime if `dtype == qd.f64` and the active backend can't carry f64 through buffer I/O (Metal, MoltenVK on Darwin). Use inside a parametrized test that sweeps both f32 and f64.
- `test_utils.expected_archs()` — list of archs that the current `QD_WANTED_ARCHS` allows. Used to skip tests with no satisfiable arch.

## Advanced

Optional knobs and runtime details. The defaults work for most contributors.

### Per-test timeout

Per-test timeouts default to 600 s and are enforced by `pytest_hardtle`, a CFFI-compiled C watchdog that can kill tests hung in native GPU calls even when the GIL is held.

### Kernel compilation cache

During each test session the kernel compilation cache lives in a fresh, empty temp directory created by pytest's [`tmp_path_factory`](https://docs.pytest.org/en/stable/how-to/tmp_path.html) — typically `/tmp/pytest-of-<user>/pytest-<N>/qdcache0/`. Old session directories are cleaned up automatically by pytest's retention policy. This cache is separate from the user-facing `~/.cache/quadrants/` cache, and avoids recompiling identical kernels after each `qd.reset()` / `qd.init()` cycle within a session.

### Per-file timing breakdown

Set `QD_FILE_TIMING=1` to print a per-file duration summary at the end of the session:

```
QD_FILE_TIMING=1 python tests/run_tests.py
```

This is enabled by default in the Mac CI job; the results appear in the GitHub Actions job summary and are the primary tool for identifying slow test files.

### `@sample` + xdist seed propagation

`tests/run_tests.py` picks the per-run sample seed before pytest is launched and passes it via `--sample-seed=<S>` on argv. xdist forwards argv to every worker, so all workers see the same seed and produce identical samples; without this, each worker would subsample independently and `--sample-seed=<S>` wouldn't reproduce. The per-test RNG inside `pytest_collection_modifyitems` is then derived deterministically via `sha256(f"{seed}|{nodeid_prefix}")`, which is what makes the **Per-test RNG independence** property above hold.
