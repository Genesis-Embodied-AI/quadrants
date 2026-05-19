# Unit testing

This page documents how to run, write, and tune the Quadrants Python unit test suite. For one-shot setup of the build / dev environment, see [contributing.md](contributing.md).

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

The target architecture can also be set via `QD_WANTED_ARCHS` (comma-separated; supports `^arch` to exclude rather than include). Per-test timeouts default to 600 s and are enforced by `pytest_hardtle`, a CFFI-compiled C watchdog that can kill tests hung in native GPU calls even when the GIL is held.

## Kernel compilation cache

During each test session the kernel compilation cache lives in a fresh, empty temp directory created by pytest's [`tmp_path_factory`](https://docs.pytest.org/en/stable/how-to/tmp_path.html) — typically `/tmp/pytest-of-<user>/pytest-<N>/qdcache0/`. Old session directories are cleaned up automatically by pytest's retention policy. This cache is separate from the user-facing `~/.cache/quadrants/` cache, and avoids recompiling identical kernels after each `qd.reset()` / `qd.init()` cycle within a session.

## Per-file timing breakdown

Set `QD_FILE_TIMING=1` to print a per-file duration summary at the end of the session:

```
QD_FILE_TIMING=1 python tests/run_tests.py
```

This is enabled by default in the Mac CI job; the results appear in the GitHub Actions job summary and are the primary tool for identifying slow test files.

## Markers

Tests can opt into two project-specific markers, in addition to pytest's built-in ones (`skip`, `xfail`, etc.).

### `@pytest.mark.slow`

Marks a test (or, more commonly, a specific `pytest.param(...)` case inside a parametrize list) as **slow** — long enough that the default test suite skips it. `tests/run_tests.py` adds `-m "not slow"` to the pytest invocation by default; pass `--run-slow` to opt back in:

```
# default: skip slow
python tests/run_tests.py

# include slow
python tests/run_tests.py --run-slow

# slow ONLY (e.g. nightly job)
python tests/run_tests.py -m slow --run-slow
```

The marker is used in two patterns:

1. **Whole-test slow**: rare. The whole test always takes a long time and there's no smaller variant.

   ```python
   @pytest.mark.slow
   def test_thing_that_is_always_slow():
       ...
   ```

2. **Slow-marked parametrize case** (preferred when applicable): a test parametrizes over a size axis and the large value is slow but the small value is cheap. The small value stays in the default suite as a smoke test; the large value moves to the slow lane. This is the dominant pattern in `tests/python/test_eig.py`, `test_linalg.py`, `test_ad_gdar_diffmpm.py`, etc.

   ```python
   @pytest.mark.parametrize("n", [4, pytest.param(12, marks=pytest.mark.slow)])
   def test_sym_eig_general(n):
       ...
   ```

   With this pattern the default suite still exercises the code path; the slow lane just adds the larger-size variant for full coverage.

### `@pytest.mark.sample(...)`

Marks a single heavily-parametrized test as opting in to **per-run stochastic sub-selection** of its parametrize cases. Use when:

- the test's parametrize space is large (≥ ~16 cases),
- each parametrize case is roughly independent (covering an independent corner case rather than a single bug class),
- running every case every CI run is overkill, and
- coverage convergence over many runs is acceptable for that test.

Apply it like any other marker, above the existing parametrize stack:

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

**Convergence math.** With `keep_n / total = k / N`, the probability that a *specific* parametrize case has been hit after `r` runs is `1 - (1 - k/N)^r`. For `n=6` out of 32 (`test_tile16_load_store`): ~65% after 5 runs, ~88% after 10, ~98% after 20. Combined with our CI cadence and the fact that any persistent bug surface lights up across multiple PRs, this gives effectively full coverage on a many-PR horizon at a fraction of the per-PR cost.

**How to reproduce.** Three levels of reproducibility:

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

**xdist note.** The seed is picked on the controller in `pytest_configure` (not in the per-worker `pytest_collection_modifyitems`), so all xdist workers see the same seed and produce the same sample. This is intentional — without this, each worker would subsample independently and `--sample-seed=<S>` wouldn't reproduce.

**When *not* to use `@sample`.** If the test's parametrize axes are not roughly independent — e.g. axis A's bug surface only lights up when axis B is at a specific value — sampling can miss the interaction. Use `@slow` on the expensive subset instead, and keep the full Cartesian product for the cheap subset.

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

## CI checks

A subset of CI jobs care about the test suite specifically:

- **linux / macosx / win** — build and run the full python suite on each platform.
- **test-gpu** — GPU-specific tests on the cluster.
- **coverage report** — a one-line diff coverage summary is posted as a PR comment, with kernel-level branch coverage. See [Kernel code coverage](kernel_coverage.md).
- **Test coverage check (`check_test_coverage.yml`)** — an AI agent that flags new or modified non-test source code that doesn't have corresponding test coverage in the PR.

See [contributing.md](contributing.md) for the full list of CI checks (linters, pyright, link checking, PR change report, etc.).
