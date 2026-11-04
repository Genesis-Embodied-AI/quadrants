# Runtime dataclass diagnostic-path experiment

Keep the existing string paths for now. Neither tuples nor lists produced a consistent end-to-end improvement.
The refined tuple implementation reduced guarded schema-inference time by about 0.8–3.2%, but dispatch results were
mixed. In the two later FEM pairs, tuple process time was 0.07% and 0.16% slower. All six FEM runs passed.

Removing ordinary path extensions entirely reduced parent schema-inference time by about 0.9–2.8% across these cases.
That diagnostic-breaking experiment estimates potential savings in these successful workloads. It does not isolate an
exact universal cost. Initial tuple and list implementations were slower on the parent. Entry-only conversion recovered
small gains in some cases. These measurements do not justify merging a path-representation change.

## Scope and implementation

This experiment compares diagnostic-path construction within two pinned Quadrants snapshots. It does not compare the
snapshots against each other as an estimate of path-handling cost. The experiment changes no schema keys, pruning,
cycle-detection rules, launch-buffer caches, or Final ancestor validation. The active PR branches are untouched.

The parent snapshot is `f862281d3300e8ba4bcec6bdd78bfbab3f0ffbe0`. The guarded-schema snapshot is
`fbff48cee6c88608e9addfc5e41272f2028262dc`. Both retain the name `resolve_dataclass`. Genesis is pinned to
`8650cfb96e70604cf2ae261c2f151c321e465467`.

The string control constructs a formatted string on each descent. The tuple candidate converts the root string to one
tuple element, appends names by tuple concatenation, and joins only when constructing an immediate error. It does not
split root argument names containing dots.

The list candidate converts the root string to a traversal-local list. Each descent appends a name and pops it in a
`finally` block. Successful returns, cache hits, and exceptions all unwind the list. Immediate errors format the path
before unwinding. No list is stored globally, on a kernel, in a schema, or in a deferred error. Existing external calls
such as `resolve_dataclass(value, "state")` still work. Both public inference entry points normalize string roots; this
normalization overhead is included in timings.

Both candidates also change path construction in `FrozenSchemaEntry.matches` on the guarded snapshot. The independent
`final_parent` string path remains unchanged. Existing deferred errors still obtain their original argument path during
compiler binding.

An additional attribution-only variant omits ordinary path extensions. It deliberately breaks nested diagnostics. It is
not a valid implementation and is excluded from correctness and FEM comparisons. Its timings estimate the benefit of
removing this work in error-free workloads; they are not a proposed optimization.

| Snapshot | Variant | Commit | Source lines added/deleted |
| --- | --- | --- | ---: |
| Parent | String control + harness | `8102aca75c48bd542a90471fa74ae9a320b729e2` | 0 / 0 |
| Parent | Tuple | `bc5627b1a3975b29550ac7febdb467b74a67eb4c` | 18 / 8 |
| Parent | List | `5de461c6a161c9517294d9b3cc16a42dcf8be2d7` | 24 / 8 |
| Parent | Elided attribution | `f65e25189d0f8aa309e4561c42eb93801c3a8534` | 2 / 4 |
| Guarded | String control + harness | `8d6de625448027c8d009fcf1bc2d353d25a89dda` | 0 / 0 |
| Guarded | Tuple | `d1f210c37b37e3671c65c22e1c2ad31a2c11c13d` | 14 / 8 |
| Guarded | List | `68e98eb29c78c487f0f4d3e8a63b8f085b9d90ee` | 29 / 9 |
| Guarded | Elided attribution | `3719f411499d28b4f827632dab5067e0ba7ec587` | 3 / 3 |

Counts exclude tests and experiment scripts. They include formatting changes to modified signatures. The parent
candidates change one implementation file. The guarded candidates change that file and the guard helper. Control
commits add the common harness and tests without changing implementation sources.

## Environment

Runs use Slurm allocation 780287 on `rtx-210-029`, with one NVIDIA RTX PRO 6000 Blackwell GPU and 16 allocated CPUs.
The node has Intel Xeon Platinum 8562Y+ processors. The container is `/mnt/data/images/genesis-v1_26.sqsh`. NVIDIA driver
version is 595.71.05. Python is 3.12.14.

The microbenchmark and Quadrants tests use the experiment-owned `.venv`: NumPy 2.5.4, SciPy 1.18.1, and pytest 9.2.0.
This environment has no PyTorch. Its native core comes from the original baseline wheel at `b2f658e58`.

FEM uses a separate experiment-owned `fem-venv`. It was freshly created using Genesis's production installation script,
with its `/venv` destination replaced by the owned environment path. It contains PyTorch 2.13.0+cu129, NumPy 2.5.4,
SciPy 1.18.1, and pytest 9.1.1. Its native core comes from the original PR CI wheel at `2da59f323`.

Each comparison keeps its native binaries and dependencies fixed. Every variant installs the complete Python package
sources from its own checkout, including the required helper modules. The driver verifies installed resolver/helper
bytes against the checkout and records their SHA-256 hashes, commit, and installation path. The benchmark independently
records `quadrants.__file__` and resolver/guard source hashes. No native-source differences exist between the pinned
snapshots and the native baseline. This is a Python-only experiment, not a fresh native build.

## Correctness

The runner uses `tests/run_tests.py`, with CPU and CUDA selected. Every valid parent variant passes the same tests:

| Snapshot | Variant | Passed | Failed | Skipped | Existing xfails |
| --- | --- | ---: | ---: | ---: | ---: |
| Parent | Strings | 340 | 0 | 0 | 6 |
| Parent | Tuples | 340 | 0 | 0 | 6 |
| Parent | Lists | 340 | 0 | 0 | 6 |
| Guarded | Strings | 298 | 6 | 2 | 6 |
| Guarded | Tuples | 298 | 6 | 2 | 6 |
| Guarded | Lists | 298 | 6 | 2 | 6 |

The guarded snapshot has the same six pre-existing unused-member failures in every variant: unused string, unused set,
and unused unsupported Final member, each on CPU and CUDA. Those tests remain unchanged. The two skips are the new
parent-only deferred-error test on the older snapshot, where that feature does not exist.

New tests cover nested cycle diagnostics, root names containing dots, shared children that are not cycles, a successful
call after failure, a later failure on the other sibling, original paths through inlined calls, and acceptance of unused
unsupported members. Existing tests cover Final diagnostics, mutable ancestors, guard hits, and invalidation.

The initial tuple and list variants pass the changed-file pre-commit hooks. Pyright reports 14 errors and 10 warnings
for each control and candidate. Normalized diagnostics are identical within each snapshot; there are no new diagnostics.
This is not a clean type-checker result. Several baseline diagnostics concern the missing PyTorch dependency.

## Initial microbenchmarks

Each process warms each case, then takes seven samples of 1,000 calls. Four process repetitions use alternating orders.
The five cases include the original 1x1, 1x32, and 8x32 frozen Tensor containers, plus eight-level frozen chains ending in
a Tensor wrapper or a mutable scalar dataclass. Caches remain enabled. Every case retains one specialization.

The table shows the median of process medians, in microseconds. Parenthesized changes are the median of the four paired
candidate/control ratios. These percentages need not equal the ratio of the two displayed medians.

| Snapshot / case | Metric | Strings | Tuples | Lists | Elided attribution |
| --- | --- | ---: | ---: | ---: | ---: |
| parent / 1x1 | schema | 8.068 | 8.358 (+1.86%) | 8.218 (+1.23%) | 7.763 (-2.84%) |
| parent / 1x1 | dispatch | 14.316 | 14.936 (+2.84%) | 14.847 (+2.26%) | 13.931 (-1.77%) |
| parent / 1x32 | schema | 89.240 | 89.671 (+0.54%) | 92.066 (+2.83%) | 88.032 (-1.35%) |
| parent / 1x32 | dispatch | 125.351 | 126.174 (+0.68%) | 128.800 (+3.05%) | 124.486 (-0.60%) |
| parent / 8x32 | schema | 727.362 | 746.034 (+3.00%) | 755.700 (+2.70%) | 715.685 (-1.45%) |
| parent / 8x32 | dispatch | 975.242 | 984.051 (+1.84%) | 1004.923 (+2.92%) | 959.773 (-1.09%) |
| parent / deep_tensor | schema | 33.469 | 33.895 (+1.57%) | 34.278 (+1.47%) | 32.937 (-1.52%) |
| parent / deep_tensor | dispatch | 51.277 | 51.628 (+0.68%) | 51.767 (+0.89%) | 50.144 (-2.21%) |
| parent / deep_mutable | schema | 33.130 | 33.347 (+1.20%) | 33.805 (+1.77%) | 32.827 (-0.86%) |
| parent / deep_mutable | dispatch | 48.301 | 48.585 (+1.03%) | 50.014 (+2.54%) | 47.935 (-1.03%) |
| guard / 1x1 | schema | 1.327 | 1.342 (+0.77%) | 1.372 (+2.44%) | 1.260 (-5.78%) |
| guard / 1x1 | dispatch | 7.971 | 7.571 (-4.68%) | 7.645 (-3.80%) | 7.473 (-8.72%) |
| guard / 1x32 | schema | 3.093 | 2.808 (-8.30%) | 2.897 (-8.69%) | 2.736 (-11.16%) |
| guard / 1x32 | dispatch | 38.614 | 36.533 (-5.76%) | 36.286 (-5.66%) | 36.377 (-5.60%) |
| guard / 8x32 | schema | 18.885 | 18.366 (-1.35%) | 18.616 (-0.49%) | 17.583 (-5.27%) |
| guard / 8x32 | dispatch | 260.751 | 256.787 (-0.81%) | 255.818 (-1.19%) | 257.488 (-1.25%) |
| guard / deep_tensor | schema | 6.251 | 6.539 (+3.82%) | 6.566 (+5.97%) | 6.014 (-5.51%) |
| guard / deep_tensor | dispatch | 22.360 | 22.176 (-0.82%) | 21.950 (+0.05%) | 21.771 (-2.63%) |
| guard / deep_mutable | schema | 7.676 | 7.913 (+3.18%) | 7.938 (+3.42%) | 7.351 (-4.24%) |
| guard / deep_mutable | dispatch | 22.825 | 23.239 (+1.84%) | 23.003 (+1.54%) | 22.683 (+0.08%) |

Neither initial candidate improves the parent snapshot's median in any case. Omitting path extensions reduces its
8x32 schema time by 1.45% and dispatch time by 1.09% in the paired measurements. This is a small portion of the measured
work, even for an invalid implementation that omits the work entirely.

The guarded results are mixed and show process variation. Some apparent improvements occur even though the affected
path is mostly bypassed by guards. The experiment therefore adds a second round with six repetitions and normalizes
paths only at external entry points. This also removes an avoidable per-recursive-call type check from the candidates.

## Entry-point-only refinement

The refined candidates retain both external functions and their root-string arguments. Small wrappers create a tuple or
list and call internal inference helpers. Recursive calls and guarded descendants use those helpers directly. The
benchmark includes the wrapper call cost. No inference or caching rules change. These candidates are measured against
fresh string-control runs, not against the earlier control timings.

| Snapshot | Entry-only variant | Commit | Source lines added/deleted |
| --- | --- | --- | ---: |
| parent | tuple | `96c6416ac9c2d08c062460118fb82930905452d2` | 30 / 8 |
| parent | list | `4212ada716a873039727576babf8f0ad8f098073` | 35 / 9 |
| guard | tuple | `b734597279cf3ebee02bd6ce4e3d2552033b46a5` | 21 / 9 |
| guard | list | `8d6959860a03c1ffd8a8304435da29cfce6396e9` | 36 / 10 |

The refined parent candidates each pass 340 tests, with six existing xfails. Each refined guarded candidate has the same
298 passes, six baseline failures, two skips, and six xfails as its control. Their pre-commit checks pass. Pyright has
exactly the same normalized diagnostics as the corresponding control.

Six process repetitions cover all six orders of strings, tuples, and lists. Each process still takes seven samples of
1,000 calls per case. The refined tuple version reduces schema-inference medians in all five guarded cases, but does not
consistently improve dispatch. The refined list version is less consistent. Table conventions match the initial round.

| Snapshot / case | Metric | Strings | Entry-only tuples | Entry-only lists |
| --- | --- | ---: | ---: | ---: |
| parent / 1x1 | schema | 7.894 | 7.895 (-0.40%) | 8.051 (+2.07%) |
| parent / 1x1 | dispatch | 14.112 | 14.204 (+0.43%) | 14.617 (+2.27%) |
| parent / 1x32 | schema | 89.693 | 89.572 (-0.39%) | 88.267 (-0.95%) |
| parent / 1x32 | dispatch | 125.838 | 125.303 (-0.29%) | 124.599 (-1.04%) |
| parent / 8x32 | schema | 728.406 | 720.630 (-1.83%) | 707.829 (-3.71%) |
| parent / 8x32 | dispatch | 973.326 | 972.602 (-0.87%) | 960.290 (-1.69%) |
| parent / deep_tensor | schema | 34.006 | 33.223 (-1.94%) | 33.395 (-1.85%) |
| parent / deep_tensor | dispatch | 51.721 | 51.132 (-1.26%) | 51.959 (+1.32%) |
| parent / deep_mutable | schema | 33.568 | 33.203 (-1.22%) | 33.425 (-0.55%) |
| parent / deep_mutable | dispatch | 49.530 | 48.007 (-1.42%) | 48.744 (+0.26%) |
| guard / 1x1 | schema | 1.291 | 1.284 (-0.98%) | 1.331 (+2.96%) |
| guard / 1x1 | dispatch | 7.415 | 7.632 (+1.65%) | 7.662 (+3.74%) |
| guard / 1x32 | schema | 2.768 | 2.717 (-0.83%) | 2.779 (+0.39%) |
| guard / 1x32 | dispatch | 36.282 | 36.897 (+1.65%) | 36.899 (+1.76%) |
| guard / 8x32 | schema | 18.206 | 17.830 (-1.96%) | 18.105 (-0.38%) |
| guard / 8x32 | dispatch | 256.043 | 257.783 (+1.09%) | 257.264 (+0.56%) |
| guard / deep_tensor | schema | 6.244 | 6.072 (-3.01%) | 6.178 (-0.69%) |
| guard / deep_tensor | dispatch | 21.712 | 21.448 (-0.99%) | 21.884 (+0.63%) |
| guard / deep_mutable | schema | 7.638 | 7.419 (-3.20%) | 7.529 (-1.98%) |
| guard / deep_mutable | dispatch | 22.756 | 22.550 (-0.97%) | 22.690 (-0.40%) |

For FEM, select the guarded entry-only tuple implementation, `b734597279cf3ebee02bd6ce4e3d2552033b46a5`.
It has the most consistent guarded inference improvement in the refined comparison. This is not a claim that it wins
all dispatch cases: the shallow dispatch medians are 1.1–1.7% slower, while the deep cases are about 1% faster.

## FEM protocol

Run the explicit float64 hard-constraint test with `GS_ENABLE_NDARRAY=1`, `MADRONA_DISABLE_CUDA_HEAP_SIZE=1`, and
`QD_OFFLINE_CACHE_CLEANING_POLICY=never`. The control uses the guarded snapshot without path changes. The candidate uses
that identical snapshot plus entry-only tuple paths. Keep the native wheel, dependencies, node, and allocation fixed.

Run control/candidate, then candidate/control, then a confirmation control/candidate pair, sequentially.
Give every run a distinct, initially absent
`QD_OFFLINE_CACHE_FILE_PATH`. Do not profile or run other tests or benchmarks alongside these measurements. The process
wall time starts immediately before launching pytest; it excludes source installation. The test's original 1,200-second
timeout remains enabled, including setup and teardown. An outer 1,300-second timeout prevents a stalled process from
outliving the experiment. OS and CUDA-driver caches are not explicitly cleared. Completed runs populate their separate
Quadrants cache directories. The additional pair was chosen after observing much greater startup and setup time in the
first control; all runs remain reported.

## FEM results

All six runs passed. Times below are seconds, in execution order.

| Pair | Variant | Process wall | Pytest total | Test call | Setup | Teardown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | guard-control | 390.29 | 376.74 | 321.62 | 12.38 | 1.90 |
| 1 | guard-tuple-entry | 314.77 | 311.07 | 296.79 | 2.53 | 1.64 |
| 2 | guard-tuple-entry | 314.88 | 311.35 | 297.49 | 2.72 | 1.57 |
| 2 | guard-control | 314.67 | 311.66 | 298.28 | 2.24 | 1.92 |
| 3 | guard-control | 314.01 | 310.17 | 295.09 | 2.66 | 1.68 |
| 3 | guard-tuple-entry | 314.51 | 311.29 | 297.52 | 2.66 | 1.63 |

| Pair | Order | Wall change | Pytest change | Test-call change |
| --- | --- | ---: | ---: | ---: |
| 1 | Strings, tuples | -19.35% | -17.43% | -7.72% |
| 2 | Tuples, strings | +0.07% | -0.10% | -0.26% |
| 3 | Strings, tuples | +0.16% | +0.36% | +0.82% |

Changes compare tuples with strings within each pair. Negative means less time.

The first control took substantially longer, including startup and setup. Its apparent 19% process-time benefit did
not repeat in either later pair. Those pairs show essentially equal process times, with small differences in opposite
directions for the test body. The experiment does not identify the cause of the first run’s extra time. All runs are
retained; the report does not discard the first pair or treat its large change as a repeatable optimization benefit.

There is no demonstrated FEM speedup from tuple paths. The small guarded inference improvements do not establish a
useful end-to-end improvement. Lists were not selected for FEM because their refined guarded microbenchmarks were less
consistent. This experiment therefore makes no direct full-FEM claim about lists.

This reproduction runs one test in isolation. Historical CI used 64 workers. Passing here does not establish that
the concurrent CI job will meet its 1,200-second timeout. Historical timings on other nodes are not comparison controls.


## Reproduction and artifacts

The experiment branch is `hp/path-experiment`. Implementation variants are on the `hp/path-parent-*` and
`hp/path-guard-*` branches. The driver pins exact implementation commits; branch tips do not choose code during replay.
The benchmark used for measurement is from harness commit `96c4997e2`. Its source hash is recorded in `results.json`.

Use a new workspace with a Quadrants checkout on the experiment branch and a Genesis checkout at the pinned commit.
Use the cluster image above, one GPU, and 16 CPUs. Create an owned Python 3.12 `.venv` with the baseline native wheel,
Quadrants test dependencies, pre-commit, and Pyright. Fetch the experiment branches so all pinned commits are available.
Obtain the original PR wheel from artifact `13286774303`, GitHub Actions run `41752788489`. The baseline native wheel
comes from artifact `13279233831`, run `41733303874`. The historical CI investigation in perso_hugh documents these
artifacts and their original environment.

From the experiment checkout, with `ROOT` pointing to the owned workspace:

```bash
python misc/experiments/dataclass_paths/setup_fem.py "$ROOT" "$PR_WHEEL"
python misc/experiments/dataclass_paths/run.py "$ROOT" tests
python misc/experiments/dataclass_paths/run.py "$ROOT" micro
python misc/experiments/dataclass_paths/run.py "$ROOT" types
python misc/experiments/dataclass_paths/check_variants.py "$ROOT"
python misc/experiments/dataclass_paths/run.py "$ROOT" tests --entry-only
python misc/experiments/dataclass_paths/run.py "$ROOT" micro --entry-only
python misc/experiments/dataclass_paths/run.py "$ROOT" types --entry-only
python misc/experiments/dataclass_paths/check_variants.py "$ROOT" --entry-only
python misc/experiments/dataclass_paths/run.py "$ROOT" fem --candidate tuple-entry --fem-pairs 3
python misc/experiments/dataclass_paths/summarize.py "$ROOT/tmp/path-results"
python misc/experiments/dataclass_paths/summarize.py "$ROOT/tmp/path-results" --entry-only
python misc/experiments/dataclass_paths/summarize_fem.py "$ROOT/tmp/path-results"
```

Run these phases sequentially. The runner installs complete Python snapshots into the selected environment and restores
the parent control in `finally`. It rejects reuse of a FEM cache directory. Use a fresh result directory for a complete
replay. To resume after two completed FEM pairs, use `--start-pair 2 --fem-pairs 1`; this appends results and uses new cache
paths. It does not overwrite earlier runs.

`make_variant.py` generates the tuple, list, or diagnostic-only elision changes from either baseline.
`entry_only.py` refines a tuple/list checkout to use entry wrappers. Format generated files with Black 25.1.0 at 120
columns. The candidate commits contain the exact measured implementations; generation scripts do not select code at
benchmark runtime.

`results.json` contains all microbenchmark samples, process medians, paired changes, source identities, driver run
records, and all six FEM timings. `packages-micro.txt` and `packages-fem.txt` record the complete installed package sets. Their wheel and Genesis
file URLs identify the original local inputs; replace those paths when replaying elsewhere. Full logs are retained under:

```text
/home/hugh/ais/misc_dataclass_runtime_types_walkthrough/tmp/path-results/
```

Subdirectories are `tests`, `tests-entry`, `micro`, `micro-entry`, `types`, `types-entry`, `lint`, `lint-entry`, and `fem`.
Filenames contain the snapshot, variant, and repetition. For example, `tests/parent-list-0.log` is the full initial list
test run. `fem/guard-control-0.log` is the first full FEM control log. Cluster files use the same suffix beneath
`/mnt/data/hugh/ais/misc_dataclass_runtime_types_walkthrough/`.

Environment and installation logs are `tmp/path-environment.log`, `tmp/path-fem-packages.txt`, and
`tmp/path-fem-setup.log`. Slurm execution is captured in `tmp/output.log` on the cluster. The allocation is released after
the experiment. No experiment changes are merged into #1059 or #1067.
