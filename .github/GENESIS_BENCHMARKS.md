# Genesis performance baselines

Quadrants performance checks use a manually activated Genesis commit. Compatibility unit tests continue using Genesis
main. The active selection is stored in `active.json` on the `genesis-benchmark-state` branch. Complete measurements
and their provenance are stored as artifacts in W&B project `genesis-ai-company/quadrants-genesis-benchmarks`.

## Bootstrap or switch Genesis

Run **Generate Genesis benchmark baseline** from the Actions page on main, or:

```sh
gh workflow run genesis_bootstrap.yml --repo Genesis-Embodied-AI/quadrants --ref main -f genesis_ref=v1.4.7
```

`genesis_ref` accepts any tag, branch, or full commit SHA available in the official Genesis repository. The workflow
resolves it once and benchmarks current Quadrants main against that exact Genesis commit. Both benchmark modes must
complete and be published before the active selection changes. Failed bootstrap leaves the previous selection active.
New releases and branch movement never change it automatically. PR runs in progress keep their recorded selection.

Bootstrap requires a successful main Manylinux wheel build with provenance metadata and an unexpired CPython 3.12
x86-64 artifact. If it is unavailable, run **Manylinux wheel build** on main first, then retry bootstrap. Bootstrap runs
benchmarks only. It does not repeat compatibility unit tests or rebuild wheels automatically.

Manual bootstrap runs share a concurrency group. GitHub may replace a queued request, and does not guarantee FIFO
ordering. Check the Actions run status and activation summary. An older request cannot replace a newer activation.

## Baseline promotion and comparisons

Complete PR measurements are published before comparison. If the PR merges, they become eligible as a main baseline
only when the wheel's source tree matches the landed tree. The merge handler and publication handler both reconcile
this condition, so either event order works. A reported performance regression does not invalidate measurements of
code that was subsequently merged.

Selection uses main history, with the PR's recorded cutoff, rather than completion time. Older matching baselines can
serve comparisons while newer results are pending. Reports identify the actual baseline and cutoff. Candidate results
are excluded from their own comparison. A missing baseline or incompatible environment is an explicit failure.

The environment manifest records resolved dependencies, Python, container version, GPU model, driver, and the build
script digest. Candidates constrain dependencies to the activated versions. An environment mismatch requires another
manual bootstrap. Bootstrap can also refresh a baseline when merged code differs from a PR's tested tree.

To reconcile a merge manually:

```sh
gh workflow run genesis_promote.yml --repo Genesis-Embodied-AI/quadrants --ref main -f pr=1234
```

Results from older configurations remain stored and cannot reactivate those configurations. There is no automatic
bootstrap dispatch or retry of comparisons waiting for a baseline. To retry a failed comparison without rerunning
benchmarks, rerun its comparison job. A benchmark retry must rerun both modes in one attempt. Earlier measurements
cannot be combined with new ones. The input snapshot retains the original Genesis selection.

## Deployment and validation

Merge the corresponding Genesis reusable workflow changes before the Quadrants caller. The helper needs the existing
`WANDB_API_KEY`; bootstrap activation additionally requests `contents: write` for its state branch. Promotion uses
trusted main code and only read access to GitHub. The benchmark runner retains the existing Genesis CI App credentials.
The repository's branch rules must permit the bootstrap token to update `genesis-benchmark-state`.

Run initial bootstrap before requiring the new performance comparison. Verify one PR run, both merge/completion orders,
and a failed bootstrap before enabling it as a merge gate. An arbitrary older Genesis version may be incompatible with
the maintained build tooling or current Quadrants. Such a bootstrap fails explicitly and leaves the selection intact.

The infrastructure tests require Python 3.12, `wandb<0.25`, and `frozendict`, without a Quadrants build:

```sh
python -m unittest discover -s .github/scripts/tests -v
```

Run the Genesis explicit-baseline tests from that repository:

```sh
python -m unittest discover -s .github/workflows/scripts/tests -v
```
