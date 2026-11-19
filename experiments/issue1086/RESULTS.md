# Controlled fast-math experiment for issue 1086

## Findings

Disabling reassociation alone removed the output differences in the issue's reproduction on CPU and CUDA, in both fp32 and fp64. The intermediate position and quaternion arrays were bitwise identical in every configuration. The no-reassociation mode retained FMA instructions.

In the tested unbatched CPU ANYmal scene, existing fast math changed the restored inertia and diverged during all ten replay steps. Disabling reassociation alone restored bitwise agreement in all recorded arrays and steps, in both precisions. Disabling contraction as well, and fully disabling fast math, also passed.

The tested CUDA Genesis scenes already replayed bitwise with existing fast math. They therefore do not demonstrate a CUDA Genesis replay improvement. This includes the 16,384-environment benchmark scene and an additional fp32 four-environment scene with an asymmetric initial orientation and joint targets.

## Reproduction

Each cell uses 20,000 matrices and seed 0, in a fresh process with both Quadrants compilation caches disabled.

| Backend | Precision | Existing fast math | No reassociation | No reassociation or contraction | Strict |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | fp32 | 18,387 | 0 | 0 | 0 |
| CPU | fp64 | 18,333 | 0 | 0 | 0 |
| CUDA | fp32 | 823 | 0 | 0 | 0 |
| CUDA | fp64 | 823 | 0 | 0 | 0 |

Entries are matrices with numerically different components. Maximum absolute differences in the baseline are 7.62939453125e-6, 2.842170943040401e-14, 1.9073486328125e-6, and 3.552713678800501e-15, respectively. The complete records are in `results/reproduction_cpu.json` and `results/reproduction_cuda.json`. A second complete matrix added direct byte comparisons: all restricted modes were byte-identical, with baseline bitwise-difference counts matching the table. See `results/bitwise_reproduction_*.json`; the additional CPU check ran on `cpu-208-039`.

In the `one_loop` module, the original LLVM IR contains 247 occurrences of `reassoc`; all restricted modes contain zero. CPU assembly contains 74 fused multiply-add/subtract instructions with existing fast math, 67 with reassociation disabled, and zero with contraction disabled or strict math. CUDA PTX counts are 96, 97, zero, and zero. These counts are the same in both precisions. See `results/codegen.json` and `inspect_codegen.py`. CPU assembly was emitted in separate diagnostic runs, which reproduced the numerical results.

## Genesis replay

The harness saves the scene after 60 steps, records ten continued steps, restores the saved state, and compares ten resumed steps. It copies and compares qpos, DOF velocity, link positions, and world-space inertia (`cinr_inertial`), including byte-level equality and finiteness. `scene.get_state()` is called before each recorded snapshot, on both trajectories.

For CPU ANYmal, the largest qpos difference over the ten baseline replay steps was 1.862645149230957e-9 in fp32 and 1.5482407023093003e-16 in fp64. All three restricted modes produced byte-identical snapshots. Repeating the baseline reproduced the same errors.

The initial two-box control passed in fp32 with both existing fast math and no reassociation on CPU and CUDA. The remaining box cells were stopped because this control did not expose the discrepancy. The completed records are retained in `results/boxes_*_partial.json`. Short preliminary ANYmal timing runs were also stopped and replaced by the longer timing matrix below. Their full logs remain in the artifact archive.

## Step timings

These are zero-target ANYmal measurements, not a run of the entire Genesis benchmark suite. CPU is unbatched (`n_envs=0`); CUDA uses 16,384 environments. Both use performance mode, deterministic algorithms, and a 25-iteration constraint-solver limit. The robot's 12 motor targets are zero with gains of 1,000. The baseline and each alternative use the same scene and configuration.

After the replay check, the harness warms up for 100 steps, then times five batches with synchronization at each batch boundary. Each CPU batch contains 10,000 steps; each CUDA batch contains 1,000. The table gives median step times in milliseconds and changes from the matching existing-fast-math baseline.

| Backend / precision | Existing fast math | No reassociation | No reassociation or contraction | Strict |
| --- | ---: | ---: | ---: | ---: |
| CPU fp32 | 0.066910 | 0.068132 (+1.83%) | 0.066441 (-0.70%) | 0.068919 (+3.00%) |
| CPU fp64 | 0.067914 | 0.069269 (+2.00%) | 0.067555 (-0.53%) | 0.067892 (-0.03%) |
| CUDA fp32 | 1.152458 | 1.152873 (+0.04%) | 1.161323 (+0.77%) | 1.242988 (+7.86%) |
| CUDA fp64 | 4.601931 | 4.594489 (-0.16%) | 5.557523 (+20.77%) | 5.651235 (+22.80%) |

The CPU samples overlap across several configurations; small differences should not be read as precise speed guarantees. A second baseline run gave medians of 0.066915 ms (fp32) and 0.067217 ms (fp64), compared with 0.066910 and 0.067914 ms initially.

CUDA fp64's first two samples were higher than its final three in every mode. The table uses all five samples consistently, and the full sequences are retained. The near-baseline no-reassociation result and the larger contraction/strict costs are also present in the final three samples. These measurements do not establish costs for other scenes, batch sizes, backends, or compiler versions.

See `results/timings.csv` for sample ranges, `results/benchmark_cpu.json` and `results/benchmark_cuda.json` for all samples and replay checks, and `results/benchmark_repeat_cpu.json` for the repeated baseline.

## Versions and environment

- Quadrants base: `2a2391b043a1afb61a4ccb9b790b3bd4934da9b5`.
- Compiled experimental Quadrants: `af5f72af7` (LLVM 22.1.0); subsequent commits changed the harness and documentation, not compiler behavior.
- Genesis: `439a26f56097c4178fbabf6a45f9d01662288be3`.
- Python: 3.12.14; NumPy: 2.5.5; PyTorch: 2.13.0+cu129.
- Primary CPU node: `cpu-207-199`, AMD EPYC 9454, 16 allocated CPUs.
- CUDA node: `rtx-209-191`, NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.71.05.
- Container image: `/mnt/data/images/genesis-v1_25.sqsh`.
- Environment setup followed Genesis's `production_build.sh` dependency recipe, relocating its venv into the assigned workspace, then installing the experimental Quadrants wheel.

## Reproduce the runs

The scripts use the workspace `/mnt/data/hugh/ais/handler_arch_1086_fast_math` on the cluster. Builds and CPU runs belong on CPU nodes; CUDA runs belong on a GPU allocation. Both use the container image above and mount `/mnt/data/hugh` at the same path. `build_cpu.sh` uses a private executable copy of the shared LLVM SDK's clang because the shared binary lacked execute permission.

From the Quadrants checkout, inside a CPU allocation:

```bash
bash experiments/issue1086/build_cpu.sh
bash experiments/issue1086/setup_genesis.sh
bash experiments/issue1086/run_stage.sh reproduce cpu
bash experiments/issue1086/run_stage.sh benchmark cpu --timing-steps 10000
```

Inside a CUDA allocation, with the same workspace mounted:

```bash
bash experiments/issue1086/run_stage.sh reproduce cuda
bash experiments/issue1086/run_stage.sh benchmark cuda --timing-steps 1000
```

To collect CPU assembly separately, use the compiler environment and run:

```bash
python experiments/issue1086/run_matrix.py --arch cpu --output /path/to/diagnostic_cpu --dump-code
```

The additional asymmetric CUDA baseline was:

```bash
bash experiments/issue1086/run_stage.sh replay cuda --scene anymal --excite \
  --precisions 32 --modes fast --output /path/to/asymmetric_cuda_baseline
```

To summarize the committed data and regenerate the timing CSV:

```bash
python experiments/issue1086/summarize.py experiments/issue1086/results/benchmark_*.json
python experiments/issue1086/export_timings.py \
  experiments/issue1086/results/benchmark_cpu.json \
  experiments/issue1086/results/benchmark_cuda.json \
  --output experiments/issue1086/results/timings.csv
```

The private environment switches and their limitations are documented in `README.md`. The experiment does not define a general bitwise-reproducibility contract or validate the untested AMDGPU switch.

## Artifacts and validation

The complete logs, IR/assembly dumps, package list, and exact compiled wheel are in `tmp/experiment_artifacts.tar.gz` in the assigned workspace, on both the cluster and the local workstation. The main logs are `tmp/output.log` and `tmp/gpu_output.log`. `results/metadata.json` records full source hashes and the artifact location. Both long-lived cluster allocations were released; the final CPU byte-check job exited automatically.

The C++ compiler build and every completed reproduction, diagnostic, long-benchmark, repeated-baseline, and asymmetric probe run succeeded. Repository-wide pre-commit checks passed. This was a controlled numerical experiment, not a full Quadrants or Genesis unit-test-suite run. The PR remains a draft because the environment controls are diagnostic and intentionally absent from cache keys.
