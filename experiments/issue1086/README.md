# Issue 1086 controlled experiment

This branch adds private diagnostic environment variables, not a supported public API. Both default to off. Set them before initializing Quadrants in a fresh process, with `offline_cache=False` and `src_ll_cache=False`; the switches are deliberately not part of compilation cache keys.

| Mode | `fast_math` | `QD_EXPERIMENT_NO_REASSOC` | `QD_EXPERIMENT_NO_CONTRACT` |
| --- | --- | --- | --- |
| fast | True | 0 | 0 |
| no_reassoc | True | 1 | 0 |
| no_reassoc_no_contract | True | 1 | 1 |
| strict | False | 0 | 0 |

The reassociation switch removes LLVM's `reassoc` flag and disables floating-point expression regrouping in Quadrants' `BinaryOpSimp`. Other fast-math simplifications, approximate functions, and floating-point assumptions remain enabled. The contraction switch changes LLVM target `AllowFPOpFusion` from `Fast` to `Strict` on CPU, CUDA, and AMDGPU. It does not remove explicit fused-multiply-add intrinsics in library routines. This experiment measures CPU and CUDA only.

`reproduce.py` adapts the issue's reproduction to one backend, precision, and mode per process. It records whether the intermediate poses are bitwise equal, and the number of output matrices with numerically differing components. Changing precision also changes all fields and the identity matrix to the requested type.

```bash
python experiments/issue1086/reproduce.py --arch cpu --precision 64 --mode no_reassoc
```

Repeat for `--arch cuda`, precisions `32` and `64`, and all four modes. Each successful run prints one JSON record prefixed by `RESULT`. An intermediate-input mismatch is reported explicitly rather than attributed to the inertia calculation.
