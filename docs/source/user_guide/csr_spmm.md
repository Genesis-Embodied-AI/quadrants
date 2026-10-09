# CSR sparse matrix multiplication

`qd.algorithms.csr_spmm` computes a weighted gather from source rows into receiver rows. It multiplies a sparse matrix stored in compressed sparse row (CSR) form by a dense matrix of source features: `out = A @ rhs`. Each stored edge has one scalar weight shared across feature channels. Call this experimental `@qd.func` at the top level of your own `@qd.kernel`.

For `A.shape == (M, N)`, the feature matrix has `rhs.shape == (N, K)` and the output has `out.shape == (M, K)`, with `K >= 1`. These matrices contain scalar float32 entries; they can have multiple feature columns. `K = 1` is the sparse matrix-vector multiplication (SpMV) specialization. The general operation is sparse matrix-matrix multiplication (SpMM).

Work is `O((M + E) * K)` for `E` stored edges, including writing every output entry. Compact CSR weight/index storage costs `O(M + E)`; overallocated buffers instead cost `O(M + C)` for edge capacity `C`. For a square graph with fixed incoming degree `d >= 1`, work is `O(N * d * K)` versus `O(N**2 * K)` for dense traversal.

The initial qualification targets ordinary contiguous `qd.ndarray` storage on CPU and Apple Metal, using `i32` indices and `f32` values, features, and output. Other backends and external framework arrays require separate qualification. This is a forward operation; it does not provide automatic differentiation.

## A small example

CSR stores each receiver's edges consecutively. `row_ptr[r]` and `row_ptr[r + 1]` delimit receiver `r`'s edges; `col_idx[e]` gives edge `e`'s source row and `values[e]` gives its weight. A receiver with equal start and end pointers has no edges.

```python
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, enable_fallback=False)  # Use qd.metal for Apple Metal.

row_ptr = qd.ndarray(qd.i32, shape=(4,))
col_idx = qd.ndarray(qd.i32, shape=(3,))
values = qd.ndarray(qd.f32, shape=(3,))
rhs = qd.ndarray(qd.f32, shape=(3, 2))
out = qd.ndarray(qd.f32, shape=(3, 2))

row_ptr.from_numpy(np.array([0, 2, 2, 3], dtype=np.int32))
col_idx.from_numpy(np.array([2, 0, 1], dtype=np.int32))
values.from_numpy(np.array([0.5, 2.0, -1.0], dtype=np.float32))
rhs.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))

@qd.kernel
def run(
    row_ptr: qd.types.NDArray[qd.i32, 1],
    col_idx: qd.types.NDArray[qd.i32, 1],
    values: qd.types.NDArray[qd.f32, 1],
    rhs: qd.types.NDArray[qd.f32, 2],
    out: qd.types.NDArray[qd.f32, 2],
):
    qd.algorithms.csr_spmm(row_ptr, col_idx, values, rhs, out)

run(row_ptr, col_idx, values, rhs, out)
print(out.to_numpy())
# [[ 4.5  7. ]
#  [ 0.   0. ]
#  [-3.  -4. ]]
```

Receiver 0 reads source 2 with weight `0.5` and source 0 with weight `2`. Receiver 1 is empty and receives zero. Receiver 2 reads source 1 with weight `-1`. Every invocation overwrites every output entry; previous output contents do not contribute.

## Shapes and caller preconditions

For `M` receivers, `N` sources, `K` feature channels, and allocated edge capacity `C`, call:

```python
qd.algorithms.csr_spmm(row_ptr, col_idx, values, rhs, out)
```

| Argument | Scalar dtype | Shape | Meaning |
|----------|--------------|-------|---------|
| `row_ptr` | `qd.i32` | `(M + 1,)` | Start and end of each receiver's incoming edges |
| `col_idx` | `qd.i32` | `(C,)` | Source row for each stored edge |
| `values` | `qd.f32` | `(C,)` | Weight for each stored edge |
| `rhs` | `qd.f32` | `(N, K)` | Source features |
| `out` | `qd.f32` | `(M, K)` | Receiver results, overwritten |

All four dimensions `M`, `N`, `K`, and `C` must be positive. Require `row_ptr[0] == 0`, nondecreasing row pointers, and `0 <= E = row_ptr[M] <= C`. Every active source index `col_idx[:E]` must lie in `[0, N)`. Edge capacities must match even when `E < C`.

Entries at edge positions `E` through `C - 1` are ignored. They may contain invalid source indices because they are never read. To represent an entirely empty graph, allocate at least one edge slot and set every row pointer to zero. Empty receiver rows produce zero in every feature channel.

Duplicate source indices remain distinct stored edges; their weighted contributions are added. Columns need not be sorted. The operation does not sort or coalesce edges, and does not change the inputs. You may update weights, source features, or valid topology before the next invocation, including changing `E` within allocated capacity.

Check dimensions, launch indexing, and buffer byte counts using Python integers before allocating. Every buffer must contain at most `2**31 - 1` bytes, and `M * K` must fit a positive signed 32-bit launch bound. This is a conservative supported bound: Metal's ndarray lowering computes a 32-bit flattened index and scales it to bytes before constructing a pointer. Since all scalar entries occupy four bytes, checking element counts alone is insufficient. Validate integer ranges before converting host indices to `np.int32`, so conversion cannot silently wrap an invalid index or pointer.

The function checks argument scalar dtypes and ranks at compilation. Cross-array shapes, CSR contents, byte limits, and non-aliasing are caller preconditions. It does not copy CSR data back to the host to validate a call. The checked preparation example in `quadrants.examples.csr_spmm` demonstrates host validation before upload and can save the actual inputs and results for reproduction.

Run the standalone example with `python -m quadrants.examples.csr_spmm --arch metal --save outputs/csr-spmm/example`. Replay those saved inputs with `python -m quadrants.examples.csr_spmm --arch cpu --load outputs/csr-spmm/example --save outputs/csr-spmm/replay`. Each saved directory contains `inputs.npz`, `outputs.npz`, `manifest.json`, and `execution.log`; the manifest records source/native identities, numerical options, platform/device information, and tolerances. Its independent dense reference is intended for small reproductions.

## Ownership and composition

Keep all inputs stable until their execution completes. `out` must not share storage with any input. The function allocates no buffers, takes no scratch argument, and performs no hidden host synchronization or graph preparation.

Place the call at the top level of a user kernel, with producer and consumer loops also at the top level. These phases execute as ordered device launches. The compiler rejects calls inside ordinary runtime loops or conditionals, including calls through another `@qd.func`. There is no grid-wide barrier inside `csr_spmm`. For a recurrent square graph, use separate previous and next state buffers, then exchange their roles between completed steps. In-place recurrence violates the ownership contract.

## Floating-point behavior

Weights, features, output, and each output accumulator use float32. Each output logically visits edges in their stored order. Compiler math options may transform floating-point arithmetic, so the interface does not promise strict summation order, bitwise agreement with a host reference, or bitwise agreement across backends.

Numerical qualification covers ordinary finite float32 inputs and arithmetic without intermediate overflow or underflow. Validate your own tolerance for long rows or heavy cancellation. A zero edge coefficient still causes a source-feature read; it is not a Boolean mask for nonfinite inputs.

## Batching and workload fit

If trials share both topology and weights, pack their feature matrices into the columns of one RHS: `rhs.shape == (N, B * K)` and `out.shape == (M, B * K)` for `B` trials. Each group of `K` columns remains an independent trial. Different weights per trial cannot be represented by column packing alone.

For independent graphs or weights, use block-diagonal CSR storage. Concatenate receiver rows, offset each trial's source indices by its source-row prefix, and offset row pointers by its active-edge prefix. Pack source features along their row dimension. This also supports different receiver/source counts per trial when they have a common channel width. Remove each trial's inactive edge padding when concatenating active edges; padding is allowed only after the combined logical edge count.

The operation can express signed spiking neural network (SNN) recurrence, fixed or irregular reservoir recurrence, multi-channel long-range aggregation, and independent or coupled linear graph recurrence. It does not skip inactive spikes, apply leak or activation, implement thresholds or plasticity, perform dense projections, or assign delay and coupling policies. Those steps belong in surrounding kernels. A regular-grid stencil requires its own comparison.

Useful performance depends on matrix dimensions, incoming-degree distribution, source locality, channel width, batch representation, data transfers, and graph reuse. Compare complete workloads against an appropriate dense or gather implementation; sparse storage alone does not establish a speedup.

## Recurrent ESN parity demonstration

The self-contained [ESN demonstration](../../../misc/demos/csr_spmm_esn.py) composes CSR multiplication with a leaky `tanh` echo state network (ESN), a ridge-fitted readout, and autonomous feedback. Each reservoir has its own fixed weights; batched trials use block-diagonal CSR storage. The comparison changes the recurrent multiply between CSR and a source-major dense matrix-vector kernel inside Quadrants. Both paths use float32, the same state updates and readout, separate previous/next buffers, and the same completion policy.

Run it from the repository root with NumPy and Quadrants installed:

```bash
python misc/demos/csr_spmm_esn.py --arch metal
python misc/demos/csr_spmm_esn.py --arch cpu
```

The default uses two independent 64-neuron reservoirs. To inspect a larger batch and save its inputs, traces, fitted readouts and JSON report, choose an existing output directory:

```bash
python misc/demos/csr_spmm_esn.py --arch metal --n 512 --batch 32 --degree 16 --washout 100 --train 1000 --test 300 --free 128 --save outputs/esn-parity.npz
```

The report shows three levels of numerical agreement: 256 successive matrix applications from a nonzero probe; driven and autonomous predictions with one shared readout; and held-out prediction scores from readouts fitted separately on each path's training states. With 1,000 training labels, the fits use prefixes of 64, 256 and 1,000 labels; smaller demonstrations choose smaller prefixes. Input normalization uses the full training inputs and remains fixed for all prefixes. Reservoir weights stay fixed throughout training.

The dense control visits every source column for each receiver, while CSR visits its stored edges. This demonstration uses the `K=1` SpMV specialization of the general SpMM primitive. Packing independent reservoirs into block-diagonal CSR increases the number of rows and sources while retaining one feature column. The dense control stores a separate square matrix per trial. Exact byte equality is reported as an observation for the actual inputs and options. Portable acceptance checks both paths against a float64 reference with stated absolute/relative bounds for chained products, driven states and predictions, and the first 16 autonomous steps. The full autonomous drift is also reported. The script exits unsuccessfully when an acceptance check fails.

The custom Mackey--Glass sequence uses a delayed differential equation integrated with four-stage Runge--Kutta (RK4), with interpolated delayed half stages. It supplies a prediction task for the numerical demonstration. Weights are scaled using NumPy's dense eigenvalue calculation for each reservoir, which can make large-batch fixture generation expensive.

### Comparing elapsed time

Add `--timing-blocks 5` to the command to collect warmed, alternating dense/CSR samples after numerical qualification. `--timing-warmup` defaults to two warmup blocks for each path and clock. The report retains every completed wall-clock sample, median, range and paired dense/CSR ratio. Fixture generation, initial allocation/upload, compilation and qualification sit outside the clocks. Timed outputs and input preservation are checked outside each clock.

The resident clock covers `min(256, washout + train + test)` teacher-driven updates with resident ping-pong states and no history writes; reset and final download sit outside it. The full-workflow clock includes reset, all teacher-driven states, state download, a float64 ridge fit using all training labels, readout upload, driven and autonomous evaluation, and prediction downloads. Both paths complete every 32 updates and at the end of each scope. Per-trial time is the completed batch time divided by trial count.

Vary `--batch` to compare many independent trials and `--degree` to examine density crossovers, while retaining the other parameters and seeds. These clocks compare the authored source-major dense `K=1` kernel with CSR inside Quadrants; performance depends on the machine, degree, trial count and chosen clock. Timing is disabled by default, and tests check numerical behavior without asserting a speed threshold.

```{toctree}
:hidden:

/autoapi/quadrants/examples/csr_spmm/index
```
