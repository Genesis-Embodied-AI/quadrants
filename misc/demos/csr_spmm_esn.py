"""Compare dense K=1 and CSR ESNs on a custom Mackey--Glass prediction task.

    python misc/demos/csr_spmm_esn.py --arch metal --n 64 --batch 2

Both paths use resident float32 ping-pong states and the same leak/tanh/readout.
Three comparisons cover 256 chained products, a shared host-fitted readout,
and independently fitted learning curves. JSON reports tolerances and drift;
--save writes the actual frozen inputs and output traces to NPZ for inspection.
--timing-blocks 5 adds alternating warmed dense/CSR completed wall-clock samples
for resident recurrence and the full teacher-state/fit/prediction workflow.
Generation uses NumPy dense eigvals per trial and can be expensive for large N.
This custom delayed-RK4 task supplies an executable parity example, with no
claim to reproduce published prediction scores. No runtime is initialized on import.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, make_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

import quadrants as qd


@dataclass(frozen=True)
class Config:
    n: int = 64
    batch: int = 2
    degree: int = 8
    seed: int = 7
    washout: int = 32
    train: int = 128
    test: int = 64
    free: int = 32
    leak: float = 0.3
    ridge: float = 1e-3

    @property
    def end(self):
        return self.washout + self.train

    @property
    def steps(self):
        return self.end + self.test

    def validate(self):
        for key in ("n", "batch", "degree", "seed", "washout", "train", "test", "free"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not (3 <= self.n <= 2048 and self.batch >= 1 and 1 <= self.degree <= self.n):
            raise ValueError("Require 3 <= n <= 2048, batch >= 1, 1 <= degree <= n")
        if self.seed < 0 or self.washout < 0 or self.train < 4 or not 2 <= self.free <= self.test:
            raise ValueError("Require seed/washout >= 0, train >= 4, and 2 <= free <= test")
        if not np.isfinite([self.leak, self.ridge]).all() or not (0 < self.leak <= 1 and self.ridge > 0):
            raise ValueError("Require finite 0 < leak <= 1 and ridge > 0")
        # Conservative total host/device workspaces, including traces and ridge copies.
        estimate = self.batch * (24 * self.n**2 + 48 * (self.steps + 256) * (self.n + 2))
        # Host CSR plus both device paths coexist, including the incoming second path.
        estimate += 24 * self.batch * self.n * self.degree + 96 * self.batch * self.n + 12
        estimate += 32 * min(self.train, self.n + 2) ** 2 + 80 * (1000 + self.steps)
        if estimate > 2 << 30 or self.batch * self.n * self.degree >= 1 << 31:
            raise MemoryError("Demo exceeds its conservative 2 GiB workspace or int32 index budget")


def mackey_glass(samples):
    """h=0.1 RK4; delay 17, constant history 1.2, burn 1000, unit sampling.

    Delayed half stages linearly interpolate stored history. This approximation
    makes the overall delay discretization lower order than ordinary RK4.
    """
    values = np.empty((1000 + samples - 1) * 10 + 1, dtype=np.float64)
    values[0] = 1.2

    def delayed(position):
        if position <= 0:
            return 1.2
        index, fraction = int(position), position % 1
        return values[index] if fraction == 0 else (values[index] + values[index + 1]) / 2

    def derivative(value, past):
        return 0.2 * past / (1 + past**10) - 0.1 * value

    for t in range(len(values) - 1):
        old = values[t]
        d0, dm, d1 = (delayed(t - 170 + offset) for offset in (0, 0.5, 1))
        k1 = derivative(old, d0)
        k2 = derivative(old + 0.05 * k1, dm)
        k3 = derivative(old + 0.05 * k2, dm)
        k4 = derivative(old + 0.1 * k3, d1)
        values[t + 1] = old + 0.1 * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return values[10000::10].copy()


def fixture(c):
    """Independent sorted unique incoming sources; freeze all model arrays in f32."""
    c.validate()
    n, b, d = c.n, c.batch, c.degree
    dense = np.zeros((b, n, n), dtype=np.float32)
    columns = np.empty((b, n, d), dtype=np.int32)
    values = np.empty((b, n, d), dtype=np.float32)
    win, bias = np.empty((b, n), np.float32), np.empty((b, n), np.float32)
    radii = []
    for trial, child in enumerate(np.random.SeedSequence(c.seed).spawn(b)):
        rng = np.random.default_rng(child)
        local = np.array([np.sort(rng.choice(n, d, replace=False)) for _ in range(n)], np.int32)
        weights = rng.uniform(-1, 1, (n, d))
        matrix = np.zeros((n, n), np.float64)
        matrix[np.arange(n)[:, None], local] = weights
        radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Reservoir requires a positive finite spectral radius")
        values[trial] = weights * (0.9 / radius)
        columns[trial] = local + trial * n
        dense[trial, np.arange(n)[:, None], local] = values[trial]
        radii.append(float(np.max(np.abs(np.linalg.eigvals(dense[trial].astype(np.float64))))))
        if abs(radii[-1] - 0.9) > 1e-4:
            raise ValueError("Float32 reservoir spectral radius differs from requested 0.9")
        win[trial], bias[trial] = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.2, 0.2, n)
    sequence = mackey_glass(c.steps + 1)
    mean, scale = sequence[c.washout : c.end].mean(), sequence[c.washout : c.end].std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training input scale must be positive and finite")
    normalized = ((sequence - mean) / scale).astype(np.float32)
    probe_rng = np.random.default_rng(np.random.SeedSequence(c.seed, spawn_key=(0x4C494E,)))
    arrays = dict(
        row_ptr=np.arange(b * n + 1, dtype=np.int32) * d,
        col_idx=columns.ravel(),
        values=values.ravel(),
        dense=dense,
        win=win,
        bias=bias,
        probe=probe_rng.standard_normal((b, n)).astype(np.float32),
        inputs=np.repeat(normalized[:-1, None], b, axis=1),
        targets=np.repeat(normalized[1:, None], b, axis=1),
    )
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Fixture contains nonfinite entries")
    metadata = dict(
        normalization=dict(mean=float(mean), scale=float(scale), fit="training inputs", ddof=0),
        frozen_spectral_radii=radii,
        spectral_estimator="NumPy dense eigvals, every trial",
    )
    return arrays, metadata


def features(inputs, states):
    return np.concatenate((np.ones((*inputs.shape, 1)), inputs[..., None], states), axis=2).astype(np.float64)


def fit(c, arrays, states, length):
    """Float64 ridge, all features penalized, followed by one float32 cast."""
    section = slice(c.washout, c.washout + length)
    design = features(arrays["inputs"][section], states[section])
    result = []
    for trial in range(c.batch):
        x, y = design[:, trial], arrays["targets"][section, trial].astype(np.float64)
        primal = x.shape[1] <= x.shape[0]
        gram = x.T @ x if primal else x @ x.T
        gram.flat[:: len(gram) + 1] += c.ridge
        solved = np.linalg.solve(gram, x.T @ y if primal else y)
        result.append(solved if primal else x.T @ solved)
    return np.asarray(result, dtype=np.float32)


def oracle(c, arrays):
    matrix, win, bias = (arrays[key].astype(np.float64) for key in ("dense", "win", "bias"))
    leak = float(np.float32(c.leak))
    retain = float(np.float32(np.float32(1) - np.float32(c.leak)))

    def product(x):
        return np.einsum("bij,bj->bi", matrix, x)

    def step(x, u):
        return retain * x + leak * np.tanh((product(x) + win * u[:, None]) + bias)

    linear, states = [], []
    x = arrays["probe"].astype(np.float64)
    for _ in range(256):
        x = product(x)
        linear.append(x)
    x = np.zeros((c.batch, c.n), np.float64)
    for u in arrays["inputs"]:
        x = step(x, u.astype(np.float64))
        states.append(x)
    states = np.array(states)

    def evaluate(coefficients):
        coefficients = coefficients.astype(np.float64)
        predictions = np.einsum("tbf,bf->tb", features(arrays["inputs"][c.end :], states[c.end :]), coefficients)
        x = states[c.end - 1].copy()
        u = np.einsum("bf,bf->b", features(arrays["inputs"][c.end - 1 : c.end], x[None])[0], coefficients)
        free = []
        for _ in range(c.free):
            x = step(x, u)
            u = coefficients[:, 0] + coefficients[:, 1] * u + np.sum(coefficients[:, 2:] * x, axis=1)
            free.append(u)
        return dict(predictions=predictions, free=np.array(free))

    return dict(linear=np.array(linear), states=states), evaluate


def device(c, arrays, dense, *, runtime=None):
    """Construct two paths in one initialized runtime; caller owns qd.init()."""

    def upload(array):
        result = qd.ndarray(qd.i32 if array.dtype == np.int32 else qd.f32, array.shape)
        result.from_numpy(np.ascontiguousarray(array))
        return result

    inputs = {key: upload(value) for key, value in arrays.items() if key != "dense" and key != "targets"}
    inputs["dense"] = upload(arrays["dense"].transpose(0, 2, 1))  # [trial, source, receiver]
    rows = c.batch * c.n
    a, b, recurrent = (qd.ndarray(qd.f32, (rows, 1)) for _ in range(3))
    states = qd.ndarray(qd.f32, (c.steps, c.batch, c.n))
    trace = qd.ndarray(qd.f32, (256, c.batch, c.n))
    predictions = qd.ndarray(qd.f32, (c.test, c.batch))
    free = qd.ndarray(qd.f32, (c.free, c.batch))
    feedback = qd.ndarray(qd.f32, (c.batch,))
    coefficients = qd.ndarray(qd.f32, (c.batch, c.n + 2))
    buffers = dict(
        inputs,
        a=a,
        b=b,
        recurrent=recurrent,
        states=states,
        trace=trace,
        predictions=predictions,
        free=free,
        feedback=feedback,
        coefficients=coefficients,
    )
    # Template traversal passes each ndarray as a real kernel argument, retaining ownership.
    bound_type = qd.data_oriented(make_dataclass("ESNBuffers", [(key, object) for key in buffers], frozen=True))
    bound = bound_type(**buffers)
    leak, retain = float(np.float32(c.leak)), float(np.float32(np.float32(1) - np.float32(c.leak)))

    @qd.kernel
    def advance(
        bound: qd.template(), previous: qd.types.NDArray, following: qd.types.NDArray, t: qd.i32, mode: qd.template()
    ):
        if qd.static(dense):
            for row in range(rows):
                trial, neuron = row // c.n, row % c.n
                total = qd.f32(0.0)
                for source in range(c.n):
                    total += bound.dense[trial, source, neuron] * previous[trial * c.n + source, 0]
                bound.recurrent[row, 0] = total
        else:
            qd.algorithms.csr_spmm(bound.row_ptr, bound.col_idx, bound.values, previous, bound.recurrent)
        for row in range(rows):
            trial, neuron = row // c.n, row % c.n
            value = bound.recurrent[row, 0]
            if qd.static(mode != 0):
                u = qd.f32(0.0)
                if qd.static(mode == 1 or mode == 3):
                    u = bound.inputs[t, trial]
                else:
                    u = bound.feedback[trial]
                value = retain * previous[row, 0] + leak * qd.tanh(
                    (value + bound.win[trial, neuron] * u) + bound.bias[trial, neuron]
                )
            following[row, 0] = value
            if qd.static(mode == 1):
                bound.states[t, trial, neuron] = value
            elif qd.static(mode == 0):
                bound.trace[t, trial, neuron] = value
        if qd.static(mode == 2):
            for trial in range(c.batch):
                total = bound.coefficients[trial, 0] + bound.coefficients[trial, 1] * bound.feedback[trial]
                for neuron in range(c.n):
                    total += bound.coefficients[trial, neuron + 2] * following[trial * c.n + neuron, 0]
                bound.free[t, trial] = total
                bound.feedback[trial] = total

    @qd.kernel
    def read_and_seed(bound: qd.template()):
        for row in range(rows):
            bound.a[row, 0] = bound.states[c.end - 1, row // c.n, row % c.n]
        for offset, trial in qd.ndrange(c.test + 1, c.batch):
            t = c.end - 1 + offset
            total = bound.coefficients[trial, 0] + bound.coefficients[trial, 1] * bound.inputs[t, trial]
            for neuron in range(c.n):
                total += bound.coefficients[trial, neuron + 2] * bound.states[t, trial, neuron]
            if offset == 0:
                bound.feedback[trial] = total  # Unscored last-training readout seeds model feedback.
            else:
                bound.predictions[offset - 1, trial] = total

    def rollout(count, mode):
        previous, following = a, b
        for t in range(count):
            advance(bound, previous, following, t, mode)
            previous, following = following, previous
            if (t + 1) % 32 == 0:
                qd.sync()
        qd.sync()
        return previous

    a.from_numpy(arrays["probe"].reshape(rows, 1))
    rollout(256, 0)
    linear = trace.to_numpy()
    a.fill(0)
    rollout(c.steps, 1)

    def evaluate(readout):
        coefficients.from_numpy(readout)
        read_and_seed(bound)
        rollout(c.free, 2)
        return dict(predictions=predictions.to_numpy(), free=free.to_numpy())

    if runtime is not None:

        def workflow():
            a.fill(0)
            rollout(c.steps, 1)
            teacher = states.to_numpy()
            readout = fit(c, arrays, teacher, c.train)
            return dict(states=teacher, readout=readout, **evaluate(readout))

        def check_inputs():
            return {
                key: bool(
                    np.array_equal(value.to_numpy(), arrays[key].transpose(0, 2, 1) if key == "dense" else arrays[key])
                )
                for key, value in inputs.items()
            }

        runtime.update(
            reset=lambda: a.fill(0),
            resident=lambda: rollout(min(256, c.steps), 3),
            read_resident=lambda last: last.to_numpy().reshape(c.batch, c.n),
            workflow=workflow,
            check_inputs=check_inputs,
        )

    return dict(linear=linear, states=states.to_numpy()), evaluate, inputs


def compare(actual, expected, atol, rtol):
    error = np.abs(actual.astype(np.float64) - expected)
    budget = atol + rtol * np.abs(expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return dict(
        passed=finite and bool(np.all(error <= budget)),
        atol=atol,
        rtol=rtol,
        max_abs=float(error.max()),
        max_tolerance_fraction=float((error / budget).max()),
        byte_equal=actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes(),
    )


def score(actual, target):
    rmse = np.sqrt(np.mean((actual.astype(np.float64) - target) ** 2, axis=0))
    return dict(rmse=rmse.tolist(), nrmse=(rmse / target.astype(np.float64).std(axis=0)).tolist())


def validate_timing(blocks, warmup):
    if type(blocks) is not int or (blocks != 0 and not 3 <= blocks <= 15):
        raise ValueError("timing_blocks must be 0 (disabled) or between 3 and 15")
    if type(warmup) is not int or not 1 <= warmup <= 10:
        raise ValueError("timing_warmup must be between 1 and 10")


def measure(c, runtimes, saved, blocks, warmup):
    """Validate the actual warmed/timed results after completed clock boundaries."""
    count = min(256, c.steps)
    integrity_before = {name: runtime["check_inputs"]() for name, runtime in runtimes.items()}
    scopes = {}
    for scope in ("resident", "workflow"):
        samples = {name: [] for name in runtimes}
        checks = {name: [] for name in runtimes}
        orders = []
        for block in range(warmup + blocks):
            # Restart alternation for recorded blocks, independent of warmup count.
            index = block if block < warmup else block - warmup
            order = ("dense", "csr") if index % 2 == 0 else ("csr", "dense")
            if block >= warmup:
                orders.append(list(order))
            for name in order:
                runtime = runtimes[name]
                if scope == "resident":
                    runtime["reset"]()
                qd.sync()
                start = perf_counter()
                output = runtime[scope]()
                qd.sync()
                seconds = perf_counter() - start
                if scope == "resident":
                    actual = dict(states=runtime["read_resident"](output))
                    expected = dict(states=saved[f"{name}_states"][count - 1])
                else:
                    actual = output
                    expected = dict(
                        states=saved[f"{name}_states"],
                        readout=saved[f"{name}_readout_{c.train}"],
                        **{key: saved[f"{name}_own_{c.train}_{key}"] for key in ("predictions", "free")},
                    )
                checks[name].append(
                    dict(
                        phase="warmup" if block < warmup else "timed",
                        block=index,
                        outputs={key: compare(value, expected[key], 3e-5, 3e-4) for key, value in actual.items()},
                        finite_positive_seconds=bool(np.isfinite(seconds) and seconds > 0),
                    )
                )
                if block >= warmup:
                    samples[name].append(seconds)
        summary = {}
        for name, values in samples.items():
            median = float(np.median(values))
            summary[name] = dict(
                seconds=values,
                median_seconds=median,
                min_seconds=min(values),
                max_seconds=max(values),
                amortized_median_seconds_per_trial=median / c.batch,
            )
            if scope == "resident":
                summary[name]["median_seconds_per_step"] = median / count
        ratios = [dense / csr for dense, csr in zip(samples["dense"], samples["csr"])]
        scopes[scope] = dict(
            **summary,
            order_by_block=orders,
            paired_dense_over_csr=ratios,
            median_paired_dense_over_csr=float(np.median(ratios)),
            ratio_of_medians=summary["dense"]["median_seconds"] / summary["csr"]["median_seconds"],
            checks=checks,
        )
    integrity_after = {name: runtime["check_inputs"]() for name, runtime in runtimes.items()}
    passed = all(ok for stage in (integrity_before, integrity_after) for path in stage.values() for ok in path.values())
    passed = passed and all(
        check["finite_positive_seconds"] and all(output["passed"] for output in check["outputs"].values())
        for scope in scopes.values()
        for path in scope["checks"].values()
        for check in path
    )
    return dict(
        passed=passed,
        blocks=blocks,
        warmup_blocks=warmup,
        resident_steps=count,
        sample_policy="One completed rollout/workflow per path per block; raw wall-clock seconds",
        synchronization="Every 32 steps and completed scope boundary",
        resident_boundary="Zero reset before clock; teacher recurrence without history writes; download after clock",
        workflow_boundary=(
            "Zero reset, all teacher states and download, final-length host ridge fit, "
            "own readout upload, driven/free evaluation and output download"
        ),
        exclusions="Initial fixture generation, device allocation/upload, parity qualification and warmup",
        per_trial_policy="Elapsed batch time divided by batch size; trials execute together",
        input_integrity=dict(before=integrity_before, after=integrity_after),
        **scopes,
    )


def run(c, *, timing_blocks=0, timing_warmup=2):
    validate_timing(timing_blocks, timing_warmup)
    arrays, metadata = fixture(c)
    reference, host_evaluate = oracle(c, arrays)
    shared = fit(c, arrays, reference["states"], c.train)
    reference.update(host_evaluate(shared))
    report, saved, paths, runtimes = {}, dict(arrays, shared_readout=shared), {}, {}
    for name, is_dense in (("dense", True), ("csr", False)):
        runtime = {} if timing_blocks else None
        result, evaluate, _ = device(c, arrays, is_dense, runtime=runtime)
        if timing_blocks:
            runtimes[name] = runtime
        result.update(evaluate(shared))
        checks = {
            key: compare(result[key], reference[key], atol, rtol)
            for key, atol, rtol in (("linear", 3e-5, 3e-4), ("states", 3e-5, 3e-4), ("predictions", 1e-4, 1e-3))
        }
        checks["free_first16"] = compare(result["free"][:16], reference["free"][:16], 3e-4, 3e-3)
        curves = []
        for length in sorted({min(64, max(2, c.train // 4)), min(256, c.train // 2), c.train}):
            fitted = fit(c, arrays, result["states"], length)
            own = evaluate(fitted)
            curves.append(
                dict(
                    training_samples=length,
                    driven=score(own["predictions"], arrays["targets"][c.end :]),
                    free=score(own["free"], arrays["targets"][c.end : c.end + c.free]),
                )
            )
            saved[f"{name}_readout_{length}"] = fitted
            saved.update({f"{name}_own_{length}_{key}": value for key, value in own.items()})
        linear_error = np.linalg.norm((result["linear"] - reference["linear"]).reshape(256, -1), axis=1)
        report[name] = dict(
            checks=checks,
            free_full_max_abs=float(np.max(np.abs(result["free"] - reference["free"]))),
            free_checked_steps=min(16, c.free),
            linear_l2_error_by_step=linear_error.tolist(),
            linear_l2_by_step=np.linalg.norm(result["linear"].reshape(256, -1), axis=1).tolist(),
            shared_driven_score=score(result["predictions"], arrays["targets"][c.end :]),
            shared_free_score=score(result["free"], arrays["targets"][c.end : c.end + c.free]),
            learning_curve=curves,
        )
        paths[name] = result
        saved.update({f"{name}_{key}": value for key, value in result.items()})
    report["dense_vs_csr"] = {key: compare(paths["csr"][key], paths["dense"][key], 3e-4, 3e-3) for key in reference}
    report["learning_curve_rmse_differences"] = [
        dict(
            training_samples=dense["training_samples"],
            **{key: (np.array(csr[key]["rmse"]) - dense[key]["rmse"]).tolist() for key in ("driven", "free")},
        )
        for dense, csr in zip(report["dense"]["learning_curve"], report["csr"]["learning_curve"])
    ]
    report["linear_reference_l2_by_step"] = np.linalg.norm(reference["linear"].reshape(256, -1), axis=1).tolist()
    report["passed"] = all(check["passed"] for name in ("dense", "csr") for check in report[name]["checks"].values())
    if timing_blocks:
        if report["passed"]:
            report["timing"] = measure(c, runtimes, saved, timing_blocks, timing_warmup)
            report["passed"] = report["timing"]["passed"]
        else:
            report["timing"] = dict(passed=False, skipped="Numerical parity checks failed")
    saved.update({f"reference_{key}": value for key, value in reference.items()})
    cfg = qd.lang.impl.current_cfg()
    metadata.update(
        config=asdict(c),
        numpy=np.__version__,
        quadrants=str(qd.__version__),
        platform=platform.platform(),
        arch=str(cfg.arch),
        default_fp=str(cfg.default_fp),
        fast_math=bool(cfg.fast_math),
        build_commit=qd._lib.core.get_commit_hash(),
        native_module=str(Path(qd._lib.core.__file__).resolve()),
        recipe="MG delayed RK4 h=0.1; leak=f32(config.leak); radius=0.9; ridge f64, all coefficients penalized",
        prefix_policy="Fixed full-training normalization and fixed held-out interval for every prefix",
    )
    metadata["fixture_sha256"] = hashlib.sha256(b"".join(arrays[key].tobytes() for key in sorted(arrays))).hexdigest()
    metadata["native_sha256"] = hashlib.sha256(Path(metadata["native_module"]).read_bytes()).hexdigest()
    metadata["demo_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(metadata=metadata, results=report), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "metal"), default="cpu")
    for key, value in asdict(Config()).items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parser.add_argument("--save", type=Path, help="Optional NPZ traces plus adjacent JSON report")
    parser.add_argument(
        "--timing-blocks", type=int, default=0, help="Paired timing blocks: 0 disables; 3 to 15 enables"
    )
    parser.add_argument("--timing-warmup", type=int, default=2, help="Warmup blocks per timing scope (1 to 10)")
    args = parser.parse_args()
    validate_timing(args.timing_blocks, args.timing_warmup)
    c = Config(**{key: getattr(args, key) for key in asdict(Config())})
    c.validate()
    qd.init(arch=getattr(qd, args.arch), enable_fallback=False, default_fp=qd.f32, fast_math=False)
    report, arrays = run(c, timing_blocks=args.timing_blocks, timing_warmup=args.timing_warmup)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.save:
        with args.save.open("wb") as handle:
            np.savez_compressed(handle, **arrays, metadata=np.array(text))
        args.save.with_suffix(".json").write_text(text + "\n")
    print(text)
    if not report["results"]["passed"]:
        raise SystemExit("ESN numerical parity checks failed")


if __name__ == "__main__":
    main()
