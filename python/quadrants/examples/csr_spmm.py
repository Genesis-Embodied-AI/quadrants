"""Checked preparation for experimental CSR SpMM; run with ``--arch cpu`` or ``--arch metal``.

The private host helpers demonstrate caller-side validation and independent numerical references. They are not
part of the public Quadrants API. Validate before uploading, and revalidate whenever topology or shapes change.
"""

import argparse
import contextlib
import hashlib
import io
import json
import operator
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import quadrants as qd
from quadrants._lib import core

_INT32_MAX = 2**31 - 1


def _check_buffer_shape(name, shape, itemsize=4):
    """Check Python-integer size arithmetic before allocating a buffer; return its byte count."""
    count = 1
    if not shape:
        raise ValueError(f"{name}: shape must have at least one dimension")
    for dimension in shape:
        if isinstance(dimension, (bool, np.bool_)):
            raise ValueError(f"{name}: dimensions must be positive integers")
        try:
            dimension = operator.index(dimension)
        except TypeError as exc:
            raise ValueError(f"{name}: dimensions must be positive integers") from exc
        if dimension <= 0 or dimension > _INT32_MAX:
            raise ValueError(f"{name}: dimensions must be positive and fit INT32_MAX")
        count *= dimension
    itemsize = operator.index(itemsize)
    if itemsize <= 0:
        raise ValueError(f"{name}: item size must be positive")
    byte_count = count * itemsize
    if byte_count > _INT32_MAX:
        raise ValueError(f"{name}: byte count {byte_count} exceeds INT32_MAX")
    return byte_count


def _checked_i32(array, name):
    """Convert integer host data only after checking its original values, avoiding wraparound."""
    array = np.asarray(array)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name}: expected integer data before i32 conversion")
    _check_buffer_shape(name, array.shape)
    if array.size and (np.min(array) < -(2**31) or np.max(array) > _INT32_MAX):
        raise ValueError(f"{name}: values must fit signed i32 before conversion")
    return np.ascontiguousarray(array, dtype=np.int32)


def _validate_csr(row_ptr, col_idx, values, rhs, out):
    """Validate host storage and active CSR data, returning ``(M, N, K, E)``.

    Host non-aliasing is necessary; callers must also keep the separately allocated device output disjoint from
    device inputs. Inactive column/weight storage may contain poison values and is deliberately not inspected.
    Finite inputs alone do not guarantee absence of intermediate overflow or underflow.
    """
    arrays = (row_ptr, col_idx, values, rhs, out)
    names = ("row_ptr", "col_idx", "values", "rhs", "out")
    ranks = (1, 1, 1, 2, 2)
    dtypes = (np.int32, np.int32, np.float32, np.float32, np.float32)
    for name, array, rank, dtype in zip(names, arrays, ranks, dtypes):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name}: expected a NumPy array")
        if array.ndim != rank:
            raise ValueError(f"{name}: expected rank {rank}")
        if array.dtype != np.dtype(dtype):
            raise TypeError(f"{name}: expected {np.dtype(dtype)}")
        _check_buffer_shape(name, array.shape, array.dtype.itemsize)
        if not array.flags.c_contiguous:
            raise ValueError(f"{name}: expected C-contiguous storage")
    if not out.flags.writeable:
        raise ValueError("out: expected writable storage")
    m, k = out.shape
    n = rhs.shape[0]
    if row_ptr.shape != (m + 1,):
        raise ValueError("row_ptr: length must equal out rows + 1")
    if rhs.shape[1] != k:
        raise ValueError("rhs and out: feature widths must match")
    if col_idx.shape != values.shape:
        raise ValueError("col_idx and values: allocated capacities must match")
    if row_ptr[0] != 0 or np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("row_ptr: must start at zero and be nondecreasing")
    e = int(row_ptr[-1])
    if e < 0 or e > col_idx.size:
        raise ValueError("row_ptr: logical edge count must lie within allocated capacity")
    if np.any(col_idx[:e] < 0) or np.any(col_idx[:e] >= n):
        raise ValueError("col_idx: active source indices must lie within rhs rows")
    if not np.all(np.isfinite(values[:e])) or not np.all(np.isfinite(rhs)):
        raise ValueError("values and rhs: active operands must be finite")
    for name, array in zip(names[:-1], arrays[:-1]):
        if np.shares_memory(out, array):
            raise ValueError(f"out: must not overlap {name}")
    return m, n, k, e


def _reference_ordered_f32(row_ptr, col_idx, values, rhs):
    """Round each product and addition to f32 in stored-edge order, independently of device compilation."""
    result = np.zeros((len(row_ptr) - 1, rhs.shape[1]), dtype=np.float32)
    for row in range(result.shape[0]):
        for edge in range(int(row_ptr[row]), int(row_ptr[row + 1])):
            product = np.multiply(values[edge], rhs[col_idx[edge]], dtype=np.float32)
            np.add(result[row], product, out=result[row])
    return result


def _reference_dense_f64(row_ptr, col_idx, values, rhs):
    """Build a duplicate-preserving dense f64 matrix from actual f32 inputs; use only for small fixtures."""
    dense = np.zeros((len(row_ptr) - 1, rhs.shape[0]), dtype=np.float64)
    for row in range(dense.shape[0]):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        np.add.at(dense[row], col_idx[start:end], values[start:end].astype(np.float64))
    return dense @ rhs.astype(np.float64)


def _error_budget(row_ptr, col_idx, values, rhs):
    """Predeclared difficult-fixture budget; not a universal theorem for relaxed backend math."""
    result = np.empty((len(row_ptr) - 1, rhs.shape[1]), dtype=np.float64)
    u = 2.0**-24
    for row in range(result.shape[0]):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        nu = 2 * (end - start) * u
        if nu >= 1:
            raise ValueError("error budget requires 2 * degree * 2**-24 < 1")
        magnitude = np.zeros(rhs.shape[1], dtype=np.float64)
        for edge in range(start, end):
            magnitude += np.abs(np.float64(values[edge]) * rhs[col_idx[edge]].astype(np.float64))
        result[row] = 4 * nu / (1 - nu) * magnitude + 2e-6
    return result


def _upload(array):
    dtype = qd.i32 if array.dtype == np.int32 else qd.f32
    result = qd.ndarray(dtype, shape=array.shape)
    result.from_numpy(array)  # pylint: disable=no-member
    return result


def _file_hash(path):
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command, cwd=None):
    try:
        process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def _reproducer_metadata(arch):
    """Record source/native identity and numerical options without enabling profiling or changing math flags."""
    assert qd.cfg is not None
    source_root = Path(__file__).resolve().parents[3]
    feature_files = (
        Path(__file__).resolve(),
        source_root / "python/quadrants/algorithms/_csr.py",
        source_root / "python/quadrants/algorithms/__init__.py",
    )
    native_path = getattr(core, "__file__", None)
    flags = (
        "default_fp",
        "default_ip",
        "fast_math",
        "advanced_optimization",
        "opt_level",
        "offline_cache",
    )
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_arch": arch,
        "actual_arch": str(qd.cfg.arch),
        "enable_fallback": False,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "quadrants_version": qd.__version_str__,
        "quadrants_python_path": qd.__file__,
        "native_path": native_path,
        "native_sha256": _file_hash(native_path),
        "native_build_commit": core.get_commit_hash(),
        "source_revision": _command_output(["git", "rev-parse", "HEAD"], cwd=source_root),
        "feature_sha256": {str(path): _file_hash(path) for path in feature_files},
        "compile_options": {name: str(getattr(qd.cfg, name)) for name in flags},
        "tolerances": {"rtol": 2e-5, "atol": 2e-6},
        "qualification": "finite float32 arithmetic without intermediate overflow or underflow",
    }
    if platform.system() == "Darwin":
        metadata["cpu_model"] = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        displays = _command_output(["system_profiler", "SPDisplaysDataType", "-json"])
        if displays:
            try:
                metadata["gpu_models"] = [
                    item.get("sppci_model", item.get("_name"))
                    for item in json.loads(displays).get("SPDisplaysDataType", [])
                ]
            except (TypeError, ValueError):
                metadata["gpu_models"] = None
    return metadata


def _demo_inputs():
    row_ptr = _checked_i32([0, 3, 3, 4], "row_ptr")
    col_idx = _checked_i32([1, 0, 1, 2, -999], "col_idx")
    values = np.array([0.5, -1, 0.25, 2, np.nan], dtype=np.float32)
    rhs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    out = np.full((3, 2), np.nan, dtype=np.float32)
    return row_ptr, col_idx, values, rhs, out


def _load_inputs(path):
    path = Path(path)
    if path.is_dir():
        path = path / "inputs.npz"
    with np.load(path, allow_pickle=False) as saved:
        return tuple(saved[name].copy() for name in ("row_ptr", "col_idx", "values", "rhs", "out_initial"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "metal"), default="cpu")
    parser.add_argument(
        "--save",
        type=Path,
        help="Save actual inputs, outputs, environment manifest, and execution log.",
    )
    parser.add_argument(
        "--load",
        type=Path,
        help="Replay an inputs.npz file or a previously saved directory.",
    )
    args = parser.parse_args()
    row_ptr, col_idx, values, rhs, out = _load_inputs(args.load) if args.load else _demo_inputs()
    _validate_csr(row_ptr, col_idx, values, rhs, out)
    ordered = _reference_ordered_f32(row_ptr, col_idx, values, rhs)
    dense = _reference_dense_f64(row_ptr, col_idx, values, rhs)
    budget = _error_budget(row_ptr, col_idx, values, rhs)
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save / "inputs.npz",
            row_ptr=row_ptr,
            col_idx=col_idx,
            values=values,
            rhs=rhs,
            out_initial=out,
        )

    @qd.kernel
    def multiply(
        rp: qd.types.NDArray,
        ci: qd.types.NDArray,
        v: qd.types.NDArray,
        x: qd.types.NDArray,
        y: qd.types.NDArray,
    ):
        qd.algorithms.csr_spmm(rp, ci, v, x, y)

    log = io.StringIO()
    result = None
    metadata = {"status": "failed", "requested_arch": args.arch}
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            qd.init(arch=getattr(qd, args.arch), enable_fallback=False)
            assert qd.cfg is not None
            if qd.cfg.arch != getattr(qd, args.arch):
                raise RuntimeError("Actual architecture differs from requested architecture")
            if args.save:
                metadata.update(_reproducer_metadata(args.arch))
            buffers = [_upload(array) for array in (row_ptr, col_idx, values, rhs, out)]
            multiply(*buffers)
            result = np.asarray(buffers[-1].to_numpy(), dtype=np.float32)
            np.testing.assert_allclose(result, ordered, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(result, dense, rtol=2e-5, atol=2e-6)
            metadata["status"] = "passed"
            print(result)
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.save:
            metadata["input_sha256"] = _file_hash(args.save / "inputs.npz")
            metadata["shapes"] = {
                name: list(array.shape)
                for name, array in zip(
                    ("row_ptr", "col_idx", "values", "rhs", "out_initial"),
                    (row_ptr, col_idx, values, rhs, out),
                )
            }
            np.savez(
                args.save / "outputs.npz",
                out=result if result is not None else np.empty(0, dtype=np.float32),
                ordered_f32=ordered,
                dense_f64=dense,
                error_budget=budget,
            )
            (args.save / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (args.save / "execution.log").write_text(log.getvalue() + metadata.get("error", "") + "\n")
        print(log.getvalue(), end="")


if __name__ == "__main__":
    main()
