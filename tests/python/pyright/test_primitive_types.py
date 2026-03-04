"""Pyright test: primitive dtype classes are usable as type annotations."""

from __future__ import annotations

import quadrants as qd


def accept_f32(x: qd.f32) -> None: ...


def accept_i32(x: qd.i32) -> None: ...


def accept_any_dtype(x: qd.f32 | qd.i32 | qd.u8) -> None: ...


def use_ndarray_subscript(
    a: qd.types.NDArray[qd.f32, 2],
    b: qd.types.NDArray[qd.i32, 1],
    c: qd.types.NDArray[qd.u8],
    d: qd.types.NDArray,
) -> None: ...


arr: qd.types.NDArray[qd.f64, 3]
