"""Pyright test: primitive dtype classes and NDArray subscript syntax.

This file must produce zero pyright errors. It validates that:
- Primitive dtypes (f32, i32, etc.) work as type annotations
- NDArray[dtype, ndim] subscript syntax is accepted
- from __future__ import annotations (stringified) works
- NDArray works via qd.types and top-level qd.NDArray
- Return types and Optional wrappers work
"""

from __future__ import annotations

from typing import Optional

import quadrants as qd


# Primitive types as annotations
def accept_f32(x: qd.f32) -> None: ...


def accept_i32(x: qd.i32) -> None: ...


def accept_any_dtype(x: qd.f32 | qd.i32 | qd.u8) -> None: ...


# NDArray subscript: dtype + ndim
def kernel_2d(a: qd.types.NDArray[qd.f32, 2]) -> None: ...


# NDArray subscript: dtype only
def kernel_dtype(a: qd.types.NDArray[qd.i32]) -> None: ...


# NDArray bare (no subscript)
def kernel_bare(a: qd.types.NDArray) -> None: ...


# Multiple NDArray args with different types
def multi_args(
    a: qd.types.NDArray[qd.f32, 2],
    b: qd.types.NDArray[qd.i32, 1],
    c: qd.types.NDArray,
) -> None: ...


# Top-level NDArray alias
def top_level(a: qd.NDArray[qd.f32, 2]) -> None: ...


# Return types
def make_arr() -> qd.types.NDArray[qd.f32, 2]: ...


# Optional wrapping
def maybe_arr(x: Optional[qd.types.NDArray[qd.f32, 2]]) -> None: ...


# Variable annotations
field1: qd.types.NDArray[qd.f32, 2]
field2: qd.types.NDArray


# In class body
class MyModel:
    buf: qd.types.NDArray[qd.f32, 3]

    def forward(self, x: qd.types.NDArray[qd.f32, 2]) -> qd.types.NDArray[qd.f32, 2]: ...


# Access via qd.types submodule
def via_types(x: qd.types.NDArray[qd.types.f32, 2]) -> None: ...
