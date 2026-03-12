from __future__ import annotations

from typing import Any

from quadrants import types
from quadrants.lang import impl
from quadrants.types import primitive_types
from quadrants.types.enums import Layout

_DTYPE_NAMES = ["f16", "f32", "f64", "i8", "i16", "i32", "i64", "u1", "u8", "u16", "u32", "u64"]
_NAME_TO_DTYPE: dict[str, Any] = {name: getattr(primitive_types, name) for name in _DTYPE_NAMES}
_DTYPE_TO_NAME: dict[Any, str] = {dt: name for name, dt in _NAME_TO_DTYPE.items()}

_LAYOUT_TO_NAME: dict[Layout, str] = {Layout.AOS: "AOS", Layout.SOA: "SOA"}
_NAME_TO_LAYOUT: dict[str, Layout] = {name: layout for name, layout in _LAYOUT_TO_NAME.items()}

_PICKLE_VERSION = 1


def serialize(ndarray: Any) -> dict[str, Any]:
    """Serialize an ndarray to a plain dict for pickling.

    Note: This creates a full NumPy copy of the underlying data via
    ``ndarray.to_numpy()``, temporarily doubling memory usage for
    large arrays. This is necessary because the device memory backing
    the ndarray could change before the pickled bytes are written.
    """
    dtype_name = _DTYPE_TO_NAME.get(ndarray.dtype)
    if dtype_name is None:
        raise TypeError(f"Cannot pickle ndarray with dtype {ndarray.dtype!r}")
    layout_name = _LAYOUT_TO_NAME.get(ndarray.layout)
    if layout_name is None:
        raise TypeError(f"Cannot pickle ndarray with layout {ndarray.layout!r}")
    return {
        "version": _PICKLE_VERSION,
        "shape": ndarray.shape,
        "element_type": dtype_name,
        "element_shape": ndarray.element_shape,
        "layout": layout_name,
        "data": ndarray.to_numpy(),
    }


def unpickle(pkl: dict[str, Any]) -> Any:
    """Reconstruct an ndarray from a pickle dict produced by :func:`serialize`."""
    if impl.get_runtime()._prog is None:  # type: ignore[union-attr]
        raise RuntimeError("qd.init() must be called before unpickling ndarrays")
    version = pkl.get("version")
    if version != _PICKLE_VERSION:
        raise ValueError(f"Unsupported ndarray pickle version {version!r} " f"(expected {_PICKLE_VERSION})")
    dtype_name = pkl["element_type"]
    if dtype_name not in _NAME_TO_DTYPE:
        raise ValueError(f"Unknown dtype '{dtype_name}' during unpickle")
    dtype = _NAME_TO_DTYPE[dtype_name]
    shape: tuple[int, ...] = pkl["shape"]
    element_shape: tuple[int, ...] = pkl["element_shape"]
    if len(element_shape) == 0:
        res = impl.ndarray(dtype=dtype, shape=shape)
    elif len(element_shape) == 1:
        element_type = types.vector(element_shape[0], dtype)
        res = impl.ndarray(element_type, shape=shape)
    elif len(element_shape) == 2:
        element_type = types.matrix(element_shape[0], element_shape[1], dtype)
        res = impl.ndarray(element_type, shape=shape)
    else:
        raise NotImplementedError(
            f"Unpickling element_shape of length {len(element_shape)} is not supported. "
            f"Supported shapes: () for scalars, (n,) for vectors, (n, m) for matrices."
        )
    layout_name = pkl.get("layout")
    if layout_name is not None:
        layout = _NAME_TO_LAYOUT.get(layout_name)
        if layout is None:
            raise ValueError(f"Unknown layout '{layout_name}' during unpickle")
        res.layout = layout
    res.from_numpy(pkl["data"])  # pylint: disable=no-member
    return res
