# type: ignore
from quadrants import types
from quadrants.lang import impl
from quadrants.types import primitive_types

_DTYPE_NAMES = ["f16", "f32", "f64", "i8", "i16", "i32", "i64", "u1", "u8", "u16", "u32", "u64"]
_NAME_TO_DTYPE = {name: getattr(primitive_types, name) for name in _DTYPE_NAMES}
_DTYPE_TO_NAME = {dt: name for name, dt in _NAME_TO_DTYPE.items()}


def serialize(ndarray):
    dtype_name = _DTYPE_TO_NAME.get(ndarray.dtype)
    if dtype_name is None:
        raise TypeError(f"Cannot pickle ndarray with dtype {ndarray.dtype!r}")
    return {
        "version": 1,
        "shape": ndarray.shape,
        "element_type": dtype_name,
        "element_shape": ndarray.element_shape,
        "data": ndarray.to_numpy(),
    }


_PICKLE_VERSION = 1


def unpickle(pkl):
    if impl.get_runtime()._prog is None:
        raise RuntimeError("qd.init() must be called before unpickling ndarrays")
    version = pkl.get("version")
    if version != _PICKLE_VERSION:
        raise ValueError(
            f"Unsupported ndarray pickle version {version!r} "
            f"(expected {_PICKLE_VERSION})"
        )
    dtype_name = pkl["element_type"]
    if dtype_name not in _NAME_TO_DTYPE:
        raise ValueError(f"Unknown dtype '{dtype_name}' during unpickle")
    dtype = _NAME_TO_DTYPE[dtype_name]
    shape = pkl["shape"]
    element_shape = pkl["element_shape"]
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
    res.from_numpy(pkl["data"])
    return res
