from quadrants.types.ndarray_type import NdarrayType


class BufferViewType:
    """Type annotation for BufferView kernel parameters.

    A BufferView is decomposed into three kernel arguments (ndarray, offset, size) and reassembled at compile time.

    Args:
        dtype: Element data type (e.g. qd.f32).
        boundary: Boundary mode for the underlying ndarray ("unsafe", "clamp").

    Prefer using the ``BufferView[dtype]`` square-bracket syntax::

        from quadrants import BufferView

        @qd.kernel
        def k(v: BufferView[qd.f32]):
            for i in range(v.size):
                v[i] *= 2.0
    """

    def __init__(self, dtype=None, boundary="unsafe"):
        self.dtype = dtype
        self.ndarray_type = NdarrayType(dtype=dtype, ndim=1, boundary=boundary)

    def __repr__(self):
        return f"BufferViewType(dtype={self.dtype}, boundary={self.ndarray_type.boundary})"


__all__ = ["BufferViewType"]
