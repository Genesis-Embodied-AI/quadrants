from quadrants.types.ndarray_type import NdarrayType


class BufferViewType:
    """Type annotation for BufferView kernel parameters.

    A BufferView is automatically decomposed into three kernel arguments
    (ndarray, offset, count) at compile time and reassembled into a
    kernel-scope BufferView object.

    Args:
        dtype: Element data type (e.g. qd.f32).
        boundary: Boundary mode for the underlying ndarray ("unsafe", "clamp").

    Example::

        @qd.kernel
        def k(v: qd.types.buffer_view(qd.f32)):
            for i in range(v.count):
                v[i] *= 2.0
    """

    def __init__(self, dtype=None, boundary="unsafe"):
        self.dtype = dtype
        self.boundary = boundary
        self.ndarray_type = NdarrayType(dtype=dtype, ndim=1, boundary=boundary)

    @classmethod
    def __class_getitem__(cls, args, **kwargs):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args, **kwargs)

    def __repr__(self):
        return f"BufferViewType(dtype={self.dtype}, boundary={self.boundary})"


buffer_view = BufferViewType

__all__ = ["buffer_view", "BufferViewType"]
