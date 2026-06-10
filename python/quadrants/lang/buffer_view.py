# type: ignore

import re

from quadrants import _logging
from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import Expr, make_expr_group
from quadrants.lang.util import quadrants_scope
from quadrants.types.buffer_view_type import BufferViewType

_LOC_RE = re.compile(r'File "(.+?)", line (\d+), in (\w+)')

_CALLSTACK_WARNING_BYTE = 1024


def _build_callstack():
    """Walk src_info_stack and extract deduplicated (file, line, func) frames.

    Returns (kernel_name, callstack_str).
    Warns if the callstack exceeds _CALLSTACK_WARNING_BYTE (large strings increase compiled binary size).
    """
    stack = impl.get_runtime().src_info_stack
    frames = []
    prev_key = None
    for info in stack:
        if not info:
            continue
        m = _LOC_RE.search(info)
        if not m:
            continue
        filepath, lineno, funcname = m.group(1), m.group(2), m.group(3)
        key = (filepath, funcname)
        if key != prev_key:
            frames.append((filepath, lineno, funcname))
            prev_key = key

    if not frames:
        return "<unknown>", "<unknown>"

    kernel_name = frames[0][2]

    def _format_frame(fp, ln, fn, depth):
        return "  " * depth + f"{fn} ({fp}:{ln})"

    def _format_chain(frame_list, start_depth=0):
        return "\n".join(_format_frame(fp, ln, fn, start_depth + i) for i, (fp, ln, fn) in enumerate(frame_list))

    callstack = _format_chain(frames)

    if len(callstack) > _CALLSTACK_WARNING_BYTE:
        _logging.warn(
            f"BufferView debug callstack is {len(callstack)} bytes (limit {_CALLSTACK_WARNING_BYTE}). "
            f"Large callstack strings increase compiled binary size."
        )

    return kernel_name, callstack


class BufferView:
    """A view into a sub-range [offset, offset+size) of an ndarray.

    Create via slice syntax::

        view = data[:16]           # offset=0, size=16
        view = data[8:24]          # offset=8, size=16

    Or construct directly::

        view = qd.BufferView(data, offset=16, size=32)

    Subviews can be created from an existing view::

        sub = view.subview(offset=4, size=8)

    Annotate kernel/func parameters with ``BufferView[dtype]`` or plain ``BufferView``::

        @qd.kernel
        def k(v: BufferView[qd.f32]):
            for i in range(v.size):
                v[i] *= 2.0
    """

    _is_quadrants_class = True

    @classmethod
    def __class_getitem__(cls, dtype):
        """Enable ``BufferView[qd.f32]`` as a kernel/func type annotation."""
        if isinstance(dtype, tuple):
            return BufferViewType(*dtype)
        return BufferViewType(dtype)

    def __init__(self, arr, offset, size):
        if impl.inside_kernel():
            # Kernel-compilation path: insert debug bounds assertion against the backing ndarray.
            cfg = impl.get_runtime().prog.config()
            if cfg.debug:
                ast_builder = impl.get_runtime().compiling_callable.ast_builder()
                src_info = impl.get_runtime().get_current_src_info()
                dbg_info = _qd_core.DebugInfo(src_info)
                arr_len = Expr(_qd_core.get_external_tensor_shape_along_axis(arr.ptr, 0, dbg_info))
                offset_expr, size_expr = Expr(offset), Expr(size)
                msg = "BufferView construction out of range: offset=%d, size=%d, ndarray length=%d."
                args = [offset_expr.ptr, size_expr.ptr, arr_len.ptr]
                impl.qd_assert((offset_expr >= Expr(0)).ptr, msg, args, dbg_info)
                impl.qd_assert((size_expr >= Expr(0)).ptr, msg, args, dbg_info)
                impl.qd_assert(((offset_expr + size_expr) <= arr_len).ptr, msg, args, dbg_info)
        else:
            # Host-side path: coerce to int and validate bounds.
            offset, size = int(offset), int(size)
            if isinstance(arr, BufferView):
                raise TypeError("Cannot nest BufferView(view, ...); use view.subview(offset, size) instead")
            if getattr(arr, "grad", None) is not None:
                raise TypeError("BufferView does not support ndarrays with gradients (needs_grad=True)")
            arr_shape = getattr(arr, "shape", None)
            if arr_shape is not None:
                if len(arr_shape) != 1:
                    raise TypeError(f"BufferView requires a 1D ndarray, got shape {arr_shape}")
                if offset < 0 or size < 0 or offset + size > arr_shape[0]:
                    raise ValueError(
                        f"BufferView out of range: offset={offset}, size={size}, ndarray length={arr_shape[0]}"
                    )
        self.arr = arr
        self.offset = offset
        self.size = size

    @property
    def shape(self):
        """Returns the shape of this view as a tuple, e.g. ``(16,)``."""
        return (self.size,)

    @property
    def dtype(self):
        """Element dtype of the backing ndarray (a view shares its array's dtype).

        Lets a host-constructed view be passed wherever an ndarray's ``.dtype`` is read - e.g. the
        ``qd.algorithms.*`` host entries that derive their compile-time element type from the buffer argument.
        """
        return self.arr.dtype

    def subview(self, offset, size):
        """Create a sub-range view within this view, with bounds checking.

        Offsets are relative to this view's start, not the ndarray's::

            a = data[8:24]          # offset=8, size=16 into data
            b = a.subview(4, 8)     # offset=12, size=8 into data
        """
        if impl.inside_kernel():
            return self._subview_expr(Expr(offset), Expr(size))
        offset, size = int(offset), int(size)
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError(f"subview out of range: offset={offset}, size={size}, parent size={self.size}")
        return BufferView(self.arr, self.offset + offset, size)

    @quadrants_scope
    def _subview_expr(self, offset, size):
        """Kernel-compilation path for subview: insert debug bounds assertions."""
        offset_expr = Expr(offset)
        size_expr = Expr(size)
        parent_size_expr = Expr(self.size)

        cfg = impl.get_runtime().prog.config()
        if cfg.debug:
            ast_builder = impl.get_runtime().compiling_callable.ast_builder()
            src_info = impl.get_runtime().get_current_src_info()
            dbg_info = _qd_core.DebugInfo(src_info)
            tid_expr = Expr(ast_builder.insert_thread_idx_expr())

            kernel_name, callstack = _build_callstack()
            msg = (
                f"BufferView subview Out Of Range: kernel[{kernel_name}]"
                " tid=%d, subview offset=%d, subview size=%d, parent size=%d.\n"
                f"Callstack:\n"
                f"{callstack}"
            )
            args = [tid_expr.ptr, offset_expr.ptr, size_expr.ptr, parent_size_expr.ptr]

            impl.qd_assert((offset_expr >= Expr(0)).ptr, msg, args, dbg_info)
            impl.qd_assert((size_expr >= Expr(0)).ptr, msg, args, dbg_info)
            impl.qd_assert(((offset_expr + size_expr) <= parent_size_expr).ptr, msg, args, dbg_info)

        new_offset = Expr(self.offset) + offset_expr
        return BufferView(self.arr, new_offset, size_expr)

    def __getitem__(self, key):
        """Slice a view to create a subview (host-side only): ``view[2:6]``.

        In kernels, slicing goes through impl.subscript() -> subview() instead.
        """
        assert (
            not impl.inside_kernel()
        ), "BufferView.__getitem__ is not reachable in kernel scope; subscripts are dispatched via impl.subscript()"
        if not isinstance(key, slice):
            raise TypeError(f"BufferView host-side indexing requires a slice, got {type(key).__name__}")
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError(f"BufferView slice requires step=1, got step={step}")
        return self.subview(start, max(stop - start, 0))

    @quadrants_scope
    def subscript(self, *indices):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        src_info = impl.get_runtime().get_current_src_info()
        dbg_info = _qd_core.DebugInfo(src_info)

        cfg = impl.get_runtime().prog.config()
        if cfg.debug:
            i_expr = Expr(indices[0])
            offset_expr = Expr(self.offset)
            size_expr = Expr(self.size)

            tid_expr = Expr(ast_builder.insert_thread_idx_expr())

            kernel_name, callstack = _build_callstack()

            msg = (
                f"BufferView Out Of Range: kernel[{kernel_name}]"
                " tid=%d, got index %d (offset=%d, size=%d).\n"
                f"Callstack:\n"
                f"{callstack}"
            )

            impl.qd_assert(
                (i_expr >= Expr(0)).ptr,
                msg,
                [tid_expr.ptr, i_expr.ptr, offset_expr.ptr, size_expr.ptr],
                dbg_info,
            )
            impl.qd_assert(
                (i_expr < size_expr).ptr,
                msg,
                [tid_expr.ptr, i_expr.ptr, offset_expr.ptr, size_expr.ptr],
                dbg_info,
            )

        new_first = Expr(indices[0]) + Expr(self.offset)
        new_indices = [new_first, *indices[1:]]

        indices_expr_group = make_expr_group(*new_indices)
        return Expr(ast_builder.expr_subscript(self.arr.ptr, indices_expr_group, dbg_info))

    def get_ndarray(self):
        """Returns the underlying ndarray (host-side only)."""
        return self.arr


__all__ = ["BufferView"]
