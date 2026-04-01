# type: ignore

import re

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import Expr, make_expr_group
from quadrants.lang.util import quadrants_scope

_LOC_RE = re.compile(r'File "(.+?)", line (\d+), in (\w+)')

def _build_callstack(max_bytes):
    """Walk src_info_stack and extract deduplicated (file, line, func) frames.

    Returns (kernel_name, callstack_str).
    The callstack is kept within *max_bytes* by trimming from the middle.
    """
    stack = impl.get_runtime().src_info_stack
    frames = []
    prev_func = None
    for info in stack:
        if not info:
            continue
        m = _LOC_RE.search(info)
        if not m:
            continue
        filepath, lineno, funcname = m.group(1), m.group(2), m.group(3)
        if funcname != prev_func:
            frames.append((filepath, lineno, funcname))
            prev_func = funcname

    if not frames:
        return "<unknown>", "<unknown>"

    kernel_name = frames[0][2]

    def _format_frame(fp, ln, fn, depth):
        return "  " * depth + f"{fn} ({fp}:{ln})"

    def _format_chain(frame_list, start_depth=0):
        return "\n".join(
            _format_frame(fp, ln, fn, start_depth + i)
            for i, (fp, ln, fn) in enumerate(frame_list)
        )

    callstack = _format_chain(frames)

    budget = max_bytes // 2
    if len(callstack) > budget and len(frames) > 2:
        kept_head = 1
        kept_tail = 1
        while kept_head + kept_tail + 1 < len(frames):
            head_part = _format_chain(frames[:kept_head + 1])
            tail_part = _format_chain(frames[-(kept_tail + 1):], start_depth=len(frames) - kept_tail - 1)
            trial = head_part + "\n" + "  " * (kept_head + 1) + "..." + "\n" + tail_part
            if len(trial) <= budget:
                kept_head += 1
            else:
                break
        head_part = _format_chain(frames[:kept_head])
        tail_part = _format_chain(frames[-kept_tail:], start_depth=len(frames) - kept_tail)
        omitted = len(frames) - kept_head - kept_tail
        callstack = head_part + "\n" + "  " * kept_head + f"...({omitted} more)..." + "\n" + tail_part

    if len(callstack) > budget:
        callstack = callstack[:max(budget - 3, 0)] + "..."

    return kernel_name, callstack


class BufferView:
    """A view into a sub-range [offset, offset+count) of an ndarray kernel argument.

    Intercepts subscript operations at AST-translation time to rewrite
    ``view[i]`` into ``arr[offset + i]`` with optional bounds checking,
    without any IR-level changes.

    Can be used in two ways:

    1. Constructed manually inside a kernel from separate parameters::

        @qd.kernel
        def k(buf: qd.types.ndarray(qd.f32, ndim=1),
              offset: qd.i32, count: qd.i32):
            view = qd.BufferView(buf, offset, count)
            for i in range(count):
                view[i] *= 2.0

    2. Passed directly as a kernel argument (auto-decomposed)::

        buf = qd.ndarray(qd.f32, shape=(N,))
        view = qd.BufferView(buf, offset=16, count=32)

        @qd.kernel
        def k(v: qd.types.buffer_view(qd.f32)):
            for i in range(v.count):
                v[i] *= 2.0

        k(view)
    """

    _is_quadrants_class = True

    def __init__(self, arr, offset, count):
        self.arr = arr
        self.offset = offset
        self.count = count

    @quadrants_scope
    def subscript(self, *indices):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        src_info = impl.get_runtime().get_current_src_info()
        dbg_info = _qd_core.DebugInfo(src_info)

        cfg = impl.get_runtime().prog.config()
        if cfg.debug:
            i_expr = Expr(indices[0])
            offset_expr = Expr(self.offset)
            count_expr = Expr(self.count)

            tid_expr = Expr(ast_builder.insert_thread_idx_expr())

            kernel_name, callstack = _build_callstack(2048)

            msg = (
                f"BufferView Out Of Range: kernel[{kernel_name}]"
                " tid=%d, got index %d (offset=%d, count=%d).\n"
                f"Callstack:\n"
                f"{callstack}"
            )

            impl.qd_assert(
                (i_expr >= Expr(0)).ptr,
                msg,
                [tid_expr.ptr, i_expr.ptr, offset_expr.ptr, count_expr.ptr],
                dbg_info,
            )
            impl.qd_assert(
                (i_expr < count_expr).ptr,
                msg,
                [tid_expr.ptr, i_expr.ptr, offset_expr.ptr, count_expr.ptr],
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
