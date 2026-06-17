"""Helpers extracted from ``kernel.py`` for the ``qd.checkpoint(...)`` pause / resume model.

``Kernel.__call__`` / ``Kernel.launch_kernel`` delegate the resume-cookie validation, the user-label-to-internal-
cp_id translation, the per-launch ``yield_on=`` arg-id table construction, and the ``GraphStatus`` build to the free
functions below so the central ``Kernel`` class doesn't accrete checkpoint-feature-specific blocks. See
``qd.checkpoint`` / ``kernel.resume`` / ``docs/source/user_guide/graph.md`` for the user-facing surface.
"""

from __future__ import annotations

from typing import Any

from quadrants.lang import impl
from quadrants.lang.graph_status import GraphStatus


def validate_resume_cookie(kernel: Any, resume_from_checkpoint: int | None) -> None:
    """Raise if ``_qd_from_checkpoint`` was passed to a kernel without any ``qd.checkpoint(yield_on=...)`` block.

    Called from the preamble of ``Kernel.__call__`` so the user gets a clear error before any compile / launch work
    happens, rather than a confusing "no GraphStatus surface" failure later.
    """
    if resume_from_checkpoint is not None and not kernel.checkpoint_yield_on_args:
        raise RuntimeError(
            "`from_checkpoint=` is only valid for kernels that contain at least one "
            "qd.checkpoint(yield_on=...) block; this kernel has none."
        )


def translate_user_label_to_internal_cp_id(kernel: Any, user_label: int) -> int:
    """Translate a user-supplied ``from_checkpoint=`` label (int or IntEnum) to the runtime's internal dense cp_id.

    The runtime indexes checkpoints by source-declaration order (0, 1, 2, ...). The user-facing label is whatever int /
    IntEnum they passed as the first positional arg of ``qd.checkpoint(cp_id, yield_on=...)``; the
    ``checkpoint_user_labels_by_cp_id`` table maps internal cp_id -> user label. Compared with ``==`` so an IntEnum
    value matches its underlying int. Implicit (auto-wrapped) checkpoints have ``None`` in the table and are never
    resume targets. Raises ``RuntimeError`` listing the available labels when the user passed an unknown one.
    """
    for internal_cp_id, label in enumerate(kernel.checkpoint_user_labels_by_cp_id):
        if label is not None and label == user_label:
            return internal_cp_id
    available = [lbl for lbl in kernel.checkpoint_user_labels_by_cp_id if lbl is not None]
    raise RuntimeError(
        f"from_checkpoint={user_label!r} does not match any qd.checkpoint(cp_id=...) in "
        f"kernel {kernel.func.__name__!r}. Available cp_id labels (source-declaration order): {available}."
    )


def init_yield_on_arg_id_table(kernel: Any) -> None:
    """Allocate / reset the per-launch ``cp_id -> C++ arg-id`` table at the top of ``launch_kernel``'s arg iteration.

    Each entry defaults to ``-1`` ("no yield_on"); the per-arg loop below fills in the C++ arg id when it visits the
    named parameter. Sized to the kernel's checkpoint count once per launch so any changes to the checkpoint set (only
    possible via re-AST-walk) reset the table cleanly. No-op for kernels with no ``yield_on=`` checkpoints.
    """
    if kernel.checkpoint_yield_on_args:
        kernel._checkpoint_yield_on_cpp_arg_ids = [-1] * len(kernel.checkpoint_yield_on_args)


def maybe_record_yield_on_arg(kernel: Any, arg_name: str, cpp_arg_id: int) -> None:
    """Fill the ``cp_id -> C++ arg-id`` slot when the arg iterator visits a named ``yield_on=`` kernel parameter.

    Walked once per kernel arg in ``launch_kernel``; cheap O(checkpoints) match. A single parameter can be the
    ``yield_on=`` for multiple checkpoints (the inner loop fills every matching slot).
    """
    if not kernel.checkpoint_yield_on_args:
        return
    for cp_idx, yield_name in enumerate(kernel.checkpoint_yield_on_args):
        if yield_name is not None and arg_name == yield_name:
            kernel._checkpoint_yield_on_cpp_arg_ids[cp_idx] = cpp_arg_id


def forward_yield_on_table_to_ctx(kernel: Any, launch_ctx: Any) -> None:
    """Copy the resolved ``cp_id -> C++ arg-id`` table onto the launch context so the runtime can find each
    ``yield_on=`` ndarray's device address at launch.
    """
    if kernel.checkpoint_yield_on_args and hasattr(kernel, "_checkpoint_yield_on_cpp_arg_ids"):
        launch_ctx.checkpoint_yield_on_arg_ids = tuple(kernel._checkpoint_yield_on_cpp_arg_ids)


def maybe_build_graph_status(kernel: Any, default_ret: Any) -> Any:
    """Translate the runtime's internal yielding cp_id back to the user-supplied label and return a ``GraphStatus``.

    Returns ``default_ret`` unchanged for kernels without any yielding checkpoint -- there's no ``yield_on=`` parameter
    to surface a status from, so the value would always be ``yielded=False, checkpoint=None`` (no information).
    Implicit (auto-wrapped) checkpoints have ``None`` in ``checkpoint_user_labels_by_cp_id`` but they never have
    ``yield_on=``, so the runtime can't surface them as the yielding cp -- the lookup is always to an explicit
    checkpoint.
    """
    if not (kernel.checkpoint_yield_on_args and any(n is not None for n in kernel.checkpoint_yield_on_args)):
        return default_ret
    cp = impl.get_runtime().prog.get_graph_last_yield_cp_id_on_last_call()
    if cp >= 0:
        user_label = (
            kernel.checkpoint_user_labels_by_cp_id[cp] if cp < len(kernel.checkpoint_user_labels_by_cp_id) else None
        )
        return GraphStatus(yielded=True, checkpoint=user_label if user_label is not None else cp)
    return GraphStatus(yielded=False, checkpoint=None)
