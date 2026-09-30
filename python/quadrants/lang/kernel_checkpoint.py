"""Helpers extracted from ``kernel.py`` for the ``qd.checkpoint(...)`` pause / resume model.

``Kernel.__call__`` / ``Kernel.launch_kernel`` delegate the resume-cookie validation, the user-label-to-internal- cp_id
translation, the per-launch ``yield_on=`` arg-id table construction, and the ``GraphStatus`` build to the free functions
below so the central ``Kernel`` class doesn't accrete checkpoint-feature-specific blocks. See ``qd.checkpoint`` /
``kernel.resume`` / ``docs/source/user_guide/graph.md`` for the user-facing surface.
"""

from __future__ import annotations

from typing import Any, Sequence

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.exception import QuadrantsRuntimeError
from quadrants.lang.graph_status import GraphStatus


def validate_resume_cookie(
    resume_from_checkpoint: int | None,
    checkpoint_yield_on_args: Sequence[str | None],
) -> None:
    """Raise if ``_qd_from_checkpoint`` was passed to a kernel without any ``qd.checkpoint(yield_on=...)`` block.

    Called after specialization selection so the check uses that specialization's checkpoint table rather than
    whichever table the shared ``Kernel`` object populated most recently.
    """
    if resume_from_checkpoint is not None and not checkpoint_yield_on_args:
        raise RuntimeError(
            "`from_checkpoint=` is only valid for kernels that contain at least one "
            "qd.checkpoint(yield_on=...) block; this kernel has none."
        )


def validate_graph_do_while_in_checkpoint_backend(graph_do_while_levels: Sequence[Any]) -> None:
    """Reject checkpoint-owned graph loops when the backend cannot skip the WHILE node as one unit.

    Per-task checkpoint gates are insufficient for this shape: a skipped child loop with a stale nonzero condition can
    re-enter and execute on a later host-driven iteration. CUDA SM 9.0+ avoids that by placing the whole WHILE node
    inside a native checkpoint IF. Keep the public surface fail-fast everywhere else until those fallback drivers gain
    equivalent structural gating.
    """
    if not any(level.checkpoint_id >= 0 for level in graph_do_while_levels):
        return

    arch = impl.current_cfg().arch
    if arch == _qd_core.Arch.cuda:
        compute_capability = impl.get_cuda_compute_capability()
        if compute_capability >= 90:
            return
        backend = f"CUDA SM {compute_capability // 10}.{compute_capability % 10}"
    else:
        backend = arch.name

    raise QuadrantsRuntimeError(
        "A qd.checkpoint() block containing qd.graph.do_while() currently requires CUDA SM 9.0+ native "
        f"conditional graph nodes; the active backend ({backend}) cannot skip the child WHILE as one checkpoint unit. "
        "Move the loop outside the checkpoint or run this kernel on CUDA SM 9.0+."
    )


def translate_user_label_to_internal_cp_id(
    kernel: Any,
    user_label: int,
    checkpoint_user_labels_by_cp_id: Sequence[int | None],
) -> int:
    """Translate a user-supplied ``from_checkpoint=`` label (int or IntEnum) to the runtime's internal dense cp_id.

    The runtime indexes checkpoints by source-declaration order (0, 1, 2, ...). The user-facing label is whatever int /
    IntEnum they passed as the first positional arg of ``qd.checkpoint(cp_id, yield_on=...)``; the
    ``checkpoint_user_labels_by_cp_id`` table maps internal cp_id -> user label. Compared with ``==`` so an IntEnum
    value matches its underlying int. Implicit (auto-wrapped) checkpoints have ``None`` in the table and are never
    resume targets. Raises ``RuntimeError`` listing the available labels when the user passed an unknown one.
    """
    for internal_cp_id, label in enumerate(checkpoint_user_labels_by_cp_id):
        if label is not None and label == user_label:
            return internal_cp_id
    available = [lbl for lbl in checkpoint_user_labels_by_cp_id if lbl is not None]
    raise RuntimeError(
        f"from_checkpoint={user_label!r} does not match any qd.checkpoint(cp_id=...) in "
        f"kernel {kernel.func.__name__!r}. Available cp_id labels (source-declaration order): {available}."
    )


def forward_yield_on_table_to_ctx(checkpoint_yield_on_cpp_arg_ids: Sequence[int], launch_ctx: Any) -> None:
    """Copy the ``cp_id -> C++ arg-id`` table onto the launch context so the runtime can find each ``yield_on=``
    ndarray's device address at launch.

    The table is populated at AST-build time by
    ``CheckpointTransformer.build_checkpoint_with`` via ``ASTTransformer._resolve_ndarray_kernel_arg_id``, which
    uniformly handles bare kernel parameters (``yield_on=flag``) and ``@qd.data_oriented`` member ndarrays
    (``yield_on=self.flag``). No per-launch arg iteration / name match is required.
    """
    if checkpoint_yield_on_cpp_arg_ids:
        launch_ctx.checkpoint_yield_on_arg_ids = tuple(checkpoint_yield_on_cpp_arg_ids)


def maybe_build_graph_status(
    checkpoint_yield_on_args: Sequence[str | None],
    checkpoint_user_labels_by_cp_id: Sequence[int | None],
    default_ret: Any,
) -> Any:
    """Translate the runtime's internal yielding cp_id back to the user-supplied label and return a ``GraphStatus``.

    Returns ``default_ret`` unchanged for kernels without any yielding checkpoint -- there's no ``yield_on=`` parameter
    to surface a status from, so the value would always be ``yielded=False, checkpoint=None`` (no information). Implicit
    (auto-wrapped) checkpoints have ``None`` in ``checkpoint_user_labels_by_cp_id`` but they never have ``yield_on=``,
    so the runtime can't surface them as the yielding cp -- the lookup is always to an explicit checkpoint.
    """
    if not (checkpoint_yield_on_args and any(n is not None for n in checkpoint_yield_on_args)):
        return default_ret
    cp = impl.get_runtime().prog.get_graph_last_yield_cp_id_on_last_call()
    if cp >= 0:
        user_label = checkpoint_user_labels_by_cp_id[cp] if cp < len(checkpoint_user_labels_by_cp_id) else None
        return GraphStatus(yielded=True, checkpoint=user_label if user_label is not None else cp)
    return GraphStatus(yielded=False, checkpoint=None)
