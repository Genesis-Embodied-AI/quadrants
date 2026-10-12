"""Specialization-specific metadata used when launching graph kernels."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class GraphDoWhileLevel:
    """One ``qd.graph_do_while`` loop, indexed outer-before-inner by the AST transformer."""

    cond_arg_name: str
    parent_id: int
    cond_cpp_arg_id: int = -1
    checkpoint_id: int = -1


@dataclasses.dataclass(frozen=True)
class KernelLaunchMetadata:
    """AST metadata consumed by one compiled specialization's launch path."""

    graph_do_while_levels: tuple[GraphDoWhileLevel, ...]
    checkpoint_yield_on_args: tuple[str | None, ...]
    checkpoint_yield_on_cpp_arg_ids: tuple[int, ...]
    checkpoint_user_labels_by_cp_id: tuple[int | None, ...]
