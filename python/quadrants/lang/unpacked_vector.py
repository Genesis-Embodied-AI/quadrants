# type: ignore
"""``qd.unpacked_array`` -- indexed groups of independently-allocated scalar fields on ``@qd.dataclass``.

An ``unpacked_array(N, dtype)`` annotation expands at struct-definition time into N individually-named synthetic scalar
members (``_{group}0`` .. ``_{group}{N-1}``). The AST transformer rewrites ``obj.{group}[i]`` into a direct reference to
``obj._{group}{i}`` for python-int / ``qd.static``-resolved indices, so generated LLVM IR / PTX is byte-identical to a
hand-rolled named-field struct.

Compare to ``qd.types.vector(N, dtype)`` which is the *packed* layout: one ``alloca`` covers all N slots. Packed storage
is fine until register pressure rises -- once LLVM SROA fails to decompose the packed ``alloca`` (e.g. two concurrent
tiles in a Cholesky + TRSM kernel), the whole group spills to local memory as a unit. ``unpacked_array`` lays each slot
out in its own ``alloca`` up front, so SROA + ``mem2reg`` can promote slots independently and the optimiser can spill
only the ones it has to. The ergonomic indexed-access syntax is preserved at the source level.

Public:
- ``UnpackedVector``        - type wrapper used as the annotation value
- ``unpacked_array``        - factory: ``r: qd.unpacked_array(N, dtype)``

Internal (used by ``StructType`` and the AST transformer):
- ``_expand_unpacked_vector_naming(group, i)`` - synthetic-field naming convention
- ``_UnpackedVectorRef``                       - transient proxy yielded by attribute access

This module has no dependency on ``struct.py``; ``struct.py`` imports from here.
"""

import numpy as np

from quadrants.lang.exception import QuadrantsSyntaxError


class UnpackedVector:
    """Type wrapper for a group of N scalar fields exposed via indexed syntax on a ``@qd.dataclass``.

    See :func:`unpacked_array` for the user-facing constructor and the motivation writeup. Holding only ``count`` and
    ``dtype``, this object is consumed at struct-definition time by ``StructType.__init__`` to lay out the N synthetic
    scalar fields.
    """

    def __init__(self, count, dtype):
        if not isinstance(count, int) or count <= 0:
            raise QuadrantsSyntaxError(f"unpacked_array count must be a positive int, got {count!r}")
        self.count = count
        self.dtype = dtype

    def __repr__(self):
        return f"unpacked_array(count={self.count}, dtype={self.dtype})"


def unpacked_array(count, dtype):
    """Declare a group of ``count`` independently-allocated fields of ``dtype`` on a ``@qd.dataclass``.

    The annotation expands at struct-definition time into ``count`` individually-named scalar members (``_{group}0`` ..
    ``_{group}{count-1}``). Each member gets its own LLVM ``alloca``, which lets SROA + ``mem2reg`` promote each slot
    into its own SSA value independently. The contrast is with ``qd.types.vector(N, dtype)``, which lays all N slots
    out in a single packed ``alloca``: when register pressure makes SROA fail to decompose the packed ``alloca``, the
    whole group spills to local memory as a unit. With ``unpacked_array`` the storage is already unpacked, so the
    optimiser can keep individual slots in registers and only spill the ones it has to.

    Example::

        @qd.dataclass
        class Tile:
            r: qd.unpacked_array(32, qd.f32)   # 32 scalar fields exposed as t.r[0..31]

        t = Tile()
        t.r[5] = 1.0       # lowers to direct write of synthetic field _r5
        v = t.r[5]         # same as v = t._r5
        for k in qd.static(range(32)):
            t.r[k] = 0.0   # each iter is one AST node, not a 32-way cascade

    For python-int / ``qd.static``-resolved indices, ``t.r[k]`` is rewritten by the AST transformer to the named-field
    access ``t._r{k}``, producing identical LLVM IR / PTX to a struct declared with N individually-named scalar fields.

    Runtime-int indexing is currently unsupported; use an explicit cascade helper for that case.
    """
    return UnpackedVector(count, dtype)


def _expand_unpacked_vector_naming(group_name, index):
    """Naming convention for the synthetic scalar fields of an ``unpacked_array`` group.

    Public-ish so the AST transformer can mirror this without a circular import on ``struct``.
    """
    return f"_{group_name}{index}"


class _UnpackedVectorRef:
    """Transient proxy returned by the AST transformer for ``obj.{group}`` where ``group`` is an unpacked-vector group
    declared on the struct type.

    Only valid as the value of a Subscript node: ``obj.{group}[i]``. Resolved by ``ASTTransformer.build_Subscript``
    to a direct reference to the synthetic scalar field ``_{group}{i}`` when ``i`` is a python-int /
    ``qd.static``-resolved integer.

    Used as a not-an-Expr marker; any attempt to use it as a value raises.
    """

    _qd_is_unpacked_vector_ref = True

    def __init__(self, struct, group_name: str, count: int, dtype, naming_fn):
        self._qd_struct = struct
        self._qd_group_name = group_name
        self._qd_count = count
        self._qd_dtype = dtype
        self._qd_naming_fn = naming_fn

    def _qd_field_for(self, index: int):
        if not isinstance(index, (int, np.integer)):
            raise QuadrantsSyntaxError(
                f"unpacked_array {self._qd_group_name}[i] requires a python-int index "
                f"(possibly via qd.static); got runtime index of type {type(index).__name__}"
            )
        i = int(index)
        if i < 0 or i >= self._qd_count:
            raise QuadrantsSyntaxError(
                f"unpacked_array index out of bounds: {self._qd_group_name}[{i}] " f"(count={self._qd_count})"
            )
        field_name = self._qd_naming_fn(self._qd_group_name, i)
        return getattr(self._qd_struct, field_name)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"<unpacked_array_ref group={self._qd_group_name!r} count={self._qd_count} " f"dtype={self._qd_dtype}>"


__all__ = ["UnpackedVector", "unpacked_array"]
