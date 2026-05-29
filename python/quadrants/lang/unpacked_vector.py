# type: ignore
"""``UnpackedVector[dtype, count]`` -- indexed groups of independently-allocated scalar fields on ``@qd.dataclass``.

An ``UnpackedVector[dtype, N]`` annotation expands at struct-definition time into N individually-named synthetic scalar
members (``_{group}0`` .. ``_{group}{N-1}``). The AST transformer rewrites ``obj.{group}[i]`` into a direct reference to
``obj._{group}{i}`` for python-int / ``qd.static``-resolved indices, so generated LLVM IR / PTX is byte-identical to a
hand-rolled named-field struct.

Compare to ``qd.types.vector(N, dtype)`` which is the *packed* layout: one ``alloca`` covers all N slots. Packed storage
is fine until register pressure rises -- once LLVM SROA fails to decompose the packed ``alloca`` (e.g. two concurrent
tiles in a Cholesky + TRSM kernel), the whole group spills to local memory as a unit. ``UnpackedVector`` lays each slot
out in its own ``alloca`` up front, so SROA + ``mem2reg`` can promote slots independently and the optimiser can spill
only the ones it has to. The ergonomic indexed-access syntax is preserved at the source level.

Public:
- ``UnpackedVector``        - subscriptable type: ``r: UnpackedVector[qd.f32, 32]``

Internal (used by ``StructType`` and the AST transformer):
- ``_expand_unpacked_vector_naming(group, i)`` - synthetic-field naming convention
- ``_UnpackedVectorRef``                       - transient proxy yielded by attribute access

This module has no dependency on ``struct.py``; ``struct.py`` imports from here.
"""

import numpy as np

from quadrants.lang.exception import QuadrantsSyntaxError


class UnpackedVector:
    """Subscriptable type for a group of N scalar fields exposed via indexed syntax on a ``@qd.dataclass``.

    Use as ``r: UnpackedVector[dtype, count]`` -- the subscript expression returns the layout marker that
    ``StructType.__init__`` consumes at struct-definition time to lay out the N synthetic scalar fields. The marker
    itself stores only ``count`` and ``dtype``.

    The subscript form (vs. a function-call factory) is chosen so the spelling at a use site visually reads as a type
    annotation, not a value-construction. Subscripting / calling a marker instance directly raises
    ``QuadrantsSyntaxError`` -- those operations only make sense on the parameterised ``Struct`` field, not on the
    annotation marker.
    """

    def __init__(self, count, dtype):
        if not isinstance(count, int) or count <= 0:
            raise QuadrantsSyntaxError(f"UnpackedVector count must be a positive int, got {count!r}")
        self.count = count
        self.dtype = dtype

    def __class_getitem__(cls, params):
        # ``UnpackedVector[dtype, count]`` -> marker instance. ``params`` is a 2-tuple from python's subscript protocol.
        if not isinstance(params, tuple) or len(params) != 2:
            raise QuadrantsSyntaxError(
                "UnpackedVector must be parameterised as UnpackedVector[dtype, count] " f"(got {params!r})"
            )
        dtype, count = params
        if not isinstance(count, int):
            raise QuadrantsSyntaxError(
                "UnpackedVector[dtype, count] requires count to be a python int "
                f"(got count={count!r}, type {type(count).__name__}); "
                "did you mean UnpackedVector[dtype, N] with the dtype first?"
            )
        return cls(count, dtype)

    def __repr__(self):
        return f"UnpackedVector[{self.dtype}, {self.count}]"

    # ----- misuse guards ----------------------------------------------------------------------------------------------
    # An ``UnpackedVector`` marker instance is consumed by ``@qd.dataclass`` (which only reads ``.count`` / ``.dtype``).
    # Any other operation on the instance is misuse -- typically the caller forgot to put the type in a ``@qd.dataclass``
    # annotation, or expects the marker to be a runtime container. Catch the most likely follow-ups with a clear error.

    def __getitem__(self, _index):
        raise QuadrantsSyntaxError(
            "UnpackedVector[dtype, count] is a @qd.dataclass field annotation, not a runtime container -- "
            "an already-parameterised marker has no values to subscript. If obj.r[i] raised this, "
            "the class declaring `r` was probably decorated with @dataclasses.dataclass (or nothing) instead of "
            "@qd.dataclass."
        )

    def __call__(self, *_args, **_kwargs):
        raise QuadrantsSyntaxError(
            "UnpackedVector[dtype, count] is a @qd.dataclass field annotation, not a constructor. "
            "Use it as `r: UnpackedVector[dtype, count]` inside a class decorated with @qd.dataclass; "
            "do not call the parameterised marker."
        )


def _expand_unpacked_vector_naming(group_name, index):
    """Naming convention for the synthetic scalar fields of an ``UnpackedVector`` group.

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
                f"UnpackedVector {self._qd_group_name}[i] requires a python-int index "
                f"(possibly via qd.static); got runtime index of type {type(index).__name__}"
            )
        i = int(index)
        if i < 0 or i >= self._qd_count:
            raise QuadrantsSyntaxError(
                f"UnpackedVector index out of bounds: {self._qd_group_name}[{i}] (count={self._qd_count})"
            )
        field_name = self._qd_naming_fn(self._qd_group_name, i)
        return getattr(self._qd_struct, field_name)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"<UnpackedVector_ref group={self._qd_group_name!r} " f"count={self._qd_count} dtype={self._qd_dtype}>"


__all__ = ["UnpackedVector"]
